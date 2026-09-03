"""Knowledge-graph algorithms.

Pure functions over an immutable in-memory snapshot. Nothing here touches the
database, the LLM, or HTTP — which is the point: prerequisite reasoning is the
foundation everything else stands on, so it is the part that must be exhaustively
testable in microseconds.

**Edge direction.** Two relation types express ordering, and they point opposite ways:

* ``PREREQUISITE_OF``: ``a -> b`` means *a is a prerequisite of b* (learn a first).
* ``DEPENDS_ON``: ``a -> b`` means *a depends on b* (also learn b first).

Both are normalised on load into a single canonical form — a ``requires`` map from
each concept to what it needs — so no algorithm below has to remember which way an
arrow points. Associative relations (``RELATED_TO``, ``BUILDS_ON``,
``ALTERNATIVE_TO``) are kept separately and are allowed to form cycles.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final
from uuid import UUID

from pathwise.api.errors import CycleError, NotFoundError
from pathwise.models.enums import ORDERING_RELATIONS, RelationType

# Baseline mastery a learner needs in a full-strength prerequisite before the
# dependent concept is considered unlocked. Scaled by edge strength, so a 0.5-strength
# "helpful but not required" edge only demands 0.30.
PREREQ_SATISFACTION_THRESHOLD: Final = 0.60

# How much blame attenuates per hop away from the failed concept. A direct
# prerequisite is a far more likely culprit than its grandparent, but the
# grandparent is not exonerated.
BLAME_HOP_DECAY: Final = 0.60

# How much an accusation is discounted when it rests on no evidence.
#
# An unmeasured prerequisite still deserves suspicion — it may well be the gap — but
# "we never tested you on this" is a weaker basis for spending someone's hours than
# "we measured you at 0.2". Without this discount the two are inverted: an unmeasured
# concept scores a full 1.0 deficit and beats every concept actually observed to be
# weak, so the system reliably sends learners to the topic it knows least about.
#
# The floor keeps unmeasured concepts in contention when nothing better exists, which
# is the common case for a learner who has only just started.
BLAME_EVIDENCE_FLOOR: Final = 0.40

MAX_TRAVERSAL_DEPTH: Final = 12


@dataclass(frozen=True, slots=True)
class GraphNode:
    """A concept, reduced to what graph algorithms actually need."""

    id: UUID
    slug: str
    name: str
    difficulty: int = 3
    estimated_minutes: int = 120
    domain: str = ""
    #: Carried so prompts can be built from graph facts alone. Empty is valid —
    #: a subgraph or a test fixture need not supply one.
    description: str = ""
    #: Learning-objective ids declared by this concept. Assessment questions bind to
    #: these, which is what turns a score into evidence about a specific capability
    #: rather than a vague number about a whole topic.
    objective_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """A typed, weighted edge as stored."""

    source: UUID
    target: UUID
    relation: RelationType
    strength: float = 1.0


@dataclass(frozen=True, slots=True)
class Requirement:
    """A concept required by another, with how far away and how hard the demand is.

    ``strength`` along a multi-hop path is the *product* of the edge strengths: a
    weak link anywhere makes the whole chain weak, which is the behaviour you want
    when deciding whether to hold a learner back.
    """

    concept_id: UUID
    hops: int
    strength: float

    @property
    def required_mastery(self) -> float:
        """The mastery level this requirement demands."""
        return PREREQ_SATISFACTION_THRESHOLD * self.strength


@dataclass(frozen=True, slots=True)
class BlameCandidate:
    """A prerequisite suspected of causing difficulty with a concept.

    ``score`` is comparable only within one blame query. ``deficit`` and ``hops`` are
    the human-readable justification, and they are what the explanation prompt is
    given — the prose must be derived from these numbers, never invented alongside
    them.
    """

    concept_id: UUID
    score: float
    deficit: float
    mastery: float
    confidence: float
    hops: int
    strength: float


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Whether a learner may start a concept, and what is blocking them if not."""

    concept_id: UUID
    is_unlocked: bool
    unmet: tuple[Requirement, ...] = ()
    # Smallest gap between actual and required mastery across direct prerequisites.
    # Negative means blocked; a large positive value means comfortably over-prepared
    # and is what triggers a COMPRESS/SKIP proposal.
    margin: float = 0.0


class KnowledgeGraph:
    """An immutable adjacency snapshot of the concept graph.

    Built once per request (or cached in Redis and rebuilt on graph writes) and then
    queried many times. Construction normalises edge direction and validates the DAG
    invariant on ordering edges; every method afterwards is a read.
    """

    __slots__ = ("_associative", "_dependents", "_nodes", "_requires", "_slugs")

    def __init__(
        self,
        nodes: Iterable[GraphNode],
        edges: Iterable[GraphEdge],
        *,
        validate_acyclic: bool = True,
    ) -> None:
        self._nodes: dict[UUID, GraphNode] = {node.id: node for node in nodes}
        self._slugs: dict[str, UUID] = {node.slug: node.id for node in self._nodes.values()}

        # canonical form: requires[dependent] = {requirement_id: strength}
        requires: dict[UUID, dict[UUID, float]] = {}
        dependents: dict[UUID, dict[UUID, float]] = {}
        associative: dict[UUID, list[GraphEdge]] = {}

        for edge in edges:
            if edge.source not in self._nodes or edge.target not in self._nodes:
                # Edges to concepts outside this snapshot (a filtered subgraph) are
                # dropped rather than dangling.
                continue

            if edge.relation is RelationType.PREREQUISITE_OF:
                dependent, requirement = edge.target, edge.source
            elif edge.relation is RelationType.DEPENDS_ON:
                dependent, requirement = edge.source, edge.target
            else:
                associative.setdefault(edge.source, []).append(edge)
                associative.setdefault(edge.target, []).append(edge)
                continue

            # Two relation types can express the same ordering; keep the stronger.
            existing = requires.setdefault(dependent, {}).get(requirement, 0.0)
            strength = max(existing, edge.strength)
            requires[dependent][requirement] = strength
            dependents.setdefault(requirement, {})[dependent] = strength

        self._requires: dict[UUID, dict[UUID, float]] = requires
        self._dependents: dict[UUID, dict[UUID, float]] = dependents
        self._associative: dict[UUID, list[GraphEdge]] = associative

        if validate_acyclic:
            self._assert_acyclic()

    # --- construction helpers ------------------------------------------------- #

    @classmethod
    def empty(cls) -> KnowledgeGraph:
        return cls(nodes=(), edges=())

    def _assert_acyclic(self) -> None:
        """Raise if the ordering edges contain a cycle.

        Enforced at construction rather than at query time so a corrupt graph fails
        loudly at load, not silently in the middle of building someone's roadmap.
        """
        try:
            self.topological_order()
        except CycleError:
            raise

    # --- basic accessors ------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, concept_id: object) -> bool:
        return concept_id in self._nodes

    @property
    def node_ids(self) -> frozenset[UUID]:
        return frozenset(self._nodes)

    def node(self, concept_id: UUID) -> GraphNode:
        try:
            return self._nodes[concept_id]
        except KeyError:
            raise NotFoundError(
                "Concept is not present in this graph snapshot.", concept_id=str(concept_id)
            ) from None

    def by_slug(self, slug: str) -> GraphNode:
        try:
            return self._nodes[self._slugs[slug]]
        except KeyError:
            raise NotFoundError("No concept with that slug.", slug=slug) from None

    def direct_requirements(self, concept_id: UUID) -> tuple[Requirement, ...]:
        """Immediate prerequisites of a concept, strongest first."""
        self.node(concept_id)
        items = self._requires.get(concept_id, {})
        return tuple(
            sorted(
                (Requirement(cid, hops=1, strength=s) for cid, s in items.items()),
                key=lambda r: (-r.strength, str(r.concept_id)),
            )
        )

    def direct_dependents(self, concept_id: UUID) -> tuple[UUID, ...]:
        """Concepts that list this one as a prerequisite."""
        self.node(concept_id)
        return tuple(sorted(self._dependents.get(concept_id, {}), key=str))

    def related(self, concept_id: UUID) -> tuple[GraphEdge, ...]:
        """Associative (non-ordering) edges touching this concept."""
        self.node(concept_id)
        return tuple(self._associative.get(concept_id, ()))

    # --- traversal ------------------------------------------------------------ #

    def prerequisite_closure(
        self, concept_id: UUID, *, max_depth: int = MAX_TRAVERSAL_DEPTH
    ) -> dict[UUID, Requirement]:
        """Every concept transitively required by ``concept_id``.

        Breadth-first, so the recorded ``hops`` is the *shortest* path to each
        requirement. When a concept is reachable by several paths, the strongest
        (highest product of edge strengths) wins, because the hardest demand on a
        learner is the one that governs.
        """
        self.node(concept_id)
        found: dict[UUID, Requirement] = {}
        queue: deque[tuple[UUID, int, float]] = deque([(concept_id, 0, 1.0)])

        while queue:
            current, depth, path_strength = queue.popleft()
            if depth >= max_depth:
                continue

            for requirement_id, edge_strength in self._requires.get(current, {}).items():
                if requirement_id == concept_id:
                    continue  # a cycle back to the origin; the DAG check reports it
                next_strength = path_strength * edge_strength
                previous = found.get(requirement_id)

                if previous is None:
                    found[requirement_id] = Requirement(
                        requirement_id, hops=depth + 1, strength=next_strength
                    )
                    queue.append((requirement_id, depth + 1, next_strength))
                elif next_strength > previous.strength:
                    # A stronger route to a requirement we already knew about.
                    found[requirement_id] = Requirement(
                        requirement_id, hops=previous.hops, strength=next_strength
                    )
                    queue.append((requirement_id, depth + 1, next_strength))

        return found

    def dependent_closure(
        self, concept_id: UUID, *, max_depth: int = MAX_TRAVERSAL_DEPTH
    ) -> dict[UUID, int]:
        """Every concept that transitively requires this one, with hop distance.

        Answers "what does this unlock?" — used to weight a concept's importance
        towards a goal and to show the learner what a weak spot is holding up.
        """
        self.node(concept_id)
        found: dict[UUID, int] = {}
        queue: deque[tuple[UUID, int]] = deque([(concept_id, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for dependent_id in self._dependents.get(current, {}):
                if dependent_id == concept_id or dependent_id in found:
                    continue
                found[dependent_id] = depth + 1
                queue.append((dependent_id, depth + 1))

        return found

    def topological_order(self, subset: Iterable[UUID] | None = None) -> tuple[UUID, ...]:
        """Kahn's algorithm: prerequisites before dependents.

        Ties are broken by concept id so the ordering is deterministic — two runs
        over the same graph must produce the same roadmap, or nothing downstream is
        reproducible.

        Raises:
            CycleError: if the ordering edges contain a cycle.
        """
        included = set(self._nodes) if subset is None else {c for c in subset if c in self._nodes}

        in_degree: dict[UUID, int] = {}
        for concept_id in included:
            requirements = self._requires.get(concept_id, {})
            in_degree[concept_id] = sum(1 for r in requirements if r in included)

        ready = deque(sorted((c for c, d in in_degree.items() if d == 0), key=str))
        ordered: list[UUID] = []

        while ready:
            current = ready.popleft()
            ordered.append(current)
            newly_ready: list[UUID] = []
            for dependent_id in self._dependents.get(current, {}):
                if dependent_id not in in_degree:
                    continue
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    newly_ready.append(dependent_id)
            # Sort each newly-freed batch, not the whole frontier, to keep the
            # ordering stable without an O(n log n) re-sort per step.
            ready.extend(sorted(newly_ready, key=str))

        if len(ordered) != len(included):
            remaining = sorted(included - set(ordered), key=str)
            raise CycleError(
                "Prerequisite edges contain a cycle; the knowledge graph must be a DAG.",
                concept_ids=[str(c) for c in remaining[:10]],
                cycle_size=len(remaining),
            )

        return tuple(ordered)

    def would_create_cycle(self, source_id: UUID, target_id: UUID, relation: RelationType) -> bool:
        """Whether adding this edge would make the graph cyclic.

        Called before every ordering-edge insert — seed data, LLM proposals, and user
        edits alike. Cheaper than inserting and re-validating: it is a single reachability
        query, not a full topological sort.
        """
        if relation not in ORDERING_RELATIONS:
            return False
        if source_id == target_id:
            return True

        if relation is RelationType.PREREQUISITE_OF:
            dependent, requirement = target_id, source_id
        else:
            dependent, requirement = source_id, target_id

        if dependent not in self._nodes or requirement not in self._nodes:
            return False

        # A cycle appears exactly when the new requirement already (transitively)
        # requires the dependent.
        return dependent in self.prerequisite_closure(requirement)

    def distance_to_goals(self, goal_ids: Iterable[UUID]) -> dict[UUID, int]:
        """Hops from each concept to the nearest goal concept it feeds into.

        This is the goal-relevance signal in the decision engine: a concept two hops
        from the target matters more than one twelve hops away, and a concept that
        reaches no goal at all is not on the path and scores nothing.
        """
        goals = [g for g in goal_ids if g in self._nodes]
        distance: dict[UUID, int] = dict.fromkeys(goals, 0)
        queue: deque[UUID] = deque(goals)

        while queue:
            current = queue.popleft()
            for requirement_id in self._requires.get(current, {}):
                if requirement_id not in distance:
                    distance[requirement_id] = distance[current] + 1
                    queue.append(requirement_id)

        return distance

    # --- learner-relative queries --------------------------------------------- #

    def readiness(
        self,
        concept_id: UUID,
        mastery: Mapping[UUID, float],
        *,
        default_mastery: float = 0.0,
    ) -> ReadinessReport:
        """Whether the learner's direct prerequisites clear their required levels.

        Deliberately checks *direct* prerequisites only. Transitive requirements are
        already reflected in the direct ones' mastery scores; re-checking them would
        punish a learner twice for the same gap.
        """
        unmet: list[Requirement] = []
        margins: list[float] = []

        for requirement in self.direct_requirements(concept_id):
            actual = mastery.get(requirement.concept_id, default_mastery)
            margins.append(actual - requirement.required_mastery)
            if actual < requirement.required_mastery:
                unmet.append(requirement)

        return ReadinessReport(
            concept_id=concept_id,
            is_unlocked=not unmet,
            unmet=tuple(unmet),
            # No prerequisites means maximally ready.
            margin=min(margins) if margins else 1.0,
        )

    def unlocked_frontier(
        self,
        candidates: Iterable[UUID],
        mastery: Mapping[UUID, float],
        *,
        default_mastery: float = 0.0,
    ) -> tuple[UUID, ...]:
        """The subset of ``candidates`` the learner is currently ready to start."""
        return tuple(
            concept_id
            for concept_id in candidates
            if concept_id in self._nodes
            and self.readiness(concept_id, mastery, default_mastery=default_mastery).is_unlocked
        )

    def blame_candidates(
        self,
        concept_id: UUID,
        mastery: Mapping[UUID, float],
        *,
        confidence: Mapping[UUID, float] | None = None,
        max_depth: int = 3,
        limit: int = 5,
        default_mastery: float = 0.0,
    ) -> tuple[BlameCandidate, ...]:
        """Rank the prerequisites most likely responsible for difficulty here.

        This is the answer to the spec's "which prerequisite is causing this learner's
        difficulty?" — and it is arithmetic, not a prompt. Three factors multiply:

        1. **deficit** — how far below its required level the prerequisite sits,
           normalised so a weak edge cannot produce a large deficit.
        2. **strength** — how hard the requirement is. A 0.4-strength edge yields a
           weak accusation even when mastery is zero.
        3. **hop decay** — a direct prerequisite outranks a distant ancestor.

        Confidence scales the result rather than boosting it: a prerequisite we
        have measured at 0.2 is a better-supported accusation than one we simply
        never tested, and inverting that sends learners to whatever the system
        happens to know least about.

        Returns an empty tuple when every prerequisite is comfortably met — in which
        case the difficulty is with *this* concept, not beneath it, and the caller
        should add practice rather than remediation.
        """
        confidence = confidence or {}
        candidates: list[BlameCandidate] = []

        for requirement in self.prerequisite_closure(concept_id, max_depth=max_depth).values():
            required = requirement.required_mastery
            if required <= 0:
                continue

            actual = mastery.get(requirement.concept_id, default_mastery)
            deficit = max(0.0, required - actual) / required
            if deficit <= 0:
                continue

            certainty = confidence.get(requirement.concept_id, 0.0)
            # Scale by how well-evidenced the accusation is, never by more than 1.
            evidence_weight = BLAME_EVIDENCE_FLOOR + (1.0 - BLAME_EVIDENCE_FLOOR) * certainty
            score = (
                deficit
                * requirement.strength
                * (BLAME_HOP_DECAY ** (requirement.hops - 1))
                * evidence_weight
            )

            candidates.append(
                BlameCandidate(
                    concept_id=requirement.concept_id,
                    score=score,
                    deficit=deficit,
                    mastery=actual,
                    confidence=certainty,
                    hops=requirement.hops,
                    strength=requirement.strength,
                )
            )

        candidates.sort(key=lambda c: (-c.score, c.hops, str(c.concept_id)))
        return tuple(candidates[:limit])

    def subgraph(self, concept_ids: Iterable[UUID]) -> KnowledgeGraph:
        """A new snapshot restricted to the given concepts.

        Edges with an endpoint outside the selection are dropped, which is what makes
        a roadmap's edge projection well-defined.
        """
        keep = {c for c in concept_ids if c in self._nodes}
        nodes = [self._nodes[c] for c in keep]
        edges = [
            GraphEdge(
                source=requirement_id,
                target=dependent_id,
                relation=RelationType.PREREQUISITE_OF,
                strength=strength,
            )
            for dependent_id, requirements in self._requires.items()
            if dependent_id in keep
            for requirement_id, strength in requirements.items()
            if requirement_id in keep
        ]
        # Already known acyclic: a subgraph of a DAG cannot contain a cycle.
        return KnowledgeGraph(nodes, edges, validate_acyclic=False)


@dataclass(slots=True)
class GraphBuilder:
    """Incremental construction with cycle rejection, for seeding and LLM proposals.

    Every ordering edge is checked against the graph built so far, so a bad edge is
    rejected at the point it is added — with the offending pair named — instead of
    surfacing as an opaque failure once the whole batch is loaded.
    """

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> GraphBuilder:
        self.nodes.append(node)
        return self

    def add_edge(self, edge: GraphEdge) -> GraphBuilder:
        """Add an edge, rejecting it if it would introduce a cycle."""
        if edge.relation in ORDERING_RELATIONS:
            current = KnowledgeGraph(self.nodes, self.edges, validate_acyclic=False)
            if current.would_create_cycle(edge.source, edge.target, edge.relation):
                raise CycleError(
                    "Edge rejected: it would introduce a prerequisite cycle.",
                    source_id=str(edge.source),
                    target_id=str(edge.target),
                    relation=str(edge.relation),
                )
        self.edges.append(edge)
        return self

    def build(self) -> KnowledgeGraph:
        return KnowledgeGraph(self.nodes, self.edges)


def build_graph(nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]) -> KnowledgeGraph:
    """Convenience constructor with DAG validation."""
    return KnowledgeGraph(nodes, edges)
