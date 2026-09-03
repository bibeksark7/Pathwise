"""Deterministic roadmap construction.

This module decides **what is in a roadmap and in what order**. No LLM is involved,
and that is the central design claim of the whole project: given the same goal, the
same graph, and the same mastery state, it produces the same roadmap every time, and
every inclusion, exclusion, and ordering decision is traceable to an algorithm rather
than to a generation.

The pipeline:

1. **Scope** — transitive prerequisite closure of the goal concepts. Graph traversal
   answers "what do I need to know for this", so a model cannot omit a prerequisite
   or pad the path with something unrelated.
2. **Reduce** — drop concepts the learner has already demonstrated. This is what lets
   Pathwise *shorten* a path rather than only lengthen it, and it demands both high
   mastery and enough evidence to trust it.
3. **Sequence** — topological order, so prerequisites always precede dependents.
4. **Pace** — total the time, compare against the learner's weekly hours and deadline,
   and report feasibility.
5. **Project edges** — keep only prerequisite edges whose endpoints are both in the
   roadmap, giving the frontend a graph it can lay out without loading the whole
   knowledge base.

The LLM's contribution comes afterwards and is purely linguistic: a title, and prose
explaining steps this module has already chosen.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from uuid import UUID

from pathwise.api.errors import ValidationError
from pathwise.models.enums import NodeStatus, NodeType
from pathwise.services.knowledge.graph import KnowledgeGraph
from pathwise.services.knowledge.mastery import MasteryEstimate

#: A roadmap larger than this is not a plan, it is a syllabus. Beyond roughly this
#: many steps a learner cannot see the path, so the goal needs narrowing instead.
MAX_ROADMAP_NODES = 60

#: Weeks of slack expected before a deadline is called comfortable.
DEADLINE_BUFFER_WEEKS = 1.0

#: Mastery at which a concept becomes a review rather than a full study step.
#:
#: Calibrated against what a single correct diagnostic answer can actually produce.
#: From a uniform prior one full-weight success yields a posterior mean of 0.75, so a
#: bar set at the 0.85 skip threshold would be unreachable and no diagnostic could
#: ever shorten a roadmap. The distinction between compressing and skipping is
#: therefore *confidence*, not level: one demonstration earns a shorter step,
#: repeated demonstration earns removal.
COMPRESSION_MASTERY_THRESHOLD = 0.70

#: Fraction of the original time a compressed step keeps.
COMPRESSION_RATIO = 0.3

#: Floor for a review step. Below this it is not a review, it is a formality.
MIN_REVIEW_MINUTES = 15


@dataclass(frozen=True, slots=True)
class PlannedNode:
    """One step in a planned roadmap, before persistence."""

    concept_id: UUID
    slug: str
    name: str
    order_index: int
    estimated_minutes: int
    difficulty: int
    domain: str
    node_type: NodeType = NodeType.TOPIC
    status: NodeStatus = NodeStatus.NOT_STARTED
    #: Direct prerequisites that are also in this roadmap. Drives the locked state.
    depends_on: tuple[UUID, ...] = ()


@dataclass(frozen=True, slots=True)
class SkippedConcept:
    """A prerequisite left out because the learner already demonstrated it.

    Kept in the plan rather than silently dropped: "you already know this, so I
    removed it" is one of the most valuable things the product can say, and it is
    only sayable if the exclusion is recorded with its evidence.
    """

    concept_id: UUID
    slug: str
    name: str
    mastery: float
    confidence: float
    evidence_count: int


@dataclass(frozen=True, slots=True)
class CompressedConcept:
    """A concept kept as a short review rather than a full study step.

    Sits between "study this properly" and "you already know this". A diagnostic can
    earn compression on a single well-aimed question; skipping needs sustained
    evidence. Recording both figures makes the decision explainable — and reversible,
    since a failed review restores the full step.
    """

    concept_id: UUID
    slug: str
    name: str
    mastery: float
    confidence: float
    original_minutes: int
    review_minutes: int

    @property
    def minutes_saved(self) -> int:
        return self.original_minutes - self.review_minutes


@dataclass(frozen=True, slots=True)
class Pacing:
    """How the plan fits the learner's available time."""

    total_minutes: int
    hours_per_week: float
    estimated_weeks: float
    deadline: date | None = None
    weeks_available: float | None = None

    @property
    def meets_deadline(self) -> bool | None:
        """``None`` when no deadline was set — not the same as "yes"."""
        if self.weeks_available is None:
            return None
        return self.estimated_weeks <= self.weeks_available

    @property
    def is_comfortable(self) -> bool | None:
        """Whether there is slack, not merely a bare fit."""
        if self.weeks_available is None:
            return None
        return self.estimated_weeks + DEADLINE_BUFFER_WEEKS <= self.weeks_available

    @property
    def weeks_over(self) -> float:
        """How far past the deadline the plan runs. Zero when it fits."""
        if self.weeks_available is None:
            return 0.0
        return max(0.0, self.estimated_weeks - self.weeks_available)

    @property
    def required_hours_per_week(self) -> float | None:
        """Weekly hours that would actually meet the deadline.

        More useful than "you are 10 weeks over": it converts an abstract shortfall
        into the one number the learner can act on. ``None`` when there is no
        deadline, and unbounded when the deadline has already passed.
        """
        if self.weeks_available is None:
            return None
        if self.weeks_available <= 0:
            return float("inf")
        return round((self.total_minutes / 60.0) / self.weeks_available, 1)


@dataclass(frozen=True, slots=True)
class ScopeTrace:
    """Why the roadmap contains what it contains.

    The structural counterpart to the decision engine's trace: it makes the plan
    auditable, and it is the only material the explanation prompt is given, so the
    prose cannot claim a reason the algorithm did not have.
    """

    goal_concept_ids: tuple[UUID, ...]
    closure_size: int
    skipped_count: int
    included_count: int
    trimmed_count: int = 0

    @property
    def reduction_ratio(self) -> float:
        """Share of the required material the learner was excused from."""
        if self.closure_size == 0:
            return 0.0
        return self.skipped_count / self.closure_size


@dataclass(frozen=True, slots=True)
class RoadmapPlan:
    """A complete, ordered plan. Pure data — nothing here has been persisted."""

    nodes: tuple[PlannedNode, ...]
    edges: tuple[tuple[UUID, UUID, float], ...]
    skipped: tuple[SkippedConcept, ...]
    #: Steps kept but shortened to a review, on evidence too thin to skip outright.
    compressed: tuple[CompressedConcept, ...]
    pacing: Pacing
    scope: ScopeTrace
    #: Steps reachable only through weak ("helpful, not required") prerequisite
    #: edges — the genuinely droppable material if the deadline cannot be met.
    #: Identified, never applied: silently deleting material the learner asked for
    #: would be a worse failure than telling them the plan does not fit.
    optional_steps: tuple[PlannedNode, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def concept_ids(self) -> tuple[UUID, ...]:
        return tuple(node.concept_id for node in self.nodes)

    @property
    def slugs(self) -> tuple[str, ...]:
        return tuple(node.slug for node in self.nodes)

    @property
    def is_empty(self) -> bool:
        return not self.nodes

    def node_for(self, concept_id: UUID) -> PlannedNode | None:
        return next((n for n in self.nodes if n.concept_id == concept_id), None)


def plan_roadmap(
    graph: KnowledgeGraph,
    goal_concept_ids: Iterable[UUID],
    mastery: Mapping[UUID, MasteryEstimate] | None = None,
    *,
    hours_per_week: float = 5.0,
    deadline: date | None = None,
    now: datetime | None = None,
    max_nodes: int = MAX_ROADMAP_NODES,
) -> RoadmapPlan:
    """Build a roadmap from a goal, a graph, and what the learner already knows.

    Args:
        graph: The knowledge graph snapshot to plan over.
        goal_concept_ids: What the learner is aiming at.
        mastery: Current estimates. Absent concepts are treated as unknown, which is
            correct — no evidence is not the same as evidence of nothing.
        hours_per_week: Study time available, for pacing.
        deadline: Optional target date.
        now: Reference time for mastery decay. Injected rather than read so planning
            is reproducible.
        max_nodes: Guard against a goal so broad the plan is unusable.

    Raises:
        ValidationError: if no goal concept exists in the graph, or the resulting
            plan would exceed ``max_nodes``.
    """
    mastery = mastery or {}
    now = now or _utcnow()

    goals = tuple(cid for cid in goal_concept_ids if cid in graph)
    if not goals:
        raise ValidationError(
            "None of the goal concepts exist in the knowledge graph.",
            goal_concept_ids=[str(c) for c in goal_concept_ids],
        )

    # 1. Scope — graph traversal decides what is required, not a model.
    required: set[UUID] = set(goals)
    for goal in goals:
        required.update(graph.prerequisite_closure(goal))

    # 2. Reduce — the spec's "reduce or skip", which are two different things.
    #
    #    SKIP removes a concept entirely, and demands both high mastery and enough
    #    evidence to trust it. A goal concept is never skipped: it is the thing they
    #    asked to learn, and prior competence should shorten the path to it, not
    #    delete the point.
    #
    #    COMPRESS keeps the concept but shortens it to a review, and is what a
    #    diagnostic can earn on its own. One well-aimed question is real evidence
    #    that someone knows a topic, but it is not enough to delete it — that is how
    #    a learner ends up stranded three steps later. Compression is the honest
    #    middle: acknowledge the demonstration, spend a fraction of the time.
    included: list[UUID] = []
    skipped: list[SkippedConcept] = []
    compressed: dict[UUID, CompressedConcept] = {}

    for concept_id in required:
        estimate = mastery.get(concept_id)
        is_goal = concept_id in goals

        if estimate is None:
            included.append(concept_id)
            continue

        node = graph.node(concept_id)

        if not is_goal and estimate.is_skippable_at(now):
            skipped.append(
                SkippedConcept(
                    concept_id=concept_id,
                    slug=node.slug,
                    name=node.name,
                    mastery=estimate.effective_mastery(now),
                    confidence=estimate.confidence,
                    evidence_count=estimate.evidence_count,
                )
            )
            continue

        if not is_goal and estimate.effective_mastery(now) >= COMPRESSION_MASTERY_THRESHOLD:
            compressed[concept_id] = CompressedConcept(
                concept_id=concept_id,
                slug=node.slug,
                name=node.name,
                mastery=estimate.effective_mastery(now),
                confidence=estimate.confidence,
                original_minutes=node.estimated_minutes,
                review_minutes=max(
                    MIN_REVIEW_MINUTES,
                    int(node.estimated_minutes * COMPRESSION_RATIO),
                ),
            )

        included.append(concept_id)

    if len(included) > max_nodes:
        raise ValidationError(
            "This goal expands into more steps than a roadmap can usefully hold. "
            "Narrow the goal, or split it into stages.",
            required=len(included),
            maximum=max_nodes,
        )

    # 3. Sequence — prerequisites before dependents, deterministically.
    ordered = graph.subgraph(included).topological_order()

    # 4. Build the nodes, with locked state derived from what precedes them.
    effective = {cid: est.effective_mastery(now) for cid, est in mastery.items()}
    included_set = set(ordered)
    nodes = tuple(
        _build_node(graph, concept_id, index, included_set, effective, compressed)
        for index, concept_id in enumerate(ordered)
    )

    # 5. Project the edges that survive the filtering.
    edges = tuple(
        (requirement.concept_id, concept_id, requirement.strength)
        for concept_id in ordered
        for requirement in graph.direct_requirements(concept_id)
        if requirement.concept_id in included_set
    )

    pacing = _compute_pacing(nodes, hours_per_week, deadline, now)
    scope = ScopeTrace(
        goal_concept_ids=goals,
        closure_size=len(required),
        skipped_count=len(skipped),
        included_count=len(nodes),
    )

    optional = _find_optional_steps(nodes, edges, goals)
    warnings = _collect_warnings(pacing, skipped, nodes, optional)

    return RoadmapPlan(
        nodes=nodes,
        edges=edges,
        skipped=tuple(sorted(skipped, key=lambda s: s.slug)),
        compressed=tuple(sorted(compressed.values(), key=lambda c: c.slug)),
        pacing=pacing,
        scope=scope,
        optional_steps=optional,
        warnings=warnings,
    )


def _build_node(
    graph: KnowledgeGraph,
    concept_id: UUID,
    index: int,
    included: set[UUID],
    effective_mastery: Mapping[UUID, float],
    compressed: Mapping[UUID, CompressedConcept] | None = None,
) -> PlannedNode:
    """One roadmap step, with its status and type derived rather than assigned."""
    node = graph.node(concept_id)
    compression = (compressed or {}).get(concept_id)
    depends_on = tuple(
        requirement.concept_id
        for requirement in graph.direct_requirements(concept_id)
        if requirement.concept_id in included
    )

    readiness = graph.readiness(concept_id, effective_mastery)
    status = NodeStatus.NOT_STARTED if readiness.is_unlocked else NodeStatus.LOCKED

    return PlannedNode(
        concept_id=concept_id,
        slug=node.slug,
        name=node.name,
        order_index=index,
        # A compressed step keeps its place in the sequence but costs a fraction of
        # the time, so pacing reflects the reduction automatically.
        estimated_minutes=(compression.review_minutes if compression else node.estimated_minutes),
        difficulty=node.difficulty,
        domain=node.domain,
        node_type=NodeType.REVIEW if compression else NodeType.TOPIC,
        status=status,
        depends_on=depends_on,
    )


def _compute_pacing(
    nodes: Sequence[PlannedNode],
    hours_per_week: float,
    deadline: date | None,
    now: datetime,
) -> Pacing:
    total_minutes = sum(node.estimated_minutes for node in nodes)
    safe_hours = max(hours_per_week, 0.5)  # a zero would divide by zero
    estimated_weeks = (total_minutes / 60.0) / safe_hours

    weeks_available: float | None = None
    if deadline is not None:
        days_remaining = (deadline - now.date()).days
        weeks_available = max(0.0, days_remaining / 7.0)

    return Pacing(
        total_minutes=total_minutes,
        hours_per_week=hours_per_week,
        estimated_weeks=round(estimated_weeks, 1),
        deadline=deadline,
        weeks_available=round(weeks_available, 1) if weeks_available is not None else None,
    )


#: Below this, a prerequisite edge means "this helps" rather than "you cannot
#: proceed without it". Concepts reachable only across such edges are the honestly
#: droppable material.
OPTIONAL_EDGE_STRENGTH = 0.8


def _find_optional_steps(
    nodes: Sequence[PlannedNode],
    edges: Sequence[tuple[UUID, UUID, float]],
    goals: Sequence[UUID],
) -> tuple[PlannedNode, ...]:
    """Steps that are helpful rather than strictly required.

    A concept is optional when *every* route from it to a goal crosses at least one
    weak edge. If any all-strong path exists, dropping it would break something the
    learner actually needs.

    The distinction matters because an earlier version of this simply offered the
    cheapest leaf nodes when a deadline could not be met. In a single-goal closure
    every concept is genuinely required, so that advice was always wrong — it would
    have told a learner to skip prerequisites they could not proceed without.
    """
    goal_set = set(goals)
    strong_out: dict[UUID, list[UUID]] = {}
    for source, target, strength in edges:
        if strength >= OPTIONAL_EDGE_STRENGTH:
            strong_out.setdefault(source, []).append(target)

    # Concepts that reach a goal without ever crossing a weak edge are required.
    required: set[UUID] = set(goal_set)
    frontier = list(goal_set)
    seen: set[UUID] = set(goal_set)
    while frontier:
        current = frontier.pop()
        for node in nodes:
            if current in strong_out.get(node.concept_id, ()) and node.concept_id not in seen:
                seen.add(node.concept_id)
                required.add(node.concept_id)
                frontier.append(node.concept_id)

    optional = [node for node in nodes if node.concept_id not in required]
    optional.sort(key=lambda n: (-n.estimated_minutes, n.slug))
    return tuple(optional)


def _collect_warnings(
    pacing: Pacing,
    skipped: Sequence[SkippedConcept],
    nodes: Sequence[PlannedNode],
    optional: Sequence[PlannedNode] = (),
) -> tuple[str, ...]:
    """Honest notes about the plan, shown to the learner rather than buried."""
    warnings: list[str] = []

    if pacing.meets_deadline is False:
        needed = pacing.required_hours_per_week
        # Name the number they can act on, not just the size of the shortfall.
        remedy = (
            f" Meeting it would take about {needed:g} hours per week."
            if needed is not None and needed != float("inf")
            else " That deadline has already passed."
        )
        warnings.append(
            f"At {pacing.hours_per_week:g} hours per week this path takes about "
            f"{pacing.estimated_weeks:g} weeks, which is {pacing.weeks_over:g} weeks "
            f"past your deadline.{remedy}"
        )
        if optional:
            optional_hours = sum(n.estimated_minutes for n in optional) / 60
            warnings.append(
                f"{len(optional)} step(s) totalling about {optional_hours:.0f} hours are "
                "helpful rather than strictly required, and could be dropped to fit."
            )
    elif pacing.is_comfortable is False:
        warnings.append(
            "This plan finishes close to your deadline, with little room for a topic "
            "taking longer than estimated."
        )

    if pacing.estimated_weeks > 52:
        warnings.append(
            "This is over a year of study at your current weekly hours. Consider a "
            "nearer milestone first."
        )

    if not nodes and skipped:
        warnings.append(
            "You already meet every prerequisite for this goal — there is nothing left to schedule."
        )

    return tuple(warnings)


def _utcnow() -> datetime:
    from datetime import UTC

    return datetime.now(UTC)
