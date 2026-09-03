"""Knowledge-graph algorithm tests.

The graph is the foundation: a wrong prerequisite edge produces a wrong roadmap, a
wrong blame attribution, and a wrong next-topic decision. So these tests cover the
invariants (acyclicity, deterministic ordering) as hard requirements rather than
nice-to-haves.

The fixture graph is the spec's own worked example:

    python -> numpy -> linear-algebra -> machine-learning -> neural-networks
    calculus ------------^ (via optimization)
"""

from __future__ import annotations

import uuid
from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pathwise.api.errors import CycleError, NotFoundError
from pathwise.models.enums import RelationType
from pathwise.services.knowledge.graph import (
    PREREQ_SATISFACTION_THRESHOLD,
    GraphBuilder,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
)


def cid(slug: str) -> UUID:
    """A stable UUID per slug, so tests read as concept names rather than hex."""
    return uuid.uuid5(uuid.NAMESPACE_DNS, slug)


SLUGS = (
    "python",
    "numpy",
    "calculus",
    "linear-algebra",
    "optimization",
    "probability",
    "machine-learning",
    "neural-networks",
)


def make_node(slug: str, *, difficulty: int = 3, domain: str = "ml") -> GraphNode:
    return GraphNode(
        id=cid(slug),
        slug=slug,
        name=slug.replace("-", " ").title(),
        difficulty=difficulty,
        domain=domain,
    )


def prereq(source: str, target: str, strength: float = 1.0) -> GraphEdge:
    """`source` is a prerequisite of `target`."""
    return GraphEdge(cid(source), cid(target), RelationType.PREREQUISITE_OF, strength)


@pytest.fixture
def graph() -> KnowledgeGraph:
    nodes = [make_node(slug) for slug in SLUGS]
    edges = [
        prereq("python", "numpy"),
        prereq("numpy", "linear-algebra", 0.8),
        prereq("calculus", "optimization"),
        prereq("linear-algebra", "machine-learning"),
        prereq("optimization", "machine-learning", 0.9),
        prereq("probability", "machine-learning", 0.7),
        prereq("machine-learning", "neural-networks"),
        GraphEdge(cid("numpy"), cid("linear-algebra"), RelationType.RELATED_TO, 0.5),
    ]
    return KnowledgeGraph(nodes, edges)


# --------------------------------------------------------------------------- #
# Edge direction normalisation
# --------------------------------------------------------------------------- #


def test_prerequisite_of_points_from_requirement_to_dependent(graph: KnowledgeGraph) -> None:
    requirement_ids = {r.concept_id for r in graph.direct_requirements(cid("numpy"))}
    assert requirement_ids == {cid("python")}


def test_depends_on_expresses_the_same_ordering_reversed() -> None:
    """`a DEPENDS_ON b` and `b PREREQUISITE_OF a` must produce identical graphs."""
    nodes = [make_node("a"), make_node("b")]
    via_prereq = KnowledgeGraph(nodes, [prereq("b", "a")])
    via_depends = KnowledgeGraph(
        nodes, [GraphEdge(cid("a"), cid("b"), RelationType.DEPENDS_ON, 1.0)]
    )
    assert via_prereq.direct_requirements(cid("a")) == via_depends.direct_requirements(cid("a"))


def test_associative_relations_are_not_ordering_edges() -> None:
    """RELATED_TO must not make anything a prerequisite of anything."""
    nodes = [make_node("a"), make_node("b")]
    graph = KnowledgeGraph(nodes, [GraphEdge(cid("a"), cid("b"), RelationType.RELATED_TO, 0.5)])
    assert graph.direct_requirements(cid("b")) == ()
    assert graph.direct_requirements(cid("a")) == ()
    assert graph.related(cid("a"))[0].relation is RelationType.RELATED_TO


def test_duplicate_orderings_keep_the_stronger_edge() -> None:
    nodes = [make_node("a"), make_node("b")]
    graph = KnowledgeGraph(nodes, [prereq("a", "b", 0.4), prereq("a", "b", 0.9)])
    assert graph.direct_requirements(cid("b"))[0].strength == pytest.approx(0.9)


def test_edges_pointing_outside_the_snapshot_are_dropped() -> None:
    """A filtered subgraph must not carry dangling edges."""
    graph = KnowledgeGraph([make_node("a")], [prereq("missing", "a")])
    assert graph.direct_requirements(cid("a")) == ()


# --------------------------------------------------------------------------- #
# Traversal
# --------------------------------------------------------------------------- #


def test_prerequisite_closure_is_transitive(graph: KnowledgeGraph) -> None:
    closure = graph.prerequisite_closure(cid("neural-networks"))
    assert set(closure) == {
        cid("machine-learning"),
        cid("linear-algebra"),
        cid("optimization"),
        cid("probability"),
        cid("numpy"),
        cid("calculus"),
        cid("python"),
    }


def test_closure_records_shortest_hop_distance(graph: KnowledgeGraph) -> None:
    closure = graph.prerequisite_closure(cid("neural-networks"))
    assert closure[cid("machine-learning")].hops == 1
    assert closure[cid("linear-algebra")].hops == 2
    assert closure[cid("numpy")].hops == 3
    assert closure[cid("python")].hops == 4


def test_path_strength_is_the_product_of_edge_strengths(graph: KnowledgeGraph) -> None:
    """A chain is only as strong as the product of its links."""
    closure = graph.prerequisite_closure(cid("neural-networks"))
    # neural-networks <- machine-learning (1.0) <- linear-algebra (1.0) <- numpy (0.8)
    assert closure[cid("numpy")].strength == pytest.approx(0.8)
    # ... <- python (1.0), so the 0.8 carries through
    assert closure[cid("python")].strength == pytest.approx(0.8)


def test_closure_respects_max_depth(graph: KnowledgeGraph) -> None:
    closure = graph.prerequisite_closure(cid("neural-networks"), max_depth=2)
    assert cid("linear-algebra") in closure
    assert cid("numpy") not in closure


def test_dependent_closure_answers_what_this_unlocks(graph: KnowledgeGraph) -> None:
    dependents = graph.dependent_closure(cid("python"))
    assert set(dependents) == {
        cid("numpy"),
        cid("linear-algebra"),
        cid("machine-learning"),
        cid("neural-networks"),
    }
    assert dependents[cid("numpy")] == 1


def test_unknown_concept_raises_not_found(graph: KnowledgeGraph) -> None:
    with pytest.raises(NotFoundError):
        graph.prerequisite_closure(cid("nonexistent"))


def test_distance_to_goals_measures_relevance(graph: KnowledgeGraph) -> None:
    distance = graph.distance_to_goals([cid("neural-networks")])
    assert distance[cid("neural-networks")] == 0
    assert distance[cid("machine-learning")] == 1
    assert distance[cid("python")] == 4


def test_concepts_that_reach_no_goal_are_absent_from_the_distance_map() -> None:
    """Off-path concepts score no goal relevance rather than a large distance."""
    nodes = [make_node("a"), make_node("goal"), make_node("unrelated")]
    graph = KnowledgeGraph(nodes, [prereq("a", "goal")])
    distance = graph.distance_to_goals([cid("goal")])
    assert cid("unrelated") not in distance


# --------------------------------------------------------------------------- #
# The DAG invariant
# --------------------------------------------------------------------------- #


def test_topological_order_places_prerequisites_first(graph: KnowledgeGraph) -> None:
    order = graph.topological_order()
    position = {concept_id: index for index, concept_id in enumerate(order)}
    assert position[cid("python")] < position[cid("numpy")]
    assert position[cid("numpy")] < position[cid("linear-algebra")]
    assert position[cid("machine-learning")] < position[cid("neural-networks")]


def test_topological_order_is_deterministic(graph: KnowledgeGraph) -> None:
    """Two runs over the same graph must produce the same roadmap."""
    assert graph.topological_order() == graph.topological_order()


def test_cyclic_graph_is_rejected_at_construction() -> None:
    nodes = [make_node("a"), make_node("b"), make_node("c")]
    with pytest.raises(CycleError) as excinfo:
        KnowledgeGraph(nodes, [prereq("a", "b"), prereq("b", "c"), prereq("c", "a")])
    assert excinfo.value.details["cycle_size"] == 3


def test_would_create_cycle_detects_a_back_edge(graph: KnowledgeGraph) -> None:
    assert graph.would_create_cycle(
        cid("neural-networks"), cid("python"), RelationType.PREREQUISITE_OF
    )


def test_would_create_cycle_allows_a_valid_edge(graph: KnowledgeGraph) -> None:
    assert not graph.would_create_cycle(
        cid("probability"), cid("neural-networks"), RelationType.PREREQUISITE_OF
    )


def test_self_loops_are_cycles(graph: KnowledgeGraph) -> None:
    assert graph.would_create_cycle(cid("python"), cid("python"), RelationType.PREREQUISITE_OF)


def test_associative_edges_may_form_cycles(graph: KnowledgeGraph) -> None:
    """Only ordering edges are constrained; `related_to` may point anywhere."""
    assert not graph.would_create_cycle(
        cid("neural-networks"), cid("python"), RelationType.RELATED_TO
    )


def test_builder_rejects_the_offending_edge_and_names_it() -> None:
    builder = GraphBuilder()
    for slug in ("a", "b"):
        builder.add_node(make_node(slug))
    builder.add_edge(prereq("a", "b"))

    with pytest.raises(CycleError) as excinfo:
        builder.add_edge(prereq("b", "a"))

    assert excinfo.value.details["source_id"] == str(cid("b"))
    assert excinfo.value.details["target_id"] == str(cid("a"))


def test_subgraph_of_a_dag_is_a_dag(graph: KnowledgeGraph) -> None:
    subset = [cid("python"), cid("numpy"), cid("linear-algebra")]
    sub = graph.subgraph(subset)
    assert len(sub) == 3
    assert sub.topological_order() == (cid("python"), cid("numpy"), cid("linear-algebra"))


def test_subgraph_drops_edges_leaving_the_selection(graph: KnowledgeGraph) -> None:
    sub = graph.subgraph([cid("numpy"), cid("linear-algebra")])
    assert sub.direct_requirements(cid("numpy")) == ()  # python was excluded


# --------------------------------------------------------------------------- #
# Learner-relative queries
# --------------------------------------------------------------------------- #


def test_concept_with_no_prerequisites_is_always_unlocked(graph: KnowledgeGraph) -> None:
    assert graph.readiness(cid("python"), {}).is_unlocked


def test_missing_mastery_blocks_a_prerequisite(graph: KnowledgeGraph) -> None:
    report = graph.readiness(cid("numpy"), {})
    assert not report.is_unlocked
    assert report.unmet[0].concept_id == cid("python")


def test_edge_strength_scales_the_required_mastery(graph: KnowledgeGraph) -> None:
    """A 0.7-strength edge demands 0.7 x the baseline, not the full amount."""
    mastery = {cid("probability"): PREREQ_SATISFACTION_THRESHOLD * 0.7 + 0.01}
    requirement = next(
        r
        for r in graph.direct_requirements(cid("machine-learning"))
        if r.concept_id == cid("probability")
    )
    assert requirement.required_mastery == pytest.approx(PREREQ_SATISFACTION_THRESHOLD * 0.7)
    assert mastery[cid("probability")] > requirement.required_mastery


def test_readiness_checks_direct_prerequisites_only(graph: KnowledgeGraph) -> None:
    """A learner strong in numpy is ready for linear algebra even with no python score.

    Re-checking transitive requirements would punish the learner twice for the same
    gap — their numpy mastery already reflects whatever python they know.
    """
    report = graph.readiness(cid("linear-algebra"), {cid("numpy"): 0.9})
    assert report.is_unlocked


def test_unlocked_frontier_filters_candidates(graph: KnowledgeGraph) -> None:
    mastery = {cid("python"): 0.9}
    frontier = graph.unlocked_frontier(
        [cid("numpy"), cid("linear-algebra"), cid("machine-learning")], mastery
    )
    assert frontier == (cid("numpy"),)


# --------------------------------------------------------------------------- #
# Blame attribution — the spec's "which prerequisite is causing the difficulty?"
# --------------------------------------------------------------------------- #


def test_blame_names_the_weak_prerequisite(graph: KnowledgeGraph) -> None:
    """The spec's worked example: struggling at ML with weak optimization."""
    mastery = {
        cid("linear-algebra"): 0.9,
        cid("optimization"): 0.2,
        cid("probability"): 0.85,
    }
    candidates = graph.blame_candidates(cid("machine-learning"), mastery)
    assert candidates[0].concept_id == cid("optimization")


def test_blame_is_empty_when_prerequisites_are_solid(graph: KnowledgeGraph) -> None:
    """Then the difficulty is with the concept itself — add practice, not remediation."""
    mastery = dict.fromkeys(
        (
            cid("linear-algebra"),
            cid("optimization"),
            cid("probability"),
            cid("numpy"),
            cid("python"),
            cid("calculus"),
        ),
        0.95,
    )
    assert graph.blame_candidates(cid("machine-learning"), mastery) == ()


def test_direct_prerequisites_outrank_distant_ancestors(graph: KnowledgeGraph) -> None:
    """Equal deficits: the nearer suspect wins."""
    mastery = {
        cid("linear-algebra"): 0.0,
        cid("numpy"): 0.0,
        cid("python"): 0.0,
        cid("optimization"): 0.95,
        cid("probability"): 0.95,
    }
    candidates = graph.blame_candidates(cid("machine-learning"), mastery, max_depth=4)
    ranked = [c.concept_id for c in candidates]
    assert ranked.index(cid("linear-algebra")) < ranked.index(cid("numpy"))
    assert ranked.index(cid("numpy")) < ranked.index(cid("python"))


def test_weak_edges_produce_weak_accusations(graph: KnowledgeGraph) -> None:
    """Zero mastery in a 0.7-strength prereq must score below the same in a 1.0 one."""
    mastery = {cid("linear-algebra"): 0.0, cid("probability"): 0.0, cid("optimization"): 0.95}
    by_id = {c.concept_id: c for c in graph.blame_candidates(cid("machine-learning"), mastery)}
    assert by_id[cid("linear-algebra")].score > by_id[cid("probability")].score


def test_a_measured_weakness_outranks_an_unmeasured_one(graph: KnowledgeGraph) -> None:
    """Evidence beats absence of evidence.

    This originally asserted the opposite — that low confidence *raised* a blame
    score, on the reasoning that an untested prerequisite is a prime suspect. The
    evaluation suite showed what that does in practice: an unmeasured concept scores
    a full 1.0 deficit and beats every concept actually observed to be weak, so the
    system reliably sends learners to whatever it knows least about. Every one of the
    six labelled blame cases failed on it.

    A concept measured at 0.2 is a better-supported accusation than one never tested,
    so confidence now scales the score rather than inflating it.
    """
    mastery = {
        **dict.fromkeys((cid("linear-algebra"), cid("probability")), 0.95),
        cid("optimization"): 0.2,
    }

    measured = graph.blame_candidates(
        cid("machine-learning"), mastery, confidence={cid("optimization"): 1.0}
    )
    unmeasured = graph.blame_candidates(
        cid("machine-learning"), mastery, confidence={cid("optimization"): 0.0}
    )
    assert measured[0].score > unmeasured[0].score


def test_an_unmeasured_prerequisite_is_still_a_suspect(graph: KnowledgeGraph) -> None:
    """The discount must not silence it entirely — for a brand-new learner nothing
    has been measured, and blame still has to name somewhere to start."""
    candidates = graph.blame_candidates(cid("machine-learning"), {}, confidence={})
    assert candidates
    assert candidates[0].score > 0


def test_a_satisfied_prerequisite_is_never_blamed_however_uncertain(
    graph: KnowledgeGraph,
) -> None:
    """Uncertainty scales an existing deficit; it must not manufacture one."""
    solid = dict.fromkeys(
        (
            cid("linear-algebra"),
            cid("optimization"),
            cid("probability"),
            cid("numpy"),
            cid("python"),
            cid("calculus"),
        ),
        0.95,
    )
    no_confidence = dict.fromkeys(solid, 0.0)
    assert graph.blame_candidates(cid("machine-learning"), solid, confidence=no_confidence) == ()


def test_blame_reports_the_numbers_behind_the_ranking(graph: KnowledgeGraph) -> None:
    """The explanation prompt gets these fields; it must not invent its own."""
    candidate = graph.blame_candidates(cid("machine-learning"), {cid("optimization"): 0.1})[0]
    assert 0.0 < candidate.deficit <= 1.0
    assert candidate.hops >= 1
    assert 0.0 < candidate.strength <= 1.0


def test_blame_respects_the_result_limit(graph: KnowledgeGraph) -> None:
    assert len(graph.blame_candidates(cid("neural-networks"), {}, max_depth=5, limit=2)) == 2


# --------------------------------------------------------------------------- #
# Properties — these must hold for any graph, not just the fixture
# --------------------------------------------------------------------------- #


@st.composite
def random_dag(draw: st.DrawFn) -> KnowledgeGraph:
    """An arbitrary DAG, built by only ever pointing edges forwards in a fixed order."""
    size = draw(st.integers(min_value=1, max_value=12))
    slugs = [f"c{i}" for i in range(size)]
    nodes = [make_node(slug) for slug in slugs]

    edges = []
    for target_index in range(size):
        for source_index in range(target_index):
            if draw(st.booleans()):
                strength = draw(st.floats(min_value=0.1, max_value=1.0))
                edges.append(prereq(slugs[source_index], slugs[target_index], strength))
    return KnowledgeGraph(nodes, edges)


@given(random_dag())
def test_topological_order_covers_every_node(graph: KnowledgeGraph) -> None:
    assert set(graph.topological_order()) == graph.node_ids


@given(random_dag())
def test_topological_order_never_places_a_dependent_before_its_requirement(
    graph: KnowledgeGraph,
) -> None:
    order = graph.topological_order()
    position = {concept_id: index for index, concept_id in enumerate(order)}
    for concept_id in order:
        for requirement in graph.direct_requirements(concept_id):
            assert position[requirement.concept_id] < position[concept_id]


@given(random_dag())
def test_a_concept_is_never_its_own_prerequisite(graph: KnowledgeGraph) -> None:
    for concept_id in graph.node_ids:
        assert concept_id not in graph.prerequisite_closure(concept_id)


@given(random_dag())
def test_closure_and_dependent_closure_are_mutually_consistent(
    graph: KnowledgeGraph,
) -> None:
    """`b` requires `a`  <=>  `a` unlocks `b`."""
    for concept_id in graph.node_ids:
        for requirement_id in graph.prerequisite_closure(concept_id):
            assert concept_id in graph.dependent_closure(requirement_id)


@given(random_dag())
def test_path_strength_never_exceeds_one(graph: KnowledgeGraph) -> None:
    for concept_id in graph.node_ids:
        for requirement in graph.prerequisite_closure(concept_id).values():
            assert 0.0 < requirement.strength <= 1.0
