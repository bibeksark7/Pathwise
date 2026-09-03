"""Roadmap planning tests.

The planner is where "deterministic logic decides, the LLM explains" is either true
or it is not. These tests pin the properties that make it true: the same inputs
always produce the same roadmap, prerequisites always precede dependents, nothing is
included that the graph does not require, and material is only skipped on evidence
strong enough to justify it.

Both a small hand-built fixture (for precise assertions) and the real 89-concept seed
graph (for realistic scale) are exercised.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from uuid import UUID

import pytest

from pathwise.api.errors import ValidationError
from pathwise.models.enums import EvidenceSource, NodeStatus, RelationType
from pathwise.services.knowledge.graph import GraphEdge, GraphNode, KnowledgeGraph
from pathwise.services.knowledge.mastery import MasteryEstimate, Observation, rebuild
from pathwise.services.knowledge.seed import (
    build_graph_from_corpus,
    concept_id_for,
    load_corpus,
)
from pathwise.services.roadmap.planner import (
    MAX_ROADMAP_NODES,
    OPTIONAL_EDGE_STRENGTH,
    plan_roadmap,
)

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def cid(slug: str) -> UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, slug)


def node(slug: str, *, minutes: int = 60, difficulty: int = 3) -> GraphNode:
    return GraphNode(
        id=cid(slug),
        slug=slug,
        name=slug.replace("-", " ").title(),
        difficulty=difficulty,
        estimated_minutes=minutes,
        domain="test",
    )


def prereq(source: str, target: str, strength: float = 1.0) -> GraphEdge:
    return GraphEdge(cid(source), cid(target), RelationType.PREREQUISITE_OF, strength)


def mastered(slug: str, *, times: int = 8, score: float = 1.0) -> MasteryEstimate:
    """A concept with enough evidence to clear the skip threshold."""
    return rebuild(
        [Observation(cid(slug), EvidenceSource.ASSESSMENT, score, NOW) for _ in range(times)]
    )


@pytest.fixture
def chain() -> KnowledgeGraph:
    """basics -> intermediate -> advanced -> goal, plus an off-path concept."""
    return KnowledgeGraph(
        [node(s) for s in ("basics", "intermediate", "advanced", "goal", "unrelated")],
        [
            prereq("basics", "intermediate"),
            prereq("intermediate", "advanced"),
            prereq("advanced", "goal"),
        ],
    )


@pytest.fixture(scope="module")
def seed_graph() -> KnowledgeGraph:
    return build_graph_from_corpus(load_corpus())


# --------------------------------------------------------------------------- #
# Scope: the graph decides what is required
# --------------------------------------------------------------------------- #


def test_the_plan_is_the_prerequisite_closure(chain: KnowledgeGraph) -> None:
    plan = plan_roadmap(chain, [cid("goal")], now=NOW)
    assert plan.slugs == ("basics", "intermediate", "advanced", "goal")


def test_unrelated_concepts_are_never_included(chain: KnowledgeGraph) -> None:
    """A model asked to "build a roadmap" pads. Graph traversal cannot."""
    plan = plan_roadmap(chain, [cid("goal")], now=NOW)
    assert "unrelated" not in plan.slugs


def test_the_goal_itself_is_included(chain: KnowledgeGraph) -> None:
    assert cid("goal") in plan_roadmap(chain, [cid("goal")], now=NOW).concept_ids


def test_multiple_goals_are_merged_without_duplication() -> None:
    graph = KnowledgeGraph(
        [node(s) for s in ("shared", "goal-a", "goal-b")],
        [prereq("shared", "goal-a"), prereq("shared", "goal-b")],
    )
    plan = plan_roadmap(graph, [cid("goal-a"), cid("goal-b")], now=NOW)
    assert sorted(plan.slugs) == ["goal-a", "goal-b", "shared"]
    assert len(plan.slugs) == len(set(plan.slugs))


def test_an_unknown_goal_is_rejected(chain: KnowledgeGraph) -> None:
    with pytest.raises(ValidationError, match=r"None of the goal concepts"):
        plan_roadmap(chain, [uuid.uuid4()], now=NOW)


def test_goals_outside_the_graph_are_filtered_but_valid_ones_still_plan(
    chain: KnowledgeGraph,
) -> None:
    plan = plan_roadmap(chain, [cid("goal"), uuid.uuid4()], now=NOW)
    assert "goal" in plan.slugs


def test_an_over_broad_goal_is_refused() -> None:
    """A 200-step "roadmap" is a syllabus. Better to say so than to render it."""
    slugs = [f"c{i}" for i in range(MAX_ROADMAP_NODES + 5)]
    graph = KnowledgeGraph(
        [node(s) for s in slugs],
        [prereq(slugs[i], slugs[-1]) for i in range(len(slugs) - 1)],
    )
    with pytest.raises(ValidationError, match="more steps than"):
        plan_roadmap(graph, [cid(slugs[-1])], now=NOW)


# --------------------------------------------------------------------------- #
# Sequencing
# --------------------------------------------------------------------------- #


def test_prerequisites_always_precede_dependents(seed_graph: KnowledgeGraph) -> None:
    """On the real graph, not a toy: 39 concepts with cross-domain dependencies."""
    plan = plan_roadmap(seed_graph, [concept_id_for("ml-system-design")], now=NOW)
    position = {n.concept_id: n.order_index for n in plan.nodes}
    for source, target, _ in plan.edges:
        assert position[source] < position[target]


def test_planning_is_deterministic(seed_graph: KnowledgeGraph) -> None:
    """Two learners with identical inputs must get identical roadmaps, and the same
    learner must not get a different one on a refresh."""
    goal = [concept_id_for("backpropagation")]
    assert (
        plan_roadmap(seed_graph, goal, now=NOW).slugs
        == plan_roadmap(seed_graph, goal, now=NOW).slugs
    )


def test_order_indexes_are_contiguous(chain: KnowledgeGraph) -> None:
    plan = plan_roadmap(chain, [cid("goal")], now=NOW)
    assert [n.order_index for n in plan.nodes] == list(range(len(plan.nodes)))


# --------------------------------------------------------------------------- #
# Reduction: shortening the path on evidence
# --------------------------------------------------------------------------- #


def test_demonstrated_prerequisites_are_skipped(chain: KnowledgeGraph) -> None:
    """The feature that lets Pathwise shorten a path rather than only lengthen it."""
    plan = plan_roadmap(chain, [cid("goal")], {cid("basics"): mastered("basics")}, now=NOW)
    assert "basics" not in plan.slugs
    assert [s.slug for s in plan.skipped] == ["basics"]


def test_a_skip_records_the_evidence_behind_it(chain: KnowledgeGraph) -> None:
    """ "You already know this" is only sayable if the exclusion carries its reason."""
    plan = plan_roadmap(chain, [cid("goal")], {cid("basics"): mastered("basics")}, now=NOW)
    skipped = plan.skipped[0]
    assert skipped.mastery > 0.85
    assert skipped.evidence_count == 8


def test_one_lucky_quiz_does_not_skip_material(chain: KnowledgeGraph) -> None:
    """Thin evidence must not excuse a prerequisite — that is how a learner gets
    stranded three topics later with no idea why."""
    thin = rebuild([Observation(cid("basics"), EvidenceSource.QUIZ, 1.0, NOW)])
    plan = plan_roadmap(chain, [cid("goal")], {cid("basics"): thin}, now=NOW)
    assert "basics" in plan.slugs


def test_poor_performance_never_skips(chain: KnowledgeGraph) -> None:
    weak = mastered("basics", score=0.3)
    plan = plan_roadmap(chain, [cid("goal")], {cid("basics"): weak}, now=NOW)
    assert "basics" in plan.slugs


def test_the_goal_is_never_skipped(chain: KnowledgeGraph) -> None:
    """Prior competence in the goal should shorten the path to it, not delete the
    point of the roadmap."""
    plan = plan_roadmap(chain, [cid("goal")], {cid("goal"): mastered("goal")}, now=NOW)
    assert "goal" in plan.slugs


def test_decayed_mastery_stops_excusing_material(chain: KnowledgeGraph) -> None:
    """Something learned two years ago is not something you still know.

    This previously asserted only that the *reported* mastery figure decayed, while
    the skip decision itself read undecayed mastery — so a concept demonstrated once,
    years ago, stayed excused from every future roadmap and the forgetting curve had
    no effect on anything the learner saw. The test documented the bug instead of
    catching it; it now checks the behaviour it is named for.
    """
    long_ago = datetime(2024, 1, 1, tzinfo=UTC)
    stale = rebuild(
        [Observation(cid("basics"), EvidenceSource.ASSESSMENT, 1.0, long_ago) for _ in range(8)]
    )
    mastery = {cid("basics"): stale}

    when_fresh = plan_roadmap(chain, [cid("goal")], mastery, now=long_ago)
    much_later = plan_roadmap(chain, [cid("goal")], mastery, now=NOW)

    # Freshly demonstrated: excused from the roadmap.
    assert "basics" not in when_fresh.slugs
    assert [s.slug for s in when_fresh.skipped] == ["basics"]

    # Two years on: back in the roadmap, because it is no longer known.
    assert "basics" in much_later.slugs
    assert much_later.skipped == ()


def test_recent_mastery_is_still_excused(chain: KnowledgeGraph) -> None:
    """The other side of the same rule — decay must not be so aggressive that
    something demonstrated last week is re-taught."""
    last_week = NOW - timedelta(days=7)
    recent = rebuild(
        [Observation(cid("basics"), EvidenceSource.ASSESSMENT, 1.0, last_week) for _ in range(8)]
    )
    plan = plan_roadmap(chain, [cid("goal")], {cid("basics"): recent}, now=NOW)
    assert "basics" not in plan.slugs


def test_the_scope_trace_quantifies_the_reduction(chain: KnowledgeGraph) -> None:
    plan = plan_roadmap(chain, [cid("goal")], {cid("basics"): mastered("basics")}, now=NOW)
    assert plan.scope.closure_size == 4
    assert plan.scope.skipped_count == 1
    assert plan.scope.included_count == 3
    assert plan.scope.reduction_ratio == pytest.approx(0.25)


def test_a_learner_who_knows_everything_gets_an_empty_plan(chain: KnowledgeGraph) -> None:
    mastery = {cid(s): mastered(s) for s in ("basics", "intermediate", "advanced")}
    mastery[cid("goal")] = mastered("goal")
    plan = plan_roadmap(chain, [cid("goal")], mastery, now=NOW)
    # The goal survives, so this is not empty — but everything beneath it is gone.
    assert plan.slugs == ("goal",)


# --------------------------------------------------------------------------- #
# Node status
# --------------------------------------------------------------------------- #


def test_the_first_step_is_startable_and_the_rest_are_locked(chain: KnowledgeGraph) -> None:
    plan = plan_roadmap(chain, [cid("goal")], now=NOW)
    assert plan.nodes[0].status is NodeStatus.NOT_STARTED
    assert all(n.status is NodeStatus.LOCKED for n in plan.nodes[1:])


def test_nodes_carry_their_in_roadmap_dependencies(chain: KnowledgeGraph) -> None:
    plan = plan_roadmap(chain, [cid("goal")], now=NOW)
    assert plan.node_for(cid("intermediate")).depends_on == (cid("basics"),)  # type: ignore[union-attr]
    assert plan.nodes[0].depends_on == ()


def test_skipping_a_prerequisite_unlocks_what_followed_it(chain: KnowledgeGraph) -> None:
    plan = plan_roadmap(chain, [cid("goal")], {cid("basics"): mastered("basics")}, now=NOW)
    assert plan.nodes[0].slug == "intermediate"
    assert plan.nodes[0].status is NodeStatus.NOT_STARTED


def test_edges_to_skipped_concepts_are_dropped(chain: KnowledgeGraph) -> None:
    """The projection must not dangle at a node that is no longer in the plan."""
    plan = plan_roadmap(chain, [cid("goal")], {cid("basics"): mastered("basics")}, now=NOW)
    present = set(plan.concept_ids)
    for source, target, _ in plan.edges:
        assert source in present
        assert target in present


# --------------------------------------------------------------------------- #
# Pacing
# --------------------------------------------------------------------------- #


def test_time_is_totalled_from_the_included_steps(chain: KnowledgeGraph) -> None:
    plan = plan_roadmap(chain, [cid("goal")], hours_per_week=4.0, now=NOW)
    assert plan.pacing.total_minutes == 240  # four 60-minute concepts
    assert plan.pacing.estimated_weeks == pytest.approx(1.0)


def test_no_deadline_means_unknown_rather_than_fine(chain: KnowledgeGraph) -> None:
    """`None` and `True` are different answers, and conflating them would tell a
    learner with no deadline that they are on track for one."""
    pacing = plan_roadmap(chain, [cid("goal")], now=NOW).pacing
    assert pacing.meets_deadline is None
    assert pacing.required_hours_per_week is None


def test_a_generous_deadline_is_comfortable(chain: KnowledgeGraph) -> None:
    pacing = plan_roadmap(
        chain, [cid("goal")], hours_per_week=4.0, deadline=date(2027, 1, 1), now=NOW
    ).pacing
    assert pacing.meets_deadline is True
    assert pacing.is_comfortable is True


def test_an_impossible_deadline_is_reported_with_the_hours_that_would_work(
    chain: KnowledgeGraph,
) -> None:
    """ "You are 10 weeks over" is not actionable; "you would need 18 hours a week" is."""
    pacing = plan_roadmap(
        chain, [cid("goal")], hours_per_week=1.0, deadline=date(2026, 9, 8), now=NOW
    ).pacing
    assert pacing.meets_deadline is False
    assert pacing.weeks_over > 0
    assert pacing.required_hours_per_week == pytest.approx(4.0)


def test_a_deadline_already_past_does_not_divide_by_zero(chain: KnowledgeGraph) -> None:
    pacing = plan_roadmap(chain, [cid("goal")], deadline=date(2026, 1, 1), now=NOW).pacing
    assert pacing.weeks_available == 0.0
    assert pacing.required_hours_per_week == float("inf")


def test_zero_weekly_hours_does_not_divide_by_zero(chain: KnowledgeGraph) -> None:
    assert (
        plan_roadmap(chain, [cid("goal")], hours_per_week=0.0, now=NOW).pacing.estimated_weeks > 0
    )


def test_a_missed_deadline_produces_a_warning(chain: KnowledgeGraph) -> None:
    plan = plan_roadmap(
        chain, [cid("goal")], hours_per_week=1.0, deadline=date(2026, 9, 8), now=NOW
    )
    assert any("past your deadline" in w for w in plan.warnings)


# --------------------------------------------------------------------------- #
# Optional steps
# --------------------------------------------------------------------------- #


def test_a_weakly_linked_concept_is_optional() -> None:
    """Reachable only across a "helpful, not required" edge."""
    graph = KnowledgeGraph(
        [node(s) for s in ("core", "nice-to-have", "goal")],
        [prereq("core", "goal"), prereq("nice-to-have", "goal", OPTIONAL_EDGE_STRENGTH - 0.1)],
    )
    plan = plan_roadmap(graph, [cid("goal")], now=NOW)
    assert [n.slug for n in plan.optional_steps] == ["nice-to-have"]


def test_a_strongly_linked_concept_is_never_optional() -> None:
    """The bug this replaced: offering to drop genuine prerequisites to hit a date."""
    graph = KnowledgeGraph([node(s) for s in ("core", "goal")], [prereq("core", "goal", 1.0)])
    assert plan_roadmap(graph, [cid("goal")], now=NOW).optional_steps == ()


def test_optionality_is_transitive_through_weak_links() -> None:
    """A prerequisite of an optional step is itself only optionally needed."""
    graph = KnowledgeGraph(
        [node(s) for s in ("deep", "nice-to-have", "goal")],
        [prereq("deep", "nice-to-have"), prereq("nice-to-have", "goal", 0.5)],
    )
    optional = {n.slug for n in plan_roadmap(graph, [cid("goal")], now=NOW).optional_steps}
    assert optional == {"deep", "nice-to-have"}


def test_a_single_goal_closure_of_strong_edges_has_nothing_to_drop(
    chain: KnowledgeGraph,
) -> None:
    """The regression this whole section exists for."""
    assert plan_roadmap(chain, [cid("goal")], now=NOW).optional_steps == ()


# --------------------------------------------------------------------------- #
# Against the real graph
# --------------------------------------------------------------------------- #


def test_the_spec_scenario_produces_a_usable_plan(seed_graph: KnowledgeGraph) -> None:
    """ "I want to become an ML engineer... I can study 8 hours a week." """
    plan = plan_roadmap(
        seed_graph, [concept_id_for("ml-system-design")], hours_per_week=8.0, now=NOW
    )
    assert 20 <= len(plan.nodes) <= MAX_ROADMAP_NODES
    assert plan.pacing.estimated_weeks > 4
    assert plan.nodes[0].status is NodeStatus.NOT_STARTED
    # The path must actually terminate at what was asked for.
    assert plan.nodes[-1].slug == "ml-system-design"


def test_prior_knowledge_measurably_shortens_the_real_path(
    seed_graph: KnowledgeGraph,
) -> None:
    """The spec's "I already know Python and basic calculus" case."""
    known = (
        "programming-basics",
        "python-syntax-and-types",
        "python-control-flow",
        "python-functions",
        "python-data-structures",
        "functions-and-graphs",
        "limits-and-continuity",
        "derivatives",
    )
    mastery = {
        concept_id_for(slug): rebuild(
            [
                Observation(concept_id_for(slug), EvidenceSource.ASSESSMENT, 1.0, NOW)
                for _ in range(8)
            ]
        )
        for slug in known
    }
    goal = [concept_id_for("ml-system-design")]

    baseline = plan_roadmap(seed_graph, goal, hours_per_week=8.0, now=NOW)
    reduced = plan_roadmap(seed_graph, goal, mastery, hours_per_week=8.0, now=NOW)

    assert len(reduced.nodes) < len(baseline.nodes)
    assert reduced.pacing.total_minutes < baseline.pacing.total_minutes
    assert reduced.scope.reduction_ratio > 0.1
    assert {s.slug for s in reduced.skipped} == set(known)


def test_every_real_plan_starts_with_something_startable(
    seed_graph: KnowledgeGraph,
) -> None:
    """A roadmap where every step is locked is a roadmap nobody can begin."""
    for goal_slug in ("backpropagation", "ensemble-methods", "attention-and-transformers"):
        plan = plan_roadmap(seed_graph, [concept_id_for(goal_slug)], now=NOW)
        assert any(n.status is NodeStatus.NOT_STARTED for n in plan.nodes), goal_slug
