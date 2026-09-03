"""Decision engine tests.

"What should I do next?" is the most important answer the product gives, and it is
arithmetic. These tests pin the behaviour a learner would notice — never being sent
at something they are not ready for, being routed at the cause when they struggle,
being reminded of what has decayed — and the property that makes the answer
*explainable*: a trace complete enough that prose built from it can be checked.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from pathwise.ai.validators import grounded_in_trace
from pathwise.models.enums import EvidenceSource, RelationType
from pathwise.services.decision.engine import (
    WEIGHTS,
    DecisionTrace,
    LearnerContext,
    fallback_explanation,
    recommend_next,
    review_candidates,
    summarise,
)
from pathwise.services.knowledge.graph import GraphEdge, GraphNode, KnowledgeGraph
from pathwise.services.knowledge.mastery import MasteryEstimate, Observation, rebuild
from pathwise.services.knowledge.seed import (
    build_graph_from_corpus,
    concept_id_for,
    load_corpus,
)
from pathwise.services.roadmap.planner import plan_roadmap

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def cid(slug: str) -> UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, slug)


def node(slug: str, difficulty: int = 3, domain: str = "test") -> GraphNode:
    return GraphNode(
        id=cid(slug),
        slug=slug,
        name=slug.replace("-", " ").title(),
        difficulty=difficulty,
        estimated_minutes=120,
        domain=domain,
    )


def prereq(source: str, target: str) -> GraphEdge:
    return GraphEdge(cid(source), cid(target), RelationType.PREREQUISITE_OF, 1.0)


def mastered(
    slug: str, *, times: int = 6, score: float = 1.0, at: datetime = NOW
) -> MasteryEstimate:
    return rebuild(
        [Observation(cid(slug), EvidenceSource.ASSESSMENT, score, at) for _ in range(times)]
    )


@pytest.fixture
def chain() -> KnowledgeGraph:
    """basics -> middle -> goal, plus a standalone side topic."""
    return KnowledgeGraph(
        [node("basics", 1), node("middle", 2), node("goal", 3), node("side", 1)],
        [prereq("basics", "middle"), prereq("middle", "goal")],
    )


def plan_for(graph: KnowledgeGraph, goal: str, mastery=None):
    return plan_roadmap(graph, [cid(goal)], mastery or {}, now=NOW)


@pytest.fixture(scope="module")
def seed_graph() -> KnowledgeGraph:
    return build_graph_from_corpus(load_corpus())


# --------------------------------------------------------------------------- #
# Filtering: never recommend something unreachable
# --------------------------------------------------------------------------- #


def test_a_locked_concept_is_never_recommended(chain: KnowledgeGraph) -> None:
    """The failure a learner notices immediately: being sent at something they
    cannot start."""
    trace = recommend_next(
        chain,
        plan_for(chain, "goal"),
        LearnerContext(mastery={}, goal_concept_ids=(cid("goal"),)),
        now=NOW,
    )
    assert trace.recommended is not None
    assert trace.recommended.slug == "basics"


def test_exclusions_record_why(chain: KnowledgeGraph) -> None:
    """ "Why not X?" should be answerable from the trace rather than by re-deriving
    the engine's reasoning."""
    trace = recommend_next(
        chain,
        plan_for(chain, "goal"),
        LearnerContext(mastery={}, goal_concept_ids=(cid("goal"),)),
        now=NOW,
    )
    reasons = dict(trace.excluded)
    assert "prerequisites not met" in reasons["middle"]
    assert "basics" in reasons["middle"]


def test_a_mastered_concept_is_excluded(chain: KnowledgeGraph) -> None:
    mastery = {cid("basics"): mastered("basics")}
    trace = recommend_next(
        chain,
        plan_for(chain, "goal", mastery),
        LearnerContext(mastery=mastery, goal_concept_ids=(cid("goal"),)),
        now=NOW,
    )
    assert trace.recommended is not None
    assert trace.recommended.slug != "basics"


def test_nothing_available_is_reported_honestly(chain: KnowledgeGraph) -> None:
    """An empty recommendation is a real state, not an error to paper over."""
    empty = KnowledgeGraph([node("goal")], [])
    plan = plan_roadmap(empty, [cid("goal")], now=NOW)
    trace = recommend_next(
        empty,
        plan,
        LearnerContext(mastery={cid("goal"): mastered("goal")}),
        now=NOW,
    )
    assert not trace.has_recommendation
    assert "nothing" in fallback_explanation(trace).lower()


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def test_the_decision_is_deterministic(seed_graph: KnowledgeGraph) -> None:
    goal = concept_id_for("backpropagation")
    plan = plan_roadmap(seed_graph, [goal], now=NOW)
    context = LearnerContext(mastery={}, goal_concept_ids=(goal,))

    first = recommend_next(seed_graph, plan, context, now=NOW)
    second = recommend_next(seed_graph, plan, context, now=NOW)
    assert first.recommended.slug == second.recommended.slug  # type: ignore[union-attr]


def test_readiness_saturates_so_entry_points_do_not_always_win(
    seed_graph: KnowledgeGraph,
) -> None:
    """The bug this guards against: a learner who already knows calculus was being
    sent to probability-fundamentals five steps off-goal, because a concept with no
    prerequisites scored a perfect readiness while one whose sole prerequisite was
    mastered at 0.93 scored 0.33."""
    goal = concept_id_for("backpropagation")
    known = (
        "programming-basics",
        "python-syntax-and-types",
        "python-control-flow",
        "python-functions",
        "python-data-structures",
        "functions-and-graphs",
        "limits-and-continuity",
        "derivatives",
        "vectors-and-spaces",
        "matrix-operations",
        "numpy-fundamentals",
    )
    mastery = {
        concept_id_for(slug): rebuild(
            [
                Observation(concept_id_for(slug), EvidenceSource.ASSESSMENT, 1.0, NOW)
                for _ in range(6)
            ]
        )
        for slug in known
    }
    plan = plan_roadmap(seed_graph, [goal], mastery, now=NOW)
    trace = recommend_next(
        seed_graph, plan, LearnerContext(mastery=mastery, goal_concept_ids=(goal,)), now=NOW
    )
    assert trace.recommended is not None
    assert trace.recommended.slug == "chain-rule"


def test_goal_relevance_prefers_the_shorter_route(chain: KnowledgeGraph) -> None:
    """`side` is startable but leads nowhere near the goal."""
    trace = recommend_next(
        chain,
        plan_roadmap(chain, [cid("goal"), cid("side")], now=NOW),
        LearnerContext(mastery={}, goal_concept_ids=(cid("goal"),)),
        now=NOW,
    )
    assert trace.recommended is not None
    assert trace.recommended.slug == "basics"


def test_momentum_prefers_staying_in_one_subject() -> None:
    graph = KnowledgeGraph(
        [node("maths-a", 2, "mathematics"), node("code-a", 2, "programming"), node("goal", 3)],
        [prereq("maths-a", "goal"), prereq("code-a", "goal")],
    )
    plan = plan_roadmap(graph, [cid("goal")], now=NOW)

    maths = recommend_next(
        graph,
        plan,
        LearnerContext(mastery={}, goal_concept_ids=(cid("goal"),), last_domain="mathematics"),
        now=NOW,
    )
    assert maths.recommended is not None
    assert maths.recommended.domain == "mathematics"


def test_remediation_routes_at_the_cause_of_a_failure() -> None:
    """Without this the engine marches a struggling learner onward, which is the
    behaviour the whole product exists to avoid.

    The two candidates are deliberately identical — same difficulty, same domain,
    both goals, neither with prerequisites — so remediation is the only thing that
    can separate them. Anything less isolated would pass for the wrong reason.
    """
    graph = KnowledgeGraph([node("topic-a", 2), node("topic-b", 2)], [])
    plan = plan_roadmap(graph, [cid("topic-a"), cid("topic-b")], now=NOW)
    goals = (cid("topic-a"), cid("topic-b"))

    baseline = recommend_next(
        graph, plan, LearnerContext(mastery={}, goal_concept_ids=goals), now=NOW
    )
    # Identical scores, so the tie breaks on slug and `topic-a` wins.
    assert baseline.recommended is not None
    assert baseline.recommended.slug == "topic-a"

    remediated = recommend_next(
        graph,
        plan,
        LearnerContext(
            mastery={},
            goal_concept_ids=goals,
            remediation_targets=frozenset({cid("topic-b")}),
        ),
        now=NOW,
    )
    assert remediated.recommended is not None
    assert remediated.recommended.slug == "topic-b"
    assert remediated.recommended.factor("remediation").value == 1.0  # type: ignore[union-attr]


def test_deadline_pressure_reweights_towards_the_goal(chain: KnowledgeGraph) -> None:
    plan = plan_for(chain, "goal")
    relaxed = recommend_next(
        chain, plan, LearnerContext(mastery={}, goal_concept_ids=(cid("goal"),)), now=NOW
    )
    pressed = recommend_next(
        chain,
        plan,
        LearnerContext(mastery={}, goal_concept_ids=(cid("goal"),), under_deadline_pressure=True),
        now=NOW,
    )
    assert pressed.weights["goal_relevance"] > relaxed.weights["goal_relevance"]


def test_weights_always_sum_to_one(chain: KnowledgeGraph) -> None:
    """Otherwise a pressured learner's scores live on a different scale from an
    unpressured one, and neither the trace nor an evaluation could compare them."""
    plan = plan_for(chain, "goal")
    for pressure in (False, True):
        trace = recommend_next(
            chain,
            plan,
            LearnerContext(
                mastery={}, goal_concept_ids=(cid("goal"),), under_deadline_pressure=pressure
            ),
            now=NOW,
        )
        assert sum(trace.weights.values()) == pytest.approx(1.0)


def test_the_shipped_weights_sum_to_one() -> None:
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Review
# --------------------------------------------------------------------------- #


def test_decayed_knowledge_becomes_a_review_candidate(chain: KnowledgeGraph) -> None:
    """Retention is worth more than coverage — relearning beats re-deriving."""
    long_ago = NOW - timedelta(days=400)
    mastery = {cid("basics"): mastered("basics", times=4, at=long_ago)}

    # Planned at NOW, by which point the knowledge has decayed below the skip bar —
    # so it is back in the roadmap and eligible to be recommended as a review.
    plan = plan_roadmap(chain, [cid("goal")], mastery, now=NOW)
    assert "basics" in plan.slugs, "decayed knowledge must return to the roadmap"

    trace = recommend_next(
        chain, plan, LearnerContext(mastery=mastery, goal_concept_ids=(cid("goal"),)), now=NOW
    )
    reviews = [c for c in (trace.recommended, *trace.alternatives) if c and c.kind == "review"]
    assert reviews


def test_fresh_knowledge_is_not_flagged_for_review(chain: KnowledgeGraph) -> None:
    assert review_candidates({cid("basics"): mastered("basics")}, NOW) == ()


def test_never_studied_material_is_not_due_for_review(chain: KnowledgeGraph) -> None:
    """You cannot be due to revise something you never learned."""
    assert review_candidates({cid("basics"): MasteryEstimate()}, NOW) == ()


def test_review_debt_scores_zero_for_new_material(chain: KnowledgeGraph) -> None:
    trace = recommend_next(chain, plan_for(chain, "goal"), LearnerContext(mastery={}), now=NOW)
    assert trace.recommended is not None
    assert trace.recommended.factor("review_debt").value == 0.0  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# The trace — what makes this explainable rather than narrated
# --------------------------------------------------------------------------- #


def test_every_factor_is_recorded(chain: KnowledgeGraph) -> None:
    trace = recommend_next(chain, plan_for(chain, "goal"), LearnerContext(mastery={}), now=NOW)
    assert trace.recommended is not None
    assert {f.name for f in trace.recommended.factors} == set(WEIGHTS)


def test_the_score_is_the_sum_of_its_contributions(chain: KnowledgeGraph) -> None:
    """If these disagreed, the trace would not explain the ranking it accompanies."""
    trace = recommend_next(chain, plan_for(chain, "goal"), LearnerContext(mastery={}), now=NOW)
    candidate = trace.recommended
    assert candidate is not None
    assert candidate.score == pytest.approx(sum(f.contribution for f in candidate.factors))


def test_every_factor_carries_a_human_readable_detail(chain: KnowledgeGraph) -> None:
    """These are handed to the explanation prompt verbatim."""
    trace = recommend_next(chain, plan_for(chain, "goal"), LearnerContext(mastery={}), now=NOW)
    assert trace.recommended is not None
    for factor in trace.recommended.factors:
        assert factor.detail.strip()


def test_the_deciding_factor_differs_from_the_largest_term(
    seed_graph: KnowledgeGraph,
) -> None:
    """Readiness saturates for nearly every viable candidate, so leading with the
    largest term would open every explanation with "you're ready for it" — true,
    identical every time, and useless."""
    goal = concept_id_for("backpropagation")
    known = ("functions-and-graphs", "limits-and-continuity", "derivatives")
    mastery = {
        concept_id_for(slug): rebuild(
            [
                Observation(concept_id_for(slug), EvidenceSource.ASSESSMENT, 1.0, NOW)
                for _ in range(6)
            ]
        )
        for slug in known
    }
    plan = plan_roadmap(seed_graph, [goal], mastery, now=NOW)
    trace = recommend_next(
        seed_graph, plan, LearnerContext(mastery=mastery, goal_concept_ids=(goal,)), now=NOW
    )
    assert trace.deciding_factor is not None
    assert trace.deciding_factor.name == "goal_relevance"


def test_a_lone_candidate_falls_back_to_its_largest_term(chain: KnowledgeGraph) -> None:
    """With no runner-up there is nothing to have decided against."""
    single = KnowledgeGraph([node("only", 2)], [])
    trace = recommend_next(
        single, plan_roadmap(single, [cid("only")], now=NOW), LearnerContext(mastery={}), now=NOW
    )
    assert trace.deciding_factor is trace.recommended.dominant_factor  # type: ignore[union-attr]


def test_no_recommendation_means_no_deciding_factor() -> None:
    assert DecisionTrace(recommended=None, alternatives=()).deciding_factor is None


def test_the_prompt_payload_carries_the_reasoning(chain: KnowledgeGraph) -> None:
    trace = recommend_next(chain, plan_for(chain, "goal"), LearnerContext(mastery={}), now=NOW)
    payload = trace.to_prompt_json()
    recommended = payload["recommended"]
    assert isinstance(recommended, dict)
    assert recommended["deciding_factor"]
    assert len(recommended["factors"]) == len(WEIGHTS)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Grounding: the trace and the hallucination guard must agree
# --------------------------------------------------------------------------- #


def test_the_deterministic_explanation_only_cites_trace_figures(
    chain: KnowledgeGraph,
) -> None:
    """The two halves of the explainability claim, checked against each other. If the
    engine's own fallback prose failed its own grounding validator, the guard would be
    calibrated wrong and would reject good generated output too.
    """
    trace = recommend_next(chain, plan_for(chain, "goal"), LearnerContext(mastery={}), now=NOW)
    validate = grounded_in_trace(trace.citable_numbers())
    assert validate(fallback_explanation(trace)).is_valid


def test_an_invented_figure_is_rejected_against_the_trace(chain: KnowledgeGraph) -> None:
    trace = recommend_next(chain, plan_for(chain, "goal"), LearnerContext(mastery={}), now=NOW)
    validate = grounded_in_trace(trace.citable_numbers())
    assert not validate("You scored 72% on this, so it is next.").is_valid


def test_citable_numbers_include_the_headline_figures(chain: KnowledgeGraph) -> None:
    trace = recommend_next(chain, plan_for(chain, "goal"), LearnerContext(mastery={}), now=NOW)
    numbers = set(trace.citable_numbers())
    assert float(trace.recommended.estimated_minutes) in numbers  # type: ignore[union-attr]
    assert float(trace.recommended.difficulty) in numbers  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Dashboard payload
# --------------------------------------------------------------------------- #


def test_the_summary_answers_what_should_i_do_next(chain: KnowledgeGraph) -> None:
    trace = recommend_next(chain, plan_for(chain, "goal"), LearnerContext(mastery={}), now=NOW)
    summary = summarise(trace, chain)
    assert summary["next"]["slug"] == "basics"  # type: ignore[index]
    assert summary["reason"]
    assert isinstance(summary["alternatives"], list)


def test_the_summary_handles_having_nothing_to_suggest() -> None:
    summary = summarise(DecisionTrace(recommended=None, alternatives=()), KnowledgeGraph.empty())
    assert summary["next"] is None
