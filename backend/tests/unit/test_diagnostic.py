"""Diagnostic assessment tests.

The diagnostic's job is to learn as much as possible about a learner from as few
questions as they will actually answer, and then to change the roadmap accordingly.
These tests cover both halves: that probe selection is informative and deterministic,
and that answers become evidence that measurably shortens the path.

The behaviour worth stating plainly, because it is a deliberate two-tier design:

* A **diagnostic compresses** — one correct answer earns a shorter review step.
* **Sustained evidence skips** — repeated demonstration removes the step entirely.

Conflating them would either make the diagnostic useless (nothing ever shortens) or
dangerous (one lucky answer deletes a prerequisite).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from uuid import UUID

import pytest

from pathwise.api.errors import ValidationError
from pathwise.models.enums import EvidenceSource, NodeType, RelationType
from pathwise.services.assessment.grading import (
    DIAGNOSTIC_FULL_WEIGHT_QUESTIONS,
    GradedAnswer,
    QuestionSpec,
    grade_diagnostic,
    grade_multiple_choice,
    grade_multiple_choice_batch,
    summarise_for_learner,
    to_observations,
)
from pathwise.services.assessment.selection import (
    DIFFICULTY_BANDS,
    select_probes,
)
from pathwise.services.knowledge.graph import GraphEdge, GraphNode, KnowledgeGraph
from pathwise.services.knowledge.mastery import Observation, rebuild
from pathwise.services.knowledge.seed import (
    build_graph_from_corpus,
    concept_id_for,
    load_corpus,
)
from pathwise.services.roadmap.planner import (
    COMPRESSION_MASTERY_THRESHOLD,
    MIN_REVIEW_MINUTES,
    plan_roadmap,
)

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def cid(slug: str) -> UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, slug)


def node(slug: str, difficulty: int = 3, minutes: int = 120) -> GraphNode:
    return GraphNode(
        id=cid(slug),
        slug=slug,
        name=slug.replace("-", " ").title(),
        difficulty=difficulty,
        estimated_minutes=minutes,
        domain="test",
    )


def prereq(source: str, target: str) -> GraphEdge:
    return GraphEdge(cid(source), cid(target), RelationType.PREREQUISITE_OF, 1.0)


def answer(slug: str, score: float) -> GradedAnswer:
    return GradedAnswer(
        question_id=uuid.uuid4(),
        concept_ids=(cid(slug),),
        objective_ids=("lo-1",),
        score=score,
        grader="deterministic",
    )


@pytest.fixture
def chain() -> KnowledgeGraph:
    """easy -> medium -> hard, spanning all three difficulty bands."""
    return KnowledgeGraph(
        [node("easy", 1), node("medium", 3), node("hard", 5)],
        [prereq("easy", "medium"), prereq("medium", "hard")],
    )


@pytest.fixture(scope="module")
def seed_graph() -> KnowledgeGraph:
    return build_graph_from_corpus(load_corpus())


# --------------------------------------------------------------------------- #
# Probe selection
# --------------------------------------------------------------------------- #


def test_selection_returns_the_requested_number_of_probes(chain: KnowledgeGraph) -> None:
    blueprint = select_probes(chain, chain.node_ids, question_count=3)
    assert blueprint.question_count == 3


def test_selection_never_repeats_a_concept(chain: KnowledgeGraph) -> None:
    blueprint = select_probes(chain, chain.node_ids, question_count=3)
    slugs = [target.slug for target in blueprint.targets]
    assert len(slugs) == len(set(slugs))


def test_selection_is_deterministic(seed_graph: KnowledgeGraph) -> None:
    """Two learners with the same goal get the same diagnostic, and regenerating one
    produces the same questions rather than a subtly different test."""
    goal = concept_id_for("ml-system-design")
    scope = [goal, *seed_graph.prerequisite_closure(goal)]
    first = select_probes(seed_graph, scope, question_count=10)
    second = select_probes(seed_graph, scope, question_count=10)
    assert [t.slug for t in first.targets] == [t.slug for t in second.targets]


def test_a_short_diagnostic_covers_most_of_a_long_path(seed_graph: KnowledgeGraph) -> None:
    """The point of the whole module: 10 questions, 39 concepts, because success on a
    downstream concept is evidence for everything beneath it."""
    goal = concept_id_for("ml-system-design")
    scope = [goal, *seed_graph.prerequisite_closure(goal)]
    blueprint = select_probes(seed_graph, scope, question_count=10)

    assert blueprint.candidate_count > 30
    assert blueprint.coverage_ratio > 0.8


def test_probes_are_spread_across_difficulty_bands(seed_graph: KnowledgeGraph) -> None:
    """Pure coverage-greedy would ask only the hardest questions, which a beginner
    fails entirely — telling you they are a beginner and nothing else."""
    goal = concept_id_for("ml-system-design")
    scope = [goal, *seed_graph.prerequisite_closure(goal)]
    counts = select_probes(seed_graph, scope, question_count=10).band_counts()

    for band, _, _ in DIFFICULTY_BANDS:
        assert counts[band] >= 1, f"no probe in the {band} band"


def test_coverage_never_counts_concepts_outside_the_goal(
    seed_graph: KnowledgeGraph,
) -> None:
    """A probe's prerequisites can reach outside the closure; counting those would
    inflate the reported coverage with concepts nobody asked to learn."""
    goal = concept_id_for("backpropagation")
    scope = [goal, *seed_graph.prerequisite_closure(goal)]
    blueprint = select_probes(seed_graph, scope, question_count=6)
    assert blueprint.covered <= blueprint.candidates


def test_uncovered_concepts_are_reported(seed_graph: KnowledgeGraph) -> None:
    """They keep their "no evidence" state — a short diagnostic never causes anything
    to be assumed, only measured."""
    goal = concept_id_for("ml-system-design")
    scope = [goal, *seed_graph.prerequisite_closure(goal)]
    blueprint = select_probes(seed_graph, scope, question_count=3)
    assert blueprint.uncovered
    assert blueprint.uncovered & blueprint.covered == frozenset()


def test_already_measured_concepts_are_not_re_tested(chain: KnowledgeGraph) -> None:
    """Spending a question to learn what is already known is pure waste."""
    blueprint = select_probes(chain, chain.node_ids, question_count=2, already_known=[cid("easy")])
    assert cid("easy") not in {t.concept_id for t in blueprint.targets}


def test_nothing_left_to_assess_is_an_error(chain: KnowledgeGraph) -> None:
    with pytest.raises(ValidationError, match="no concepts left"):
        select_probes(chain, chain.node_ids, already_known=chain.node_ids)


def test_a_zero_question_diagnostic_is_rejected(chain: KnowledgeGraph) -> None:
    with pytest.raises(ValidationError, match="at least one question"):
        select_probes(chain, chain.node_ids, question_count=0)


def test_asking_for_more_questions_than_concepts_is_bounded(
    chain: KnowledgeGraph,
) -> None:
    """Three concepts cannot yield twenty distinct questions."""
    blueprint = select_probes(chain, chain.node_ids, question_count=20)
    assert blueprint.question_count == 3


def test_estimated_time_is_reported(chain: KnowledgeGraph) -> None:
    assert select_probes(chain, chain.node_ids, question_count=3).estimated_minutes > 0


# --------------------------------------------------------------------------- #
# Deterministic grading
# --------------------------------------------------------------------------- #


def test_a_correct_choice_scores_full_marks() -> None:
    question = QuestionSpec(uuid.uuid4(), (cid("x"),), correct_option="b")
    assert grade_multiple_choice(question, "b").score == 1.0


def test_a_wrong_choice_scores_zero() -> None:
    question = QuestionSpec(uuid.uuid4(), (cid("x"),), correct_option="b")
    assert grade_multiple_choice(question, "c").score == 0.0


def test_grading_tolerates_case_and_whitespace() -> None:
    """A learner who answered "A" should not be marked wrong because the key says "a"."""
    question = QuestionSpec(uuid.uuid4(), (cid("x"),), correct_option="a")
    assert grade_multiple_choice(question, " A ").score == 1.0


def test_grading_uses_no_model_call() -> None:
    """Comparing two strings with an LLM would add cost, latency, and variance to the
    one part of assessment that has none."""
    question = QuestionSpec(uuid.uuid4(), (cid("x"),), correct_option="a")
    assert grade_multiple_choice(question, "a").grader == "deterministic"


def test_an_unanswered_question_scores_zero() -> None:
    """Skipping it instead would let a learner improve their estimate by leaving hard
    questions blank."""
    questions = [QuestionSpec(uuid.uuid4(), (cid("x"),), correct_option="a")]
    graded = grade_multiple_choice_batch(questions, {})
    assert graded[0].score == 0.0


def test_a_question_with_no_key_is_never_marked_correct() -> None:
    question = QuestionSpec(uuid.uuid4(), (cid("x"),), correct_option=None)
    assert grade_multiple_choice(question, "anything").score == 0.0


# --------------------------------------------------------------------------- #
# Answers into evidence
# --------------------------------------------------------------------------- #


def test_answers_become_observations() -> None:
    observations = to_observations(
        [answer("x", 1.0)], occurred_at=NOW, source=EvidenceSource.ASSESSMENT
    )
    assert len(observations) == 1
    assert observations[0].concept_id == cid("x")
    assert observations[0].score == 1.0


def test_repeated_questions_on_one_concept_become_a_single_observation() -> None:
    """Three questions are one better-supported measurement, not three independent
    ones — treating them separately would inflate confidence by how often we asked."""
    observations = to_observations(
        [answer("x", 1.0), answer("x", 0.0)],
        occurred_at=NOW,
        source=EvidenceSource.ASSESSMENT,
    )
    assert len(observations) == 1
    assert observations[0].score == pytest.approx(0.5)


def test_more_questions_carry_more_weight() -> None:
    one = to_observations([answer("x", 1.0)], occurred_at=NOW, source=EvidenceSource.QUIZ)
    three = to_observations([answer("x", 1.0)] * 3, occurred_at=NOW, source=EvidenceSource.QUIZ)
    assert three[0].weight_multiplier > one[0].weight_multiplier


def test_a_diagnostic_probe_carries_full_weight() -> None:
    """A diagnostic asks one deliberately-chosen question per concept. Scaling that
    down as if it were incidental would mean no diagnostic could ever move an
    estimate far enough to change the roadmap."""
    observations = to_observations(
        [answer("x", 1.0)],
        occurred_at=NOW,
        source=EvidenceSource.ASSESSMENT,
        full_weight_at=DIAGNOSTIC_FULL_WEIGHT_QUESTIONS,
    )
    assert observations[0].weight_multiplier == pytest.approx(1.0)


def test_observation_order_is_deterministic() -> None:
    first = to_observations(
        [answer("b", 1.0), answer("a", 1.0)], occurred_at=NOW, source=EvidenceSource.QUIZ
    )
    second = to_observations(
        [answer("a", 1.0), answer("b", 1.0)], occurred_at=NOW, source=EvidenceSource.QUIZ
    )
    assert [o.concept_id for o in first] == [o.concept_id for o in second]


# --------------------------------------------------------------------------- #
# The diagnostic as a whole
# --------------------------------------------------------------------------- #


def test_success_propagates_to_prerequisites(chain: KnowledgeGraph) -> None:
    """Why 10 questions can say something about 39 concepts."""
    outcome = grade_diagnostic(chain, [answer("hard", 1.0)], occurred_at=NOW)
    assert outcome.propagated
    assert cid("medium") in outcome.estimates
    assert cid("hard") in outcome.estimates


def test_failure_does_not_propagate(chain: KnowledgeGraph) -> None:
    """A wrong answer says something is missing but not what. Guessing would corrupt
    the prerequisite estimates blame attribution later depends on."""
    outcome = grade_diagnostic(chain, [answer("hard", 0.0)], occurred_at=NOW)
    assert outcome.propagated == ()


def test_a_diagnostic_measures_more_concepts_than_it_asks_about(
    seed_graph: KnowledgeGraph,
) -> None:
    goal = concept_id_for("ml-system-design")
    scope = [goal, *seed_graph.prerequisite_closure(goal)]
    blueprint = select_probes(seed_graph, scope, question_count=10)

    outcome = grade_diagnostic(
        seed_graph,
        [
            GradedAnswer(uuid.uuid4(), (target.concept_id,), ("lo-1",), 1.0, "deterministic")
            for target in blueprint.targets
        ],
        occurred_at=NOW,
    )
    assert len(outcome.estimates) > len(outcome.answers) * 2


def test_the_overall_score_reflects_the_answers(chain: KnowledgeGraph) -> None:
    outcome = grade_diagnostic(chain, [answer("easy", 1.0), answer("medium", 0.0)], occurred_at=NOW)
    assert outcome.overall_score == pytest.approx(0.5)


def test_per_objective_scores_are_available(chain: KnowledgeGraph) -> None:
    """A score of 48% is not evidence; "missed the chain-rule objective" is."""
    outcome = grade_diagnostic(chain, [answer("easy", 0.0)], occurred_at=NOW)
    assert outcome.objective_scores() == {"lo-1": 0.0}


def test_the_summary_ranks_strengths_and_weaknesses(chain: KnowledgeGraph) -> None:
    outcome = grade_diagnostic(chain, [answer("easy", 1.0), answer("hard", 0.0)], occurred_at=NOW)
    summary = summarise_for_learner(outcome, chain)
    assert summary["weakest"][0]["slug"] == "hard"  # type: ignore[index]
    assert summary["strongest"][0]["slug"] == "easy"  # type: ignore[index]


def test_an_empty_diagnostic_does_not_divide_by_zero(chain: KnowledgeGraph) -> None:
    outcome = grade_diagnostic(chain, [], occurred_at=NOW)
    assert outcome.overall_score == 0.0
    assert outcome.estimates == {}


# --------------------------------------------------------------------------- #
# Compression: the tier a diagnostic can actually earn
# --------------------------------------------------------------------------- #


def test_a_diagnostic_compresses_rather_than_skips(seed_graph: KnowledgeGraph) -> None:
    """The calibration that makes the diagnostic worth taking. One correct answer
    earns a shorter review; it must not delete a prerequisite outright."""
    goal = concept_id_for("ml-system-design")
    scope = [goal, *seed_graph.prerequisite_closure(goal)]
    blueprint = select_probes(seed_graph, scope, question_count=10)

    outcome = grade_diagnostic(
        seed_graph,
        [
            GradedAnswer(uuid.uuid4(), (t.concept_id,), ("lo-1",), 1.0, "deterministic")
            for t in blueprint.targets
        ],
        occurred_at=NOW,
    )
    plan = plan_roadmap(seed_graph, [goal], outcome.estimates, hours_per_week=8.0, now=NOW)

    assert plan.compressed, "a passed diagnostic must shorten something"
    assert not plan.skipped, "one question per concept must not delete a prerequisite"


def test_compression_measurably_shortens_the_path(seed_graph: KnowledgeGraph) -> None:
    goal = concept_id_for("ml-system-design")
    scope = [goal, *seed_graph.prerequisite_closure(goal)]
    blueprint = select_probes(seed_graph, scope, question_count=10)

    outcome = grade_diagnostic(
        seed_graph,
        [
            GradedAnswer(uuid.uuid4(), (t.concept_id,), ("lo-1",), 1.0, "deterministic")
            for t in blueprint.targets
        ],
        occurred_at=NOW,
    )
    baseline = plan_roadmap(seed_graph, [goal], hours_per_week=8.0, now=NOW)
    after = plan_roadmap(seed_graph, [goal], outcome.estimates, hours_per_week=8.0, now=NOW)

    assert after.pacing.total_minutes < baseline.pacing.total_minutes
    # The step count is unchanged: compression shortens steps, it does not remove them.
    assert len(after.nodes) == len(baseline.nodes)


def test_sustained_evidence_skips_instead_of_compressing(
    seed_graph: KnowledgeGraph,
) -> None:
    """The other tier. Repeated demonstration earns removal, not just a shorter step."""
    slug = "derivatives"
    estimates = {
        concept_id_for(slug): rebuild(
            [
                Observation(concept_id_for(slug), EvidenceSource.ASSESSMENT, 1.0, NOW)
                for _ in range(8)
            ]
        )
    }
    plan = plan_roadmap(seed_graph, [concept_id_for("ml-system-design")], estimates, now=NOW)
    assert [s.slug for s in plan.skipped] == [slug]
    assert slug not in {c.slug for c in plan.compressed}


def test_a_compressed_step_becomes_a_review_node(chain: KnowledgeGraph) -> None:
    estimates = {
        cid("easy"): rebuild([Observation(cid("easy"), EvidenceSource.ASSESSMENT, 1.0, NOW)])
    }
    plan = plan_roadmap(chain, [cid("hard")], estimates, now=NOW)

    easy_node = plan.node_for(cid("easy"))
    assert easy_node is not None
    assert easy_node.node_type is NodeType.REVIEW


def test_a_compressed_step_keeps_a_meaningful_minimum(chain: KnowledgeGraph) -> None:
    """Below a floor it is not a review, it is a formality."""
    tiny = KnowledgeGraph([node("tiny", 1, minutes=20), node("goal", 3)], [prereq("tiny", "goal")])
    estimates = {
        cid("tiny"): rebuild([Observation(cid("tiny"), EvidenceSource.ASSESSMENT, 1.0, NOW)])
    }
    plan = plan_roadmap(tiny, [cid("goal")], estimates, now=NOW)
    assert plan.compressed[0].review_minutes >= MIN_REVIEW_MINUTES


def test_compression_records_both_figures(chain: KnowledgeGraph) -> None:
    """Mastery and confidence together are what explain why this was compressed
    rather than skipped."""
    estimates = {
        cid("easy"): rebuild([Observation(cid("easy"), EvidenceSource.ASSESSMENT, 1.0, NOW)])
    }
    compressed = plan_roadmap(chain, [cid("hard")], estimates, now=NOW).compressed[0]

    assert compressed.mastery >= COMPRESSION_MASTERY_THRESHOLD
    assert compressed.confidence < 0.55  # below the skip bar, which is the whole point
    assert compressed.minutes_saved > 0


def test_a_failed_diagnostic_compresses_nothing(seed_graph: KnowledgeGraph) -> None:
    """Getting everything wrong must not shorten the path."""
    goal = concept_id_for("ml-system-design")
    scope = [goal, *seed_graph.prerequisite_closure(goal)]
    blueprint = select_probes(seed_graph, scope, question_count=10)

    outcome = grade_diagnostic(
        seed_graph,
        [
            GradedAnswer(uuid.uuid4(), (t.concept_id,), ("lo-1",), 0.0, "deterministic")
            for t in blueprint.targets
        ],
        occurred_at=NOW,
    )
    plan = plan_roadmap(seed_graph, [goal], outcome.estimates, hours_per_week=8.0, now=NOW)
    baseline = plan_roadmap(seed_graph, [goal], hours_per_week=8.0, now=NOW)

    assert plan.compressed == ()
    assert plan.skipped == ()
    assert plan.pacing.total_minutes == baseline.pacing.total_minutes


def test_the_goal_is_never_compressed(chain: KnowledgeGraph) -> None:
    """As with skipping: prior competence shortens the path *to* the goal, it does
    not delete the thing they asked to learn."""
    estimates = {
        cid("hard"): rebuild([Observation(cid("hard"), EvidenceSource.ASSESSMENT, 1.0, NOW)])
    }
    plan = plan_roadmap(chain, [cid("hard")], estimates, now=NOW)
    assert cid("hard") not in {c.concept_id for c in plan.compressed}
