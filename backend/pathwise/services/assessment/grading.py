"""Turning answers into evidence.

This closes the loop the whole system runs on:

    diagnostic answers -> evidence events -> mastery estimates -> a shorter roadmap

Multiple choice is graded deterministically — there is no reason to spend a model
call comparing two strings, and doing so would introduce variance into the one part
of assessment that has none. Open responses need a rubric grader, which lives
separately; this module handles the deterministic half and the conversion to evidence
for both.

Two decisions worth stating.

**A single question is thin evidence.** One correct answer about gradient descent is
not the same as four. Weight scales with how many questions actually targeted a
concept, so the mastery model's confidence reflects how much was really asked.

**Propagation happens here, not in the mastery model.** Success on a downstream
concept credits its prerequisites — which is what let the diagnostic cover 39
concepts with 10 questions. Generating those derived observations at grading time
keeps them in the evidence log, so the inference is auditable rather than implicit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from pathwise.models.enums import EvidenceSource
from pathwise.services.knowledge.graph import KnowledgeGraph
from pathwise.services.knowledge.mastery import (
    MasteryEstimate,
    Observation,
    merge_observations,
    propagate_to_prerequisites,
    rebuild_all,
)

#: A concept asked about this many times is considered thoroughly probed; more
#: questions stop adding weight. Prevents one over-represented concept from
#: dominating an estimate.
FULL_WEIGHT_QUESTION_COUNT = 3

#: A diagnostic asks exactly one deliberately-chosen question per concept, selected
#: to be maximally informative about it. Scaling that down to a third — as if it were
#: an incidental question that happened to touch the topic — would understate the one
#: measurement the diagnostic exists to make, and no diagnostic could ever move a
#: mastery estimate far enough to change the roadmap.
DIAGNOSTIC_FULL_WEIGHT_QUESTIONS = 1


@dataclass(frozen=True, slots=True)
class QuestionSpec:
    """The gradeable part of a question.

    Deliberately independent of the ORM row so grading can be tested, replayed, and
    reasoned about without a database.
    """

    question_id: UUID
    concept_ids: tuple[UUID, ...]
    #: Which learning objectives this question measures. The reason a wrong answer
    #: can say "missed the chain-rule objective" rather than just "scored 48%".
    objective_ids: tuple[str, ...] = ()
    #: The correct option id, for multiple choice. ``None`` for open responses.
    correct_option: str | None = None
    points: float = 1.0


@dataclass(frozen=True, slots=True)
class GradedAnswer:
    """One scored response."""

    question_id: UUID
    concept_ids: tuple[UUID, ...]
    objective_ids: tuple[str, ...]
    score: float
    grader: str
    #: Only set by a rubric grader. Deterministic grading has nothing to say here.
    misconceptions: tuple[str, ...] = ()

    @property
    def is_correct(self) -> bool:
        return self.score >= 0.999


@dataclass(frozen=True, slots=True)
class DiagnosticOutcome:
    """Everything a completed diagnostic produced."""

    answers: tuple[GradedAnswer, ...]
    #: Direct observations, before propagation.
    observations: tuple[Observation, ...]
    #: Observations derived from prerequisite propagation.
    propagated: tuple[Observation, ...]
    estimates: Mapping[UUID, MasteryEstimate] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Mean score across answers. A headline number, not an input to anything."""
        if not self.answers:
            return 0.0
        return sum(answer.score for answer in self.answers) / len(self.answers)

    @property
    def all_observations(self) -> tuple[Observation, ...]:
        return merge_observations(self.observations, self.propagated)

    def concept_scores(self) -> dict[UUID, float]:
        """Mean raw score per directly-tested concept."""
        totals: dict[UUID, list[float]] = {}
        for answer in self.answers:
            for concept_id in answer.concept_ids:
                totals.setdefault(concept_id, []).append(answer.score)
        return {cid: sum(scores) / len(scores) for cid, scores in totals.items()}

    def objective_scores(self) -> dict[str, float]:
        """Mean score per learning objective — the per-capability breakdown."""
        totals: dict[str, list[float]] = {}
        for answer in self.answers:
            for objective_id in answer.objective_ids:
                totals.setdefault(objective_id, []).append(answer.score)
        return {oid: sum(scores) / len(scores) for oid, scores in totals.items()}


def grade_multiple_choice(question: QuestionSpec, response: str) -> GradedAnswer:
    """Score a multiple-choice answer. All or nothing, no model call.

    Whitespace and case are normalised: a learner who answered "A" should not be
    marked wrong because the key stores "a".
    """
    expected = (question.correct_option or "").strip().lower()
    given = (response or "").strip().lower()
    score = 1.0 if expected and given == expected else 0.0

    return GradedAnswer(
        question_id=question.question_id,
        concept_ids=question.concept_ids,
        objective_ids=question.objective_ids,
        score=score,
        grader="deterministic",
    )


def grade_multiple_choice_batch(
    questions: Iterable[QuestionSpec], responses: Mapping[UUID, str]
) -> tuple[GradedAnswer, ...]:
    """Grade a whole submission.

    An unanswered question scores zero rather than being skipped: skipping would let
    a learner improve their estimate by leaving hard questions blank.
    """
    return tuple(
        grade_multiple_choice(question, responses.get(question.question_id, ""))
        for question in questions
    )


def to_observations(
    answers: Sequence[GradedAnswer],
    *,
    occurred_at: datetime,
    source: EvidenceSource,
    full_weight_at: int = FULL_WEIGHT_QUESTION_COUNT,
) -> tuple[Observation, ...]:
    """Convert graded answers into weighted observations.

    Answers about the same concept are combined into one observation rather than
    several: three questions on gradient descent are a single, better-supported
    measurement, not three independent ones. Treating them independently would
    inflate confidence in proportion to how often we happened to ask.
    """
    grouped: dict[UUID, list[float]] = {}
    for answer in answers:
        for concept_id in answer.concept_ids:
            grouped.setdefault(concept_id, []).append(answer.score)

    observations: list[Observation] = []
    for concept_id, scores in grouped.items():
        question_count = len(scores)
        mean_score = sum(scores) / question_count
        # Weight rises with how much was actually asked, then plateaus.
        weight = min(question_count, full_weight_at) / full_weight_at

        observations.append(
            Observation(
                concept_id=concept_id,
                source=source,
                score=mean_score,
                occurred_at=occurred_at,
                weight_multiplier=weight,
            )
        )

    observations.sort(key=lambda o: str(o.concept_id))
    return tuple(observations)


def grade_diagnostic(
    graph: KnowledgeGraph,
    answers: Sequence[GradedAnswer],
    *,
    occurred_at: datetime,
    source: EvidenceSource = EvidenceSource.ASSESSMENT,
) -> DiagnosticOutcome:
    """Produce evidence and mastery estimates from a completed diagnostic.

    The propagation step is what makes a 10-question diagnostic able to say something
    about 39 concepts: a correct answer about PyTorch is real evidence that the
    learner has the NumPy and neural-network fundamentals beneath it.

    Propagation only ever fires on clear success. A wrong answer says something is
    missing but not *what*, and guessing would corrupt precisely the prerequisite
    estimates that blame attribution depends on later.
    """
    observations = to_observations(
        answers,
        occurred_at=occurred_at,
        source=source,
        full_weight_at=DIAGNOSTIC_FULL_WEIGHT_QUESTIONS,
    )

    propagated: list[Observation] = []
    for observation in observations:
        propagated.extend(propagate_to_prerequisites(graph, observation))

    estimates = rebuild_all(merge_observations(observations, tuple(propagated)))

    return DiagnosticOutcome(
        answers=tuple(answers),
        observations=observations,
        propagated=tuple(propagated),
        estimates=estimates,
    )


def summarise_for_learner(
    outcome: DiagnosticOutcome, graph: KnowledgeGraph, *, limit: int = 5
) -> dict[str, object]:
    """A plain summary of what the diagnostic found.

    Built entirely from the computed estimates — no generation. The numbers a learner
    is shown here are the same ones the planner acts on, so the two can never
    disagree.
    """
    scores = outcome.concept_scores()
    ranked = sorted(
        ((cid, score) for cid, score in scores.items() if cid in graph),
        key=lambda item: item[1],
    )

    return {
        "overall_score": round(outcome.overall_score, 3),
        "questions_answered": len(outcome.answers),
        "concepts_measured": len(outcome.estimates),
        "concepts_directly_tested": len(scores),
        "strongest": [
            {"slug": graph.node(cid).slug, "score": round(score, 2)}
            for cid, score in reversed(ranked[-limit:])
        ],
        "weakest": [
            {"slug": graph.node(cid).slug, "score": round(score, 2)}
            for cid, score in ranked[:limit]
        ],
    }
