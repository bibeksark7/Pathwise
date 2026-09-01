"""The mastery model.

Mastery is a Beta(alpha, beta) posterior over "would this learner succeed at a task
requiring this concept?", not a running average of scores. That choice buys three
things a percentage cannot:

* **Confidence is separable from level.** 0.80 from one quiz and 0.80 from twelve
  assessments are different states, and only the second justifies skipping material.
* **Evidence of different quality composes correctly.** A project counts more than a
  self-report because it contributes more pseudo-counts, not because of an ad-hoc
  weighting bolted on afterwards.
* **Order does not matter.** Beta updates commute, so replaying the same events in a
  different sequence yields the same state — which is what makes the append-only
  evidence log a genuine source of truth rather than an audit trail that has drifted.

Everything here is a pure function. No database, no clock reads (``now`` is always
passed in), no LLM. That is what makes the property tests below meaningful.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Final
from uuid import UUID

from pathwise.models.enums import EvidenceSource
from pathwise.services.knowledge.graph import KnowledgeGraph

# --------------------------------------------------------------------------- #
# Tuning constants. Every one of these is a product decision, so each is named
# and justified rather than buried as a magic number at its use site.
# --------------------------------------------------------------------------- #

#: How reliable each evidence source is as a signal of genuine understanding.
#: Building something working demonstrates transfer; saying you feel confident
#: about it demonstrates very little.
EVIDENCE_WEIGHTS: Final[Mapping[EvidenceSource, float]] = {
    EvidenceSource.PROJECT: 1.20,
    EvidenceSource.ASSESSMENT: 1.00,
    EvidenceSource.QUIZ: 0.80,
    EvidenceSource.LESSON: 0.25,
    EvidenceSource.TUTOR: 0.20,
    EvidenceSource.TIME_ON_TASK: 0.10,
    EvidenceSource.SELF_REPORT: 0.15,
    EvidenceSource.PROPAGATED: 0.35,
}

#: Pseudo-counts contributed by one unit-weight observation. Larger values make the
#: model move faster and trust individual results more.
PSEUDO_COUNT_SCALE: Final = 2.0

#: Weakly-informative prior: mean 0.5, effectively no confidence. A mastery row is
#: only created once real evidence exists, so this prior never stands alone —
#: "no evidence" is represented by the absence of a row, not by 0.5.
PRIOR_ALPHA: Final = 1.0
PRIOR_BETA: Final = 1.0

#: Pseudo-count mass at which confidence reaches 0.5. Two full-weight assessments
#: contribute 4.0 of mass, so that is the half-way point — roughly "I have seen you
#: do this twice". Confidence is deliberately derived from *how much* evidence exists
#: rather than from the posterior's spread: Beta variance also depends on where the
#: mean sits, so a variance-based measure would drop when a mixed result pulled the
#: estimate towards 0.5, reporting less confidence after more evidence.
CONFIDENCE_HALF_MASS: Final = 4.0

#: Forgetting. Retention is modelled as exponential decay towards a floor rather
#: than towards zero — well-learned material fades but does not vanish.
BASE_HALFLIFE_DAYS: Final = 30.0
RETENTION_FLOOR_RATIO: Final = 0.60
REVIEW_HALFLIFE_BONUS: Final = 0.75  # each successful review extends the half-life

#: Effective mastery below this is flagged for review.
REVIEW_THRESHOLD: Final = 0.70
#: Effective mastery at or above this counts as mastered; the decision engine will
#: not schedule further study of the concept.
MASTERY_THRESHOLD: Final = 0.75
#: Skipping material demands both a high level *and* enough evidence to trust it.
SKIP_MASTERY_THRESHOLD: Final = 0.85
SKIP_CONFIDENCE_THRESHOLD: Final = 0.55

#: Success on a concept is weak evidence for its prerequisites — you could not have
#: done it without them. Damped per hop, and capped at two hops because the
#: inference gets thin fast.
PROPAGATION_DAMPING: Final = 0.35
PROPAGATION_MAX_HOPS: Final = 2
#: Only clear success propagates. A mediocre score says nothing useful about
#: prerequisites, and a failure says nothing at all — that is blame attribution's job.
PROPAGATION_MIN_SCORE: Final = 0.75

_SECONDS_PER_DAY: Final = 86_400.0


@dataclass(frozen=True, slots=True)
class Observation:
    """One piece of evidence, in the form the mastery model consumes.

    Deliberately decoupled from the ``EvidenceEvent`` ORM row so the maths can be
    tested, replayed, and reasoned about without a database.
    """

    concept_id: UUID
    source: EvidenceSource
    score: float
    occurred_at: datetime
    #: Multiplies the source weight. Used for partial credit — an assessment with
    #: three questions on a concept is stronger evidence than one with a single
    #: question, and this is where that shows up.
    weight_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")
        if self.weight_multiplier < 0.0:
            raise ValueError(
                f"weight_multiplier must be non-negative, got {self.weight_multiplier}"
            )

    @property
    def effective_weight(self) -> float:
        """Source reliability combined with this observation's own strength."""
        return EVIDENCE_WEIGHTS.get(self.source, 0.5) * self.weight_multiplier


@dataclass(frozen=True, slots=True)
class MasteryEstimate:
    """A learner's posterior belief about one concept, at a point in time.

    ``mastery`` is the posterior mean and does not decay. ``effective_mastery(now)``
    applies forgetting. Storing the undecayed value keeps the state a pure function of
    the evidence log; decay is applied at read time, where the current clock lives.
    """

    alpha: float = PRIOR_ALPHA
    beta: float = PRIOR_BETA
    evidence_count: int = 0
    review_count: int = 0
    last_evidence_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(f"Beta parameters must be positive, got ({self.alpha}, {self.beta})")

    @property
    def mastery(self) -> float:
        """Posterior mean — the learner's level, ignoring forgetting."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1.0))

    @property
    def evidence_mass(self) -> float:
        """Total pseudo-counts accumulated beyond the prior.

        This is the amount of evidence, independent of what it said.
        """
        return max(0.0, (self.alpha - PRIOR_ALPHA) + (self.beta - PRIOR_BETA))

    @property
    def confidence(self) -> float:
        """How much to trust ``mastery``, on [0, 1].

        Monotonically increasing in evidence: zero at the prior, 0.5 at
        ``CONFIDENCE_HALF_MASS``, asymptotically one. Never decreases when evidence
        arrives, which is what lets it gate irreversible decisions like skipping
        material. It is a property of the record, not something the learner or the
        model can assert.
        """
        mass = self.evidence_mass
        return mass / (mass + CONFIDENCE_HALF_MASS)

    @property
    def is_mastered(self) -> bool:
        return self.mastery >= MASTERY_THRESHOLD

    @property
    def is_skippable(self) -> bool:
        """High level *and* enough evidence to act on it.

        The conjunction is the point: skipping material on a confident-looking score
        from a single lucky quiz is how an adaptive system strands a learner.
        """
        return (
            self.mastery >= SKIP_MASTERY_THRESHOLD and self.confidence >= SKIP_CONFIDENCE_THRESHOLD
        )

    def halflife_days(self) -> float:
        """How long until half the decayable portion of mastery is lost.

        Grows with mastery (deeply-learned material is more durable) and with the
        number of successful reviews (spacing effect).
        """
        return (
            BASE_HALFLIFE_DAYS
            * (1.0 + REVIEW_HALFLIFE_BONUS * self.review_count)
            * (0.5 + self.mastery)
        )

    def effective_mastery(self, now: datetime) -> float:
        """Mastery with forgetting applied — what every decision should read.

        Decays towards ``RETENTION_FLOOR_RATIO * mastery`` rather than towards zero.
        """
        if self.last_evidence_at is None or self.evidence_count == 0:
            return self.mastery

        elapsed_days = (now - self.last_evidence_at).total_seconds() / _SECONDS_PER_DAY
        if elapsed_days <= 0:
            return self.mastery

        floor = RETENTION_FLOOR_RATIO * self.mastery
        retained = math.pow(0.5, elapsed_days / self.halflife_days())
        decayed: float = floor + (self.mastery - floor) * retained
        return max(0.0, min(1.0, decayed))

    def review_due_at(self) -> datetime | None:
        """When effective mastery will fall to ``REVIEW_THRESHOLD``.

        Returns ``None`` when it never will — either because the retention floor sits
        above the threshold (the concept is durably known) or because there is no
        evidence to decay from. Solved analytically rather than by simulation, so a
        due date is exact and stable across recomputations.
        """
        if self.last_evidence_at is None or self.evidence_count == 0:
            return None

        floor = RETENTION_FLOOR_RATIO * self.mastery
        if floor >= REVIEW_THRESHOLD:
            return None  # never decays below the threshold
        if self.mastery <= REVIEW_THRESHOLD:
            return self.last_evidence_at  # already due

        # m(t) = floor + (m0 - floor) * 2^(-t / H)  =>  solve m(t) = REVIEW_THRESHOLD
        ratio = (REVIEW_THRESHOLD - floor) / (self.mastery - floor)
        days = -self.halflife_days() * math.log2(ratio)
        return self.last_evidence_at + timedelta(days=days)


@dataclass(frozen=True, slots=True)
class MasteryDelta:
    """What one observation did to an estimate — the material for an explanation.

    Returned alongside the new state so the UI can say "that assessment moved
    probability from 0.62 to 0.48" using stored numbers rather than a regenerated
    guess.
    """

    concept_id: UUID
    before: float
    after: float
    before_confidence: float
    after_confidence: float
    source: EvidenceSource
    observation_score: float

    @property
    def change(self) -> float:
        return self.after - self.before


def apply_observation(
    estimate: MasteryEstimate, observation: Observation
) -> tuple[MasteryEstimate, MasteryDelta]:
    """Fold one observation into an estimate.

    A score of ``s`` with effective weight ``w`` contributes ``w * s * SCALE``
    successes and ``w * (1 - s) * SCALE`` failures. A zero-weight observation is
    recorded in the count but moves nothing, which keeps "we saw this but it told us
    nothing" distinguishable from "we saw nothing".
    """
    weight = observation.effective_weight
    successes = weight * observation.score * PSEUDO_COUNT_SCALE
    failures = weight * (1.0 - observation.score) * PSEUDO_COUNT_SCALE

    # A review only counts when the learner actually demonstrated retention;
    # failing a review must not extend the half-life.
    is_successful_review = (
        estimate.evidence_count > 0 and observation.score >= PROPAGATION_MIN_SCORE
    )

    updated = MasteryEstimate(
        alpha=estimate.alpha + successes,
        beta=estimate.beta + failures,
        evidence_count=estimate.evidence_count + 1,
        review_count=estimate.review_count + (1 if is_successful_review else 0),
        last_evidence_at=_latest(estimate.last_evidence_at, observation.occurred_at),
    )

    delta = MasteryDelta(
        concept_id=observation.concept_id,
        before=estimate.mastery,
        after=updated.mastery,
        before_confidence=estimate.confidence,
        after_confidence=updated.confidence,
        source=observation.source,
        observation_score=observation.score,
    )
    return updated, delta


def rebuild(observations: Iterable[Observation]) -> MasteryEstimate:
    """Replay an evidence log into an estimate from the prior.

    This is what makes changing the mastery algorithm safe: recompute every learner's
    state from stored events instead of leaving historical estimates computed under
    the old rules.
    """
    estimate = MasteryEstimate()
    for observation in observations:
        estimate, _ = apply_observation(estimate, observation)
    return estimate


def rebuild_all(
    observations: Iterable[Observation],
) -> dict[UUID, MasteryEstimate]:
    """Replay a mixed evidence log, grouping by concept."""
    estimates: dict[UUID, MasteryEstimate] = {}
    for observation in observations:
        current = estimates.get(observation.concept_id, MasteryEstimate())
        estimates[observation.concept_id], _ = apply_observation(current, observation)
    return estimates


def propagate_to_prerequisites(
    graph: KnowledgeGraph,
    observation: Observation,
    *,
    max_hops: int = PROPAGATION_MAX_HOPS,
) -> tuple[Observation, ...]:
    """Derive weak evidence about prerequisites from success on a dependent concept.

    Succeeding at neural networks is real evidence that you can do the calculus
    underneath it. This is what lets Pathwise *reduce* a roadmap: a learner who
    demonstrates a downstream skill does not need to be marched through everything
    beneath it.

    Only clear success propagates (``score >= PROPAGATION_MIN_SCORE``). A failure
    tells us something is wrong but not *where*, and guessing would corrupt exactly
    the prerequisite estimates that blame attribution needs to stay clean.

    The derived score is attenuated towards neutral rather than copied, because the
    inference is weaker than the observation that produced it.
    """
    if observation.score < PROPAGATION_MIN_SCORE:
        return ()
    if observation.concept_id not in graph:
        return ()

    derived: list[Observation] = []
    for requirement in graph.prerequisite_closure(
        observation.concept_id, max_depth=max_hops
    ).values():
        if requirement.hops > max_hops:
            continue

        # Hop 1 carries the observation at full score; each further hop pulls the
        # score towards neutral and shrinks its weight.
        hop_damping = PROPAGATION_DAMPING ** (requirement.hops - 1)
        derived_score = 0.5 + (observation.score - 0.5) * hop_damping

        derived.append(
            Observation(
                concept_id=requirement.concept_id,
                source=EvidenceSource.PROPAGATED,
                score=max(0.0, min(1.0, derived_score)),
                occurred_at=observation.occurred_at,
                weight_multiplier=hop_damping * requirement.strength,
            )
        )

    return tuple(derived)


def effective_mastery_map(
    estimates: Mapping[UUID, MasteryEstimate], now: datetime
) -> dict[UUID, float]:
    """Decay-adjusted mastery for every concept, ready for graph queries."""
    return {
        concept_id: estimate.effective_mastery(now) for concept_id, estimate in estimates.items()
    }


def confidence_map(estimates: Mapping[UUID, MasteryEstimate]) -> dict[UUID, float]:
    """Confidence for every concept, ready for blame attribution."""
    return {concept_id: estimate.confidence for concept_id, estimate in estimates.items()}


def concepts_due_for_review(
    estimates: Mapping[UUID, MasteryEstimate], now: datetime
) -> tuple[UUID, ...]:
    """Concepts whose effective mastery has decayed below the review threshold.

    Ordered by how far they have slipped, so the most degraded is reviewed first.
    """
    due = [
        (concept_id, estimate.effective_mastery(now))
        for concept_id, estimate in estimates.items()
        if estimate.evidence_count > 0
        and estimate.mastery >= REVIEW_THRESHOLD
        and estimate.effective_mastery(now) < REVIEW_THRESHOLD
    ]
    due.sort(key=lambda item: (item[1], str(item[0])))
    return tuple(concept_id for concept_id, _ in due)


def weakest_concepts(
    estimates: Mapping[UUID, MasteryEstimate],
    now: datetime,
    *,
    limit: int = 5,
    min_evidence: int = 1,
) -> tuple[tuple[UUID, float], ...]:
    """The learner's weakest measured concepts, for the dashboard's "weak areas".

    Requires at least ``min_evidence`` observations: a concept never studied is not a
    weakness, it is simply not yet reached, and conflating the two makes the dashboard
    useless in week one.
    """
    scored = [
        (concept_id, estimate.effective_mastery(now))
        for concept_id, estimate in estimates.items()
        if estimate.evidence_count >= min_evidence
    ]
    scored.sort(key=lambda item: (item[1], str(item[0])))
    return tuple(scored[:limit])


def aggregate_by_domain(
    estimates: Mapping[UUID, MasteryEstimate],
    graph: KnowledgeGraph,
    now: datetime,
) -> dict[str, float]:
    """Mean effective mastery per domain — the dashboard's "mastery by subject".

    Weighted by concept difficulty so that competence in hard concepts counts for
    more than ticking off easy ones.
    """
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}

    for concept_id, estimate in estimates.items():
        if concept_id not in graph:
            continue
        node = graph.node(concept_id)
        weight = float(node.difficulty)
        totals[node.domain] = (
            totals.get(node.domain, 0.0) + estimate.effective_mastery(now) * weight
        )
        weights[node.domain] = weights.get(node.domain, 0.0) + weight

    return {domain: totals[domain] / weights[domain] for domain in totals if weights[domain] > 0}


def _latest(current: datetime | None, candidate: datetime) -> datetime:
    """Keep the most recent timestamp.

    Out-of-order arrival must not rewind ``last_evidence_at``, or a late-arriving old
    event would make a freshly-assessed concept look stale and due for review.
    """
    return candidate if current is None or candidate > current else current


def merge_observations(*groups: Sequence[Observation]) -> tuple[Observation, ...]:
    """Combine observation groups into one chronologically ordered sequence."""
    merged = [observation for group in groups for observation in group]
    merged.sort(key=lambda o: (o.occurred_at, str(o.concept_id)))
    return tuple(merged)


def with_review(estimate: MasteryEstimate) -> MasteryEstimate:
    """Record a completed review without changing the level.

    Used when a learner revisits material and the system wants to extend retention
    without treating the revisit as a fresh measurement.
    """
    return replace(estimate, review_count=estimate.review_count + 1)
