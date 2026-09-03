"""The decision engine: what should this learner do next.

The single most important answer the product gives, and it is arithmetic. No model
is consulted. Candidates are filtered by prerequisite readiness, scored against
weighted factors, and ranked — and every term of every score is recorded in a
``DecisionTrace``.

That trace is the point. It is what makes the recommendation *explainable* rather
than merely *narrated*: the engine decides and emits its reasoning as data, then a
model renders that data into a sentence, and a validator checks the sentence cites
nothing the trace does not contain. Swap the model out and the decision is unchanged.

The factors, and why each earns its weight:

* **goal relevance** — how directly this leads to what they asked for. Without it the
  engine wanders into adjacent material that is interesting but off-path.
* **readiness** — how comfortably the prerequisites are met. A concept you barely
  qualify for is a concept you will struggle with.
* **review debt** — how far a previously-learned concept has decayed. Retention is
  worth more than coverage: relearning is cheaper than re-deriving.
* **difficulty fit** — distance between the concept's difficulty and the learner's
  demonstrated level. Both too-easy and too-hard waste time.
* **remediation** — a boost for concepts implicated in a recent failure. This is what
  makes the engine respond to struggle rather than march on regardless.
* **momentum** — a small preference for staying in the domain just worked in, so the
  path does not thrash between unrelated subjects.
* **deadline pressure** — raises the weight on goal relevance when time is short.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final
from uuid import UUID

from pathwise.services.knowledge.graph import KnowledgeGraph
from pathwise.services.knowledge.mastery import (
    MASTERY_THRESHOLD,
    REVIEW_THRESHOLD,
    MasteryEstimate,
)
from pathwise.services.roadmap.planner import PlannedNode, RoadmapPlan

#: Factor weights. These are the product's opinion about what matters, stated in one
#: place rather than smeared across conditionals. They sum to 1.0 so a total score is
#: interpretable on its own scale.
WEIGHTS: Final[dict[str, float]] = {
    "goal_relevance": 0.30,
    "readiness": 0.25,
    "review_debt": 0.20,
    "difficulty_fit": 0.10,
    "remediation": 0.10,
    "momentum": 0.05,
}

#: Under deadline pressure, goal relevance matters more and exploration matters less.
DEADLINE_PRESSURE_SHIFT: Final = 0.15

#: A concept this far past its review date is maximally overdue; beyond it the score
#: stops climbing, so one very stale concept cannot monopolise every recommendation.
MAX_REVIEW_OVERDUE_DAYS: Final = 30.0

#: Mastery margin above a prerequisite's requirement at which a learner counts as
#: comfortably ready. Readiness saturates here rather than scaling with the raw
#: margin: the question is "are you ready", not "by how much". Without saturation a
#: concept with no prerequisites always scores a perfect 1.0, so entry-point topics
#: outrank the concept the learner is actually aiming at — which is exactly the wrong
#: recommendation for someone who has already covered the basics.
READINESS_COMFORTABLE_MARGIN: Final = 0.25

#: Concept difficulty is expressed on a 1-5 scale throughout.
DIFFICULTY_SCALE_MAX: Final = 5

_SECONDS_PER_DAY: Final = 86_400.0


@dataclass(frozen=True, slots=True)
class FactorScore:
    """One weighted term in a decision."""

    name: str
    #: The raw measurement, on [0, 1].
    value: float
    weight: float
    #: A plain-language statement of what this measured. The explanation prompt is
    #: given these verbatim, so the prose can be checked against them.
    detail: str

    @property
    def contribution(self) -> float:
        return self.value * self.weight


@dataclass(frozen=True, slots=True)
class Candidate:
    """One thing the learner could do next, with the full reasoning behind its rank."""

    concept_id: UUID
    slug: str
    name: str
    estimated_minutes: int
    difficulty: int
    domain: str
    #: Why this is being recommended at all: new material, or overdue review.
    kind: str
    factors: tuple[FactorScore, ...]

    @property
    def score(self) -> float:
        return sum(factor.contribution for factor in self.factors)

    @property
    def dominant_factor(self) -> FactorScore:
        """The factor that actually drove this ranking.

        The explanation must lead with this. A recommendation justified by its
        third-largest term reads as a machine hedging.
        """
        return max(self.factors, key=lambda f: f.contribution)

    def factor(self, name: str) -> FactorScore | None:
        return next((f for f in self.factors if f.name == name), None)


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """Everything behind a recommendation, as data.

    Serialised into the explanation prompt and used to validate what comes back. Any
    figure in the generated prose that is not derivable from here is a fabrication,
    and `grounded_in_trace` rejects it.
    """

    recommended: Candidate | None
    alternatives: tuple[Candidate, ...]
    #: Candidates excluded before scoring, with the reason. Answers "why not X?"
    #: without re-running the engine.
    excluded: tuple[tuple[str, str], ...] = ()
    evaluated_at: datetime | None = None
    weights: Mapping[str, float] = field(default_factory=lambda: dict(WEIGHTS))

    @property
    def has_recommendation(self) -> bool:
        return self.recommended is not None

    @property
    def deciding_factor(self) -> FactorScore | None:
        """The factor that actually chose this candidate over the runner-up.

        Not the same as the largest contribution, and this is what an explanation
        must lead with. Readiness saturates for almost every viable candidate, so it
        is usually the biggest single term — meaning `dominant_factor` would have
        every recommendation open with "you're ready for it", which is true, useless,
        and identical every time.

        The deciding factor is where this candidate most out-scored the next best. If
        chain-rule beat probability-fundamentals on goal relevance, that is the
        sentence worth writing.
        """
        if self.recommended is None:
            return None
        if not self.alternatives:
            return self.recommended.dominant_factor

        runner_up = {f.name: f.contribution for f in self.alternatives[0].factors}
        return max(
            self.recommended.factors,
            key=lambda f: f.contribution - runner_up.get(f.name, 0.0),
        )

    def citable_numbers(self) -> tuple[float, ...]:
        """Every figure the explanation is allowed to quote.

        Handed to `grounded_in_trace`, which is what stops a plausible-sounding "you
        scored 72%" from reaching a learner when no such number exists.
        """
        numbers: list[float] = []
        for candidate in (self.recommended, *self.alternatives):
            if candidate is None:
                continue
            numbers.append(round(candidate.score, 3))
            numbers.append(float(candidate.estimated_minutes))
            numbers.append(round(candidate.estimated_minutes / 60, 1))
            numbers.append(float(candidate.difficulty))
            # The scale denominator, so "difficulty 3/5" is a grounded phrasing
            # rather than an apparent fabrication of the number 5.
            numbers.append(float(DIFFICULTY_SCALE_MAX))
            numbers.extend(round(factor.value, 3) for factor in candidate.factors)
        return tuple(numbers)

    def to_prompt_json(self) -> dict[str, object]:
        """The trace as the explanation prompt receives it."""
        if self.recommended is None:
            return {"recommended": None, "reason": "nothing is currently available"}
        return {
            "recommended": {
                "concept": self.recommended.slug,
                "name": self.recommended.name,
                "estimated_minutes": self.recommended.estimated_minutes,
                "difficulty": self.recommended.difficulty,
                "kind": self.recommended.kind,
                "total_score": round(self.recommended.score, 3),
                "dominant_factor": self.recommended.dominant_factor.name,
                # What an explanation must lead with — see `deciding_factor`.
                "deciding_factor": (self.deciding_factor.name if self.deciding_factor else None),
                "deciding_detail": (self.deciding_factor.detail if self.deciding_factor else None),
                "factors": [
                    {
                        "name": factor.name,
                        "value": round(factor.value, 3),
                        "weight": factor.weight,
                        "contribution": round(factor.contribution, 3),
                        "detail": factor.detail,
                    }
                    for factor in self.recommended.factors
                ],
            },
            "alternatives": [
                {"concept": c.slug, "total_score": round(c.score, 3)} for c in self.alternatives[:3]
            ],
        }


@dataclass(frozen=True, slots=True)
class LearnerContext:
    """Everything about the learner the engine reads."""

    mastery: Mapping[UUID, MasteryEstimate]
    goal_concept_ids: tuple[UUID, ...] = ()
    hours_per_week: float = 5.0
    #: Concepts implicated in a recent failure, from blame attribution. Boosted so a
    #: struggling learner is routed at the cause rather than marched onward.
    remediation_targets: frozenset[UUID] = frozenset()
    #: Domain of the last completed concept, for momentum.
    last_domain: str | None = None
    #: True when a deadline is at risk; shifts weight towards goal relevance.
    under_deadline_pressure: bool = False


def recommend_next(
    graph: KnowledgeGraph,
    plan: RoadmapPlan,
    context: LearnerContext,
    *,
    now: datetime,
    limit: int = 3,
) -> DecisionTrace:
    """Decide what the learner should do next.

    Deterministic: the same plan, mastery state, and clock always produce the same
    recommendation and the same trace.
    """
    weights = _weights_for(context)
    effective = {
        concept_id: estimate.effective_mastery(now)
        for concept_id, estimate in context.mastery.items()
    }

    candidates: list[Candidate] = []
    excluded: list[tuple[str, str]] = []

    for node in plan.nodes:
        reason = _exclusion_reason(graph, node, context, effective, now)
        if reason is not None:
            excluded.append((node.slug, reason))
            continue

        candidates.append(_score(graph, node, context, effective, weights, now, plan=plan))

    # Ties break on slug so a rerun never silently reorders equal candidates.
    candidates.sort(key=lambda c: (-c.score, c.slug))

    return DecisionTrace(
        recommended=candidates[0] if candidates else None,
        alternatives=tuple(candidates[1 : limit + 1]),
        excluded=tuple(sorted(excluded)),
        evaluated_at=now,
        weights=weights,
    )


def _weights_for(context: LearnerContext) -> dict[str, float]:
    """Adjust the weights for deadline pressure, keeping the total at 1.0.

    Renormalising matters: without it a pressured learner's scores would live on a
    different scale from an unpressured one, and the two would not be comparable in
    the trace or in any evaluation.
    """
    weights = dict(WEIGHTS)
    if not context.under_deadline_pressure:
        return weights

    weights["goal_relevance"] += DEADLINE_PRESSURE_SHIFT
    for name in ("difficulty_fit", "momentum", "review_debt"):
        weights[name] = max(0.0, weights[name] - DEADLINE_PRESSURE_SHIFT / 3)

    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def _exclusion_reason(
    graph: KnowledgeGraph,
    node: PlannedNode,
    context: LearnerContext,
    effective: Mapping[UUID, float],
    now: datetime,
) -> str | None:
    """Why this node cannot be recommended, or ``None`` if it can.

    Recorded rather than silently filtered, so "why not X?" is answerable from the
    trace instead of by re-deriving the engine's reasoning.
    """
    estimate = context.mastery.get(node.concept_id)
    mastery = effective.get(node.concept_id, 0.0)

    if estimate is not None and mastery >= MASTERY_THRESHOLD:
        if _is_due_for_review(estimate, now):
            return None  # mastered but decayed — a legitimate review candidate
        return "already mastered"

    readiness = graph.readiness(node.concept_id, effective)
    if not readiness.is_unlocked:
        blocking = ", ".join(graph.node(r.concept_id).slug for r in readiness.unmet[:3])
        return f"prerequisites not met ({blocking})"

    return None


def _is_due_for_review(estimate: MasteryEstimate, now: datetime) -> bool:
    due = estimate.review_due_at()
    return due is not None and due <= now


def _score(
    graph: KnowledgeGraph,
    node: PlannedNode,
    context: LearnerContext,
    effective: Mapping[UUID, float],
    weights: Mapping[str, float],
    now: datetime,
    *,
    plan: RoadmapPlan,
) -> Candidate:
    """Score one candidate, recording every term."""
    estimate = context.mastery.get(node.concept_id)
    is_review = estimate is not None and _is_due_for_review(estimate, now)

    factors = (
        _goal_relevance(graph, node, context, weights),
        _readiness(graph, node, effective, weights),
        _review_debt(estimate, now, weights, is_review=is_review),
        _difficulty_fit(node, context, effective, weights),
        _remediation(node, context, weights),
        _momentum(node, context, weights),
    )

    return Candidate(
        concept_id=node.concept_id,
        slug=node.slug,
        name=node.name,
        estimated_minutes=node.estimated_minutes,
        difficulty=node.difficulty,
        domain=node.domain,
        kind="review" if is_review else "new",
        factors=factors,
    )


def _goal_relevance(
    graph: KnowledgeGraph,
    node: PlannedNode,
    context: LearnerContext,
    weights: Mapping[str, float],
) -> FactorScore:
    """How directly this leads to what the learner asked for.

    Decays with graph distance rather than dropping to zero off-path: everything in a
    roadmap is on the path by construction, so this ranks *within* the path.
    """
    if not context.goal_concept_ids:
        return FactorScore(
            "goal_relevance", 0.5, weights["goal_relevance"], "no goal set, treated as neutral"
        )

    distances = graph.distance_to_goals(context.goal_concept_ids)
    distance = distances.get(node.concept_id)

    if distance is None:
        return FactorScore(
            "goal_relevance", 0.0, weights["goal_relevance"], "does not lead to the goal"
        )

    value = 1.0 / (1.0 + distance)
    return FactorScore(
        "goal_relevance",
        value,
        weights["goal_relevance"],
        f"{distance} step(s) from the goal",
    )


def _readiness(
    graph: KnowledgeGraph,
    node: PlannedNode,
    effective: Mapping[UUID, float],
    weights: Mapping[str, float],
) -> FactorScore:
    """Whether the learner is comfortably ready, not by how much.

    Saturating rather than linear, and the distinction matters. A linear score gives
    a concept with no prerequisites a perfect 1.0 while a concept whose sole
    prerequisite is mastered at 0.93 scores 0.33 — so an entry-point topic five steps
    off the goal outranks the one the learner is actually ready for and aiming at.
    Clearing the bar by 0.25 is as ready as clearing it by 0.9.
    """
    report = graph.readiness(node.concept_id, effective)
    if not report.unmet and report.margin >= 1.0:
        return FactorScore("readiness", 1.0, weights["readiness"], "no prerequisites to satisfy")

    value = max(0.0, min(1.0, report.margin / READINESS_COMFORTABLE_MARGIN))
    descriptor = "comfortably" if value >= 1.0 else "narrowly"
    return FactorScore(
        "readiness",
        value,
        weights["readiness"],
        f"{descriptor} clears its prerequisites (margin {report.margin:.2f})",
    )


def _review_debt(
    estimate: MasteryEstimate | None,
    now: datetime,
    weights: Mapping[str, float],
    *,
    is_review: bool,
) -> FactorScore:
    """How far a previously-learned concept has decayed.

    Only ever positive for material actually learned. New material scores zero here,
    which is correct — it cannot be overdue for review.
    """
    if estimate is None or not is_review:
        return FactorScore("review_debt", 0.0, weights["review_debt"], "not due for review")

    due = estimate.review_due_at()
    if due is None:
        return FactorScore("review_debt", 0.0, weights["review_debt"], "not due for review")

    overdue_days = (now - due).total_seconds() / _SECONDS_PER_DAY
    value = max(0.0, min(1.0, overdue_days / MAX_REVIEW_OVERDUE_DAYS))
    return FactorScore(
        "review_debt",
        value,
        weights["review_debt"],
        f"{overdue_days:.0f} day(s) overdue for review "
        f"(mastery has decayed to {estimate.effective_mastery(now):.2f})",
    )


def _difficulty_fit(
    node: PlannedNode,
    context: LearnerContext,
    effective: Mapping[UUID, float],
    weights: Mapping[str, float],
) -> FactorScore:
    """Distance between the concept's difficulty and the learner's demonstrated level.

    Both directions are penalised. Too hard stalls; too easy wastes time that a
    learner with a deadline does not have.
    """
    if not effective:
        return FactorScore(
            "difficulty_fit", 0.5, weights["difficulty_fit"], "no measured level yet"
        )

    mean_mastery = sum(effective.values()) / len(effective)
    # Map mastery onto the 1-5 difficulty scale the graph uses.
    learner_level = 1.0 + mean_mastery * 4.0
    gap = abs(node.difficulty - learner_level)
    value = max(0.0, 1.0 - gap / 4.0)

    return FactorScore(
        "difficulty_fit",
        value,
        weights["difficulty_fit"],
        f"difficulty {node.difficulty}/5 against a demonstrated level of {learner_level:.1f}/5",
    )


def _remediation(
    node: PlannedNode, context: LearnerContext, weights: Mapping[str, float]
) -> FactorScore:
    """A boost for concepts implicated in a recent failure.

    This is what makes the engine respond to struggle. Without it a learner who fails
    an assessment is simply marched on to the next topic, which is the behaviour the
    whole product exists to avoid.
    """
    if node.concept_id in context.remediation_targets:
        return FactorScore(
            "remediation",
            1.0,
            weights["remediation"],
            "identified as a likely cause of a recent difficulty",
        )
    return FactorScore("remediation", 0.0, weights["remediation"], "not a remediation target")


def _momentum(
    node: PlannedNode, context: LearnerContext, weights: Mapping[str, float]
) -> FactorScore:
    """A small preference for continuing in the domain just worked in.

    Deliberately the lightest factor. Enough to stop the path thrashing between
    unrelated subjects, not enough to keep a learner in one domain when something
    more important is waiting elsewhere.
    """
    if context.last_domain and node.domain == context.last_domain:
        return FactorScore("momentum", 1.0, weights["momentum"], f"continues in {node.domain}")
    return FactorScore("momentum", 0.0, weights["momentum"], "switches subject area")


def fallback_explanation(trace: DecisionTrace) -> str:
    """A deterministic explanation, for when generation fails.

    Plainer than generated prose, but every clause is read straight off the trace —
    so the learner is never left with a recommendation and no reason for it.
    """
    if trace.recommended is None:
        return (
            "There is nothing available to start right now — every remaining step is "
            "waiting on a prerequisite."
        )

    candidate = trace.recommended
    hours = candidate.estimated_minutes / 60
    lead = trace.deciding_factor or candidate.dominant_factor

    opening = (
        f"{candidate.name} is next"
        if candidate.kind == "new"
        else f"{candidate.name} is due for review"
    )
    return (
        f"{opening} — {lead.detail}. "
        f"About {hours:.1f} hours at difficulty {candidate.difficulty}/5."
    )


def review_candidates(estimates: Mapping[UUID, MasteryEstimate], now: datetime) -> tuple[UUID, ...]:
    """Concepts whose mastery has decayed below the review threshold.

    Exposed separately so the dashboard can show "needs review" without running a
    full decision pass.
    """
    due = [
        (concept_id, estimate.effective_mastery(now))
        for concept_id, estimate in estimates.items()
        if estimate.evidence_count > 0
        and estimate.mastery >= REVIEW_THRESHOLD
        and _is_due_for_review(estimate, now)
    ]
    due.sort(key=lambda item: (item[1], str(item[0])))
    return tuple(concept_id for concept_id, _ in due)


def summarise(trace: DecisionTrace, graph: KnowledgeGraph) -> dict[str, object]:
    """The dashboard's "what should I do next" payload."""
    if trace.recommended is None:
        return {"next": None, "reason": "nothing available", "blocked": len(trace.excluded)}

    candidate = trace.recommended
    return {
        "next": {
            "slug": candidate.slug,
            "name": candidate.name,
            "kind": candidate.kind,
            "estimated_minutes": candidate.estimated_minutes,
            "difficulty": candidate.difficulty,
        },
        "reason": (trace.deciding_factor or candidate.dominant_factor).detail,
        "score": round(candidate.score, 3),
        "alternatives": [c.slug for c in trace.alternatives],
    }
