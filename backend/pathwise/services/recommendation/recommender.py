"""Recommending resources for a concept.

A three-stage funnel, and the division of labour is the point:

1. **Filter** — deterministic. Concept match, difficulty band from measured mastery,
   duration against remaining weekly time, and anything already consumed.
2. **Rank** — deterministic. Weighted scoring over relevance, difficulty fit, format
   preference, publisher quality, and freshness, with a full trace.
3. **Explain** — the model's only job, and it may only describe rows that stage two
   already selected.

The model never chooses *what* to recommend and never writes a URL. That is enforced
structurally: `known_resources` validates generated output against the catalogue, so
an invented link fails validation rather than reaching a learner.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

from pathwise.models.enums import LearningStyle, ResourceType
from pathwise.services.resource.catalogue import ResourceSpec

#: Ranking weights. Relevance dominates: a perfectly-pitched resource about the wrong
#: topic is worthless, while a slightly mismatched one about the right topic is not.
WEIGHTS: Final[dict[str, float]] = {
    "difficulty_fit": 0.30,
    "style_fit": 0.22,
    "focus": 0.18,
    "quality": 0.16,
    "duration_fit": 0.09,
    "freshness": 0.05,
}

#: Concepts a resource can cover before it stops being *about* any one of them. A
#: 20-hour course touching eight topics is a worse answer to "help me with gradient
#: descent" than a 25-minute video written about exactly that, even though both are
#: relevant and the course is more thorough.
FOCUS_SATURATION_CONCEPTS: Final = 6

#: How much each format suits each stated learning style. A learner who says they
#: learn from video should not be handed an API reference first.
STYLE_AFFINITY: Final[dict[LearningStyle, dict[ResourceType, float]]] = {
    LearningStyle.VIDEO: {
        ResourceType.VIDEO: 1.0,
        ResourceType.COURSE: 0.85,
        ResourceType.INTERACTIVE: 0.6,
        ResourceType.TUTORIAL: 0.5,
        ResourceType.ARTICLE: 0.35,
        ResourceType.DOCUMENTATION: 0.25,
        ResourceType.BOOK: 0.2,
        ResourceType.PAPER: 0.15,
        ResourceType.EXERCISE: 0.4,
    },
    LearningStyle.READING: {
        ResourceType.BOOK: 1.0,
        ResourceType.DOCUMENTATION: 0.9,
        ResourceType.ARTICLE: 0.85,
        ResourceType.TUTORIAL: 0.7,
        ResourceType.PAPER: 0.6,
        ResourceType.COURSE: 0.45,
        ResourceType.INTERACTIVE: 0.4,
        ResourceType.VIDEO: 0.25,
        ResourceType.EXERCISE: 0.5,
    },
    LearningStyle.INTERACTIVE: {
        ResourceType.INTERACTIVE: 1.0,
        ResourceType.EXERCISE: 0.95,
        ResourceType.TUTORIAL: 0.8,
        ResourceType.COURSE: 0.6,
        ResourceType.VIDEO: 0.45,
        ResourceType.DOCUMENTATION: 0.4,
        ResourceType.ARTICLE: 0.35,
        ResourceType.BOOK: 0.3,
        ResourceType.PAPER: 0.2,
    },
    LearningStyle.PROJECT_BASED: {
        ResourceType.EXERCISE: 1.0,
        ResourceType.INTERACTIVE: 0.85,
        ResourceType.TUTORIAL: 0.85,
        ResourceType.COURSE: 0.6,
        ResourceType.DOCUMENTATION: 0.55,
        ResourceType.VIDEO: 0.4,
        ResourceType.ARTICLE: 0.35,
        ResourceType.BOOK: 0.3,
        ResourceType.PAPER: 0.2,
    },
}

#: With no stated preference, format barely matters and the other factors decide.
NEUTRAL_STYLE_AFFINITY: Final = 0.7

#: A resource older than this starts losing freshness score. Generous, because a
#: linear algebra lecture from 2010 has not aged and a JavaScript tutorial from 2019
#: has — the catalogue is mostly the former.
FRESHNESS_HALF_LIFE_YEARS: Final = 8.0

#: Anything longer than the learner's remaining week is scored down rather than
#: excluded: a good long resource is still worth surfacing, just not first.
DURATION_TOLERANCE: Final = 1.5


@dataclass(frozen=True, slots=True)
class RecommendationContext:
    """What the learner brings to the request."""

    #: Measured mastery of the concept, on [0, 1]. Drives the difficulty band.
    concept_mastery: float = 0.0
    learning_style: LearningStyle = LearningStyle.MIXED
    #: Minutes of study left this week, for duration fit.
    minutes_available: int | None = None
    #: Canonical URLs already seen. Recommending the same video twice is the fastest
    #: way to look like nothing is being tracked.
    already_seen: frozenset[str] = frozenset()
    #: Objectives the learner actually missed. When present, resources that cover
    #: them are preferred — the difference between "study gradient descent again" and
    #: "here is the part you got wrong".
    weak_objectives: frozenset[str] = frozenset()
    free_only: bool = False
    today: date | None = None


@dataclass(frozen=True, slots=True)
class ScoredResource:
    """One ranked resource with its full reasoning."""

    resource: ResourceSpec
    factors: Mapping[str, float]
    #: Human-readable justification per factor, handed to the explanation prompt.
    details: Mapping[str, str]

    @property
    def score(self) -> float:
        return sum(value * WEIGHTS[name] for name, value in self.factors.items())

    @property
    def dominant_factor(self) -> str:
        return max(self.factors, key=lambda name: self.factors[name] * WEIGHTS[name])

    @property
    def url(self) -> str:
        return self.resource.url


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    """A ranked shortlist, with what was filtered out and why."""

    ranked: tuple[ScoredResource, ...]
    excluded: tuple[tuple[str, str], ...] = ()

    @property
    def best(self) -> ScoredResource | None:
        return self.ranked[0] if self.ranked else None

    @property
    def is_empty(self) -> bool:
        return not self.ranked

    def urls(self) -> frozenset[str]:
        """The set an explanation is allowed to cite."""
        return frozenset(item.resource.canonical for item in self.ranked)

    def to_prompt_json(self) -> list[dict[str, object]]:
        """The shortlist as the explanation prompt receives it.

        Every field the model may mention appears here, and nothing else does — it
        cannot describe a resource it was not given.
        """
        return [
            {
                "url": item.resource.canonical,
                "title": item.resource.title,
                "type": str(item.resource.resource_type),
                "difficulty": item.resource.difficulty,
                "duration_minutes": item.resource.duration_minutes,
                "publisher": item.resource.publisher,
                "description": item.resource.description,
                "covers_objectives": item.resource.covers_objectives,
                "why_ranked": item.details[item.dominant_factor],
            }
            for item in self.ranked
        ]


def recommend(
    resources: Iterable[ResourceSpec],
    concept_slug: str,
    context: RecommendationContext | None = None,
    *,
    limit: int = 3,
) -> RecommendationResult:
    """Rank resources for one concept.

    Deterministic throughout: the same learner state and catalogue always produce the
    same shortlist in the same order.
    """
    context = context or RecommendationContext()
    today = context.today or date.today()

    candidates: list[ResourceSpec] = []
    excluded: list[tuple[str, str]] = []

    for resource in resources:
        if concept_slug not in resource.concepts:
            # Not an exclusion worth recording — it was never a candidate. Logging
            # every unrelated resource would bury the ones a person cares about.
            continue

        reason = _exclusion_reason(resource, context)
        if reason is None:
            candidates.append(resource)
        else:
            excluded.append((resource.title, reason))

    scored = [_score(resource, context, today) for resource in candidates]
    # Ties break on canonical URL so a rerun never silently reorders equals.
    scored.sort(key=lambda item: (-item.score, item.resource.canonical))

    return RecommendationResult(
        ranked=tuple(scored[:limit]),
        excluded=tuple(sorted(excluded)),
    )


def _exclusion_reason(resource: ResourceSpec, context: RecommendationContext) -> str | None:
    """Why an otherwise-relevant resource was filtered out, or ``None``.

    Only called for resources that do cover the concept, so every reason recorded
    here is one a learner might reasonably ask about.
    """
    if resource.canonical in context.already_seen:
        return "already seen"
    if context.free_only and not resource.is_free:
        return "not free"
    return None


def _score(resource: ResourceSpec, context: RecommendationContext, today: date) -> ScoredResource:
    """Score one resource, recording every term."""
    factors: dict[str, float] = {}
    details: dict[str, str] = {}

    # --- difficulty fit ------------------------------------------------------- #
    # Map mastery onto the 1-5 scale the catalogue uses, then prefer material just
    # above the learner: at or slightly beyond their level, never far below it.
    learner_level = 1.0 + context.concept_mastery * 4.0
    gap = resource.difficulty - learner_level
    if gap < 0:
        # Too easy wastes time, but is survivable — penalise gently.
        fit = max(0.0, 1.0 + gap / 3.0)
        verdict = "below your current level"
    else:
        fit = max(0.0, 1.0 - gap / 2.5)
        verdict = "pitched just above your current level" if gap <= 1.5 else "a stretch"
    factors["difficulty_fit"] = fit
    details["difficulty_fit"] = (
        f"difficulty {resource.difficulty}/5, {verdict} ({learner_level:.1f}/5 measured)"
    )

    # --- style fit ------------------------------------------------------------- #
    affinity = STYLE_AFFINITY.get(context.learning_style, {}).get(
        resource.resource_type, NEUTRAL_STYLE_AFFINITY
    )
    factors["style_fit"] = affinity
    details["style_fit"] = (
        f"a {resource.resource_type} resource, which suits how you said you learn"
        if affinity >= 0.8
        else f"a {resource.resource_type} resource"
    )

    # --- focus ------------------------------------------------------------------- #
    # How much of this resource is actually about the concept asked for. Without it,
    # a broad survey ties with material written specifically for the topic, and the
    # tie breaks arbitrarily on URL.
    covered = len(resource.concepts)
    factors["focus"] = max(0.15, 1.0 - (covered - 1) / FOCUS_SATURATION_CONCEPTS)
    details["focus"] = (
        "written specifically about this topic"
        if covered == 1
        else f"covers this among {covered} topics"
    )

    # --- publisher quality ------------------------------------------------------ #
    factors["quality"] = resource.quality_prior
    details["quality"] = f"published by {resource.publisher}"

    # --- duration fit ----------------------------------------------------------- #
    if context.minutes_available is None or resource.duration_minutes is None:
        factors["duration_fit"] = 0.6
        details["duration_fit"] = "no time constraint given"
    else:
        budget = max(1, context.minutes_available)
        ratio = resource.duration_minutes / budget
        factors["duration_fit"] = (
            1.0 if ratio <= 1.0 else max(0.0, 1.0 - (ratio - 1.0) / DURATION_TOLERANCE)
        )
        hours = resource.duration_minutes / 60
        details["duration_fit"] = (
            f"about {hours:.1f} hours, against the {budget / 60:.1f} you have left"
        )

    # --- freshness -------------------------------------------------------------- #
    if resource.published_at is None:
        # Most of the catalogue is undated documentation that tracks its project.
        factors["freshness"] = 0.75
        details["freshness"] = "continuously maintained"
    else:
        years = (today - resource.published_at).days / 365.25
        factors["freshness"] = max(0.2, 1.0 - years / (FRESHNESS_HALF_LIFE_YEARS * 2))
        details["freshness"] = f"published {resource.published_at.year}"

    # --- objective targeting ---------------------------------------------------- #
    # Not a weighted factor: a resource that covers the exact objective the learner
    # missed is boosted directly, because "here is the part you got wrong" beats any
    # amount of general fit.
    if context.weak_objectives and resource.covers_objectives:
        overlap = context.weak_objectives & set(resource.covers_objectives)
        if overlap:
            factors["difficulty_fit"] = min(1.0, factors["difficulty_fit"] + 0.3)
            details["difficulty_fit"] = (
                f"covers {', '.join(sorted(overlap))}, which is where you struggled"
            )

    return ScoredResource(resource=resource, factors=factors, details=details)


def fallback_explanation(item: ScoredResource) -> str:
    """A deterministic justification, for when generation fails.

    Read straight off the ranking, so a learner is never shown a recommendation with
    no reason attached.
    """
    return f"{item.resource.title} — {item.details[item.dominant_factor]}."


def summarise(result: RecommendationResult) -> dict[str, object]:
    """The payload behind a resource panel."""
    if result.is_empty:
        return {"resources": [], "reason": "no suitable resources found"}
    return {
        "resources": [
            {
                "title": item.resource.title,
                "url": item.resource.url,
                "type": str(item.resource.resource_type),
                "duration_minutes": item.resource.duration_minutes,
                "why": fallback_explanation(item),
                "score": round(item.score, 3),
            }
            for item in result.ranked
        ]
    }


def citable_urls(results: Sequence[RecommendationResult]) -> frozenset[str]:
    """Every URL an explanation across these results may reference."""
    return frozenset(url for result in results for url in result.urls())
