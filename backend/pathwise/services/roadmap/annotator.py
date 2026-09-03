"""Adding language to a decided plan.

The planner has already fixed what is in the roadmap and in what order. This module
asks a model for a title and one rationale per step — and nothing else. It cannot add
a step, remove one, or reorder anything, because the only thing it returns is prose
keyed by slugs that must match the plan exactly.

When the model fails, the roadmap is still delivered. Rationales are the part a
learner can most afford to lose, so an annotation failure degrades to a deterministic
template rather than failing the whole request. A learner with a working roadmap and
plain rationales is far better served than one with an error page.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass

from pathwise.ai.client import AIClient
from pathwise.ai.providers.base import Effort
from pathwise.ai.validators import ValidationResult
from pathwise.api.errors import AIError
from pathwise.logging_config import get_logger
from pathwise.services.knowledge.catalogue import render_plan_steps, render_skipped
from pathwise.services.knowledge.graph import KnowledgeGraph
from pathwise.services.roadmap.planner import RoadmapPlan
from pathwise.services.roadmap.schemas import RoadmapAnnotation

log = get_logger(__name__)

FEATURE = "roadmap_annotate"
PROMPT = "roadmap_annotate"

#: Rationales are long-form and there is one per step, so a 40-step plan needs real
#: headroom. Too small a ceiling truncates mid-document and fails validation with a
#: confusing parse error rather than an obvious one.
ANNOTATION_MAX_TOKENS = 16_000


@dataclass(frozen=True, slots=True)
class AnnotatedRoadmap:
    """A plan plus its language layer."""

    plan: RoadmapPlan
    title: str
    summary: str
    rationales: dict[str, str]
    #: True when the model failed and deterministic text was substituted. Surfaced
    #: so the UI can decide whether to offer a regenerate action, and so the quality
    #: of generated rationales is measurable rather than assumed.
    is_fallback: bool = False

    def rationale_for(self, slug: str) -> str:
        return self.rationales.get(slug, "")


def annotation_validator(
    plan: RoadmapPlan,
) -> Callable[[RoadmapAnnotation], ValidationResult]:
    """Every step gets exactly one rationale, and no step outside the plan gets one.

    The membership check is what makes it structurally impossible for the model to
    smuggle in a step the planner did not choose: an extra slug is rejected, not
    quietly rendered.
    """
    expected = set(plan.slugs)

    def validate(annotation: RoadmapAnnotation) -> ValidationResult:
        result = ValidationResult()
        produced = annotation.annotated_slugs

        invented = sorted(produced - expected)
        if invented:
            result.add(
                "rationales",
                f"cover {len(invented)} step(s) that are not in this roadmap: {invented[:5]}",
                "Write one rationale per listed step, using the exact slugs given. "
                "Do not add steps.",
            )

        missing = sorted(expected - produced)
        if missing:
            result.add(
                "rationales",
                f"omit {len(missing)} step(s): {missing[:5]}",
                f"Every step needs a rationale. Missing: {', '.join(missing[:15])}.",
            )

        duplicates = len(annotation.rationales) - len(produced)
        if duplicates > 0:
            result.add(
                "rationales",
                f"contain {duplicates} duplicate slug(s)",
                "Return exactly one rationale per step.",
            )

        return result

    return validate


class RoadmapAnnotator:
    """Generates the title, summary, and per-step rationales for a plan."""

    def __init__(self, client: AIClient, graph: KnowledgeGraph) -> None:
        self._client = client
        self._graph = graph

    async def annotate(
        self,
        plan: RoadmapPlan,
        *,
        interpreted_goal: str,
        user_id: uuid.UUID | None = None,
    ) -> AnnotatedRoadmap:
        """Annotate a plan, falling back to deterministic text on failure."""
        if plan.is_empty:
            return self._fallback(plan, interpreted_goal, reason="empty_plan")

        try:
            annotation = await self._client.generate_structured(
                feature=FEATURE,
                prompt_name=PROMPT,
                schema=RoadmapAnnotation,
                variables=self._variables(plan, interpreted_goal),
                effort=Effort.HIGH,
                max_tokens=ANNOTATION_MAX_TOKENS,
                user_id=user_id,
                validate=annotation_validator(plan),
            )
        except AIError as exc:
            # Deliberately broad across the AI error family: a refusal, a provider
            # outage, and an unrepairable output all mean the same thing here —
            # deliver the roadmap without generated prose.
            log.warning(
                "roadmap_annotation_failed",
                error_type=type(exc).__name__,
                steps=len(plan.nodes),
            )
            return self._fallback(plan, interpreted_goal, reason=type(exc).__name__)

        return AnnotatedRoadmap(
            plan=plan,
            title=annotation.title,
            summary=annotation.summary,
            rationales={r.slug: r.rationale for r in annotation.rationales},
        )

    def _variables(self, plan: RoadmapPlan, interpreted_goal: str) -> dict[str, object]:
        return {
            "interpreted_goal": interpreted_goal,
            "plan_steps": render_plan_steps(self._graph, plan.concept_ids),
            "skipped_summary": render_skipped(
                (s.slug, s.mastery, s.evidence_count) for s in plan.skipped
            ),
            "step_count": len(plan.nodes),
            "total_hours": f"{plan.pacing.total_minutes / 60:.0f}",
            "hours_per_week": f"{plan.pacing.hours_per_week:g}",
            "estimated_weeks": f"{plan.pacing.estimated_weeks:g}",
        }

    def _fallback(
        self, plan: RoadmapPlan, interpreted_goal: str, *, reason: str
    ) -> AnnotatedRoadmap:
        """Deterministic text, built from graph facts alone.

        Plainer than generated prose, but every sentence is true — which is the
        right trade when the alternative is showing the learner nothing.
        """
        if plan.is_empty:
            title = "Nothing left to learn here"
            summary = (
                "You already meet every prerequisite for this goal, so there is "
                "nothing left to schedule."
            )
        else:
            destination = plan.nodes[-1].name
            title = f"Path to {destination}"
            summary = (
                f"{len(plan.nodes)} steps from {plan.nodes[0].name} to {destination}, "
                f"about {plan.pacing.total_minutes / 60:.0f} hours of study "
                f"(~{plan.pacing.estimated_weeks:g} weeks at "
                f"{plan.pacing.hours_per_week:g} hours per week)."
            )
            if plan.skipped:
                summary += (
                    f" {len(plan.skipped)} prerequisite(s) were skipped because you "
                    "have already demonstrated them."
                )

        return AnnotatedRoadmap(
            plan=plan,
            title=title,
            summary=summary,
            rationales={
                node.slug: self._deterministic_rationale(node.concept_id, plan)
                for node in plan.nodes
            },
            is_fallback=True,
        )

    def _deterministic_rationale(self, concept_id: uuid.UUID, plan: RoadmapPlan) -> str:
        """A rationale assembled from prerequisite structure, with no generation."""
        node = self._graph.node(concept_id)
        included = set(plan.concept_ids)

        unlocks = [
            self._graph.node(dependent).name
            for dependent in self._graph.direct_dependents(concept_id)
            if dependent in included
        ]
        requires = [
            self._graph.node(r.concept_id).name
            for r in self._graph.direct_requirements(concept_id)
            if r.concept_id in included
        ]

        parts = [f"{node.name} takes about {node.estimated_minutes / 60:.1f} hours."]
        if requires:
            parts.append(f"It builds on {_join(requires)}.")
        if unlocks:
            parts.append(f"It is needed for {_join(unlocks[:3])}.")
        return " ".join(parts)


def _join(names: list[str]) -> str:
    """Human list joining — 'a', 'a and b', 'a, b and c'."""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])} and {names[-1]}"
