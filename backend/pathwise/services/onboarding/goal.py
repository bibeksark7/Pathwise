"""Resolving a free-text learning goal onto the knowledge graph.

The one place natural language enters the planning pipeline. A learner writes "I want
to become an ML engineer"; this turns that into concept ids the planner can traverse
from.

The model's freedom is bounded on both sides: it is given the catalogue and told to
select from it, and its selection is then checked against the graph. Anything it
invents fails validation and gets one repair attempt with the valid options named.
Whatever survives is guaranteed to be a real concept — so nothing downstream has to
defend against a hallucinated goal.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from pathwise.ai.client import AIClient
from pathwise.ai.providers.base import Effort
from pathwise.ai.validators import ValidationPipeline, known_concepts, non_empty
from pathwise.api.errors import AIValidationError, ValidationError
from pathwise.logging_config import get_logger
from pathwise.services.knowledge.catalogue import render_catalogue
from pathwise.services.knowledge.graph import KnowledgeGraph
from pathwise.services.roadmap.schemas import ParsedGoal

log = get_logger(__name__)

FEATURE = "goal_parse"
PROMPT = "goal_parse"


def _referenced_slugs(parsed: ParsedGoal) -> list[str]:
    """Every slug the model claimed, goal and prior knowledge alike."""
    return [*parsed.goal_slugs, *parsed.claimed_knowledge]


@dataclass(frozen=True, slots=True)
class ResolvedGoal:
    """A goal, resolved to graph concepts and ready to plan from."""

    goal_concept_ids: tuple[uuid.UUID, ...]
    goal_slugs: tuple[str, ...]
    interpreted_goal: str
    claimed_knowledge_ids: tuple[uuid.UUID, ...] = ()
    claimed_knowledge_slugs: tuple[str, ...] = ()
    unmapped_topics: tuple[str, ...] = ()
    needs_clarification: bool = False
    clarifying_question: str | None = None

    @property
    def is_actionable(self) -> bool:
        """Whether a roadmap can be planned from this."""
        return bool(self.goal_concept_ids) and not self.needs_clarification


class GoalParser:
    """Turns a stated goal into concepts."""

    def __init__(self, client: AIClient, graph: KnowledgeGraph) -> None:
        self._client = client
        self._graph = graph

    async def parse(
        self,
        goal_text: str,
        *,
        experience_text: str = "",
        user_id: uuid.UUID | None = None,
    ) -> ResolvedGoal:
        """Resolve a free-text goal.

        Raises:
            ValidationError: if the goal text is empty.
            AIValidationError: if the model could not produce a goal grounded in the
                catalogue, even after repair. Callers should ask the learner to
                rephrase rather than plan from a guess.
        """
        goal_text = goal_text.strip()
        if not goal_text:
            raise ValidationError("Tell us what you want to learn.")

        known_slugs = {self._graph.node(cid).slug for cid in self._graph.node_ids}

        # Both lists of slugs are checked, not just the goal: a fabricated
        # "claimed knowledge" entry would silently seed a diagnostic for a concept
        # that does not exist.
        validate: ValidationPipeline[ParsedGoal] = ValidationPipeline(
            known_concepts(known_slugs, _referenced_slugs),
            non_empty("interpreted_goal"),
        )

        parsed = await self._client.generate_structured(
            feature=FEATURE,
            prompt_name=PROMPT,
            schema=ParsedGoal,
            variables={
                "goal_text": goal_text,
                "experience_text": experience_text.strip() or "(they did not say)",
            },
            # The catalogue is identical for every learner, so it goes in the cached
            # system prefix rather than the per-request message — billed at the
            # cached rate after the first call instead of in full every time.
            system=self._system_prompt(),
            effort=Effort.MEDIUM,
            user_id=user_id,
            validate=validate,
        )

        return self._to_resolved(parsed)

    def _system_prompt(self) -> str:
        """The stable, cacheable half of the request.

        The catalogue is the same on every call, so putting it here — before the
        cache breakpoint — means it is billed at the cached rate after the first
        request of a session rather than in full every time.
        """
        return (
            "You map learning goals onto a fixed catalogue of concepts. "
            "You may only ever return slugs that appear in this catalogue.\n\n"
            "# Concept catalogue\n\n" + render_catalogue(self._graph)
        )

    def _to_resolved(self, parsed: ParsedGoal) -> ResolvedGoal:
        """Convert validated slugs to concept ids.

        Validation has already guaranteed every slug exists, so this cannot fail —
        but it filters defensively anyway, because a silent `KeyError` here would
        surface much later as an empty roadmap with no explanation.
        """
        goal_ids = self._resolve(parsed.goal_slugs)
        claimed_ids = self._resolve(parsed.claimed_knowledge)

        if parsed.unmapped_topics:
            log.info(
                "goal_contains_unmapped_topics",
                topics=parsed.unmapped_topics,
                interpreted=parsed.interpreted_goal,
            )

        return ResolvedGoal(
            goal_concept_ids=tuple(goal_ids.values()),
            goal_slugs=tuple(goal_ids),
            interpreted_goal=parsed.interpreted_goal,
            claimed_knowledge_ids=tuple(claimed_ids.values()),
            claimed_knowledge_slugs=tuple(claimed_ids),
            unmapped_topics=tuple(parsed.unmapped_topics),
            needs_clarification=parsed.needs_clarification,
            clarifying_question=parsed.clarifying_question,
        )

    def _resolve(self, slugs: list[str]) -> dict[str, uuid.UUID]:
        resolved: dict[str, uuid.UUID] = {}
        for slug in slugs:
            try:
                resolved[slug] = self._graph.by_slug(slug).id
            except Exception:
                log.warning("goal_slug_not_in_graph", slug=slug)
        return resolved


async def parse_goal_or_fallback(
    parser: GoalParser,
    goal_text: str,
    *,
    experience_text: str = "",
    user_id: uuid.UUID | None = None,
) -> ResolvedGoal:
    """Parse a goal, degrading to a clarifying question rather than failing.

    Every AI feature in Pathwise has a deterministic path for when the model cannot
    deliver. Here it is the honest one: if the goal cannot be grounded in the graph,
    ask the learner to name their target instead of planning from a guess.
    """
    try:
        return await parser.parse(goal_text, experience_text=experience_text, user_id=user_id)
    except AIValidationError:
        log.warning("goal_parse_fell_back", goal_text=goal_text[:120])
        return ResolvedGoal(
            goal_concept_ids=(),
            goal_slugs=(),
            interpreted_goal=goal_text[:300],
            needs_clarification=True,
            clarifying_question=(
                "Which specific skill are you aiming for? Naming a concrete "
                "capability — 'train and deploy a model', 'pass algorithm "
                "interviews' — lets us build a path to it."
            ),
        )
