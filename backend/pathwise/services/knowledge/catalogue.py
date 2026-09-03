"""Rendering the knowledge graph as prompt text.

Every prompt that asks a model to reason about concepts is given this catalogue and
told to select from it. That is what makes "the model may not invent a concept" an
enforceable rule rather than an aspiration — the valid answers are enumerated in the
prompt, and a validator rejects anything outside them.

Two properties matter:

**Deterministic ordering.** Concepts are sorted by domain then slug, never by set
iteration order. Prompt caching is a prefix match, so a catalogue that shuffled
between calls would invalidate the cache on every request and quietly multiply cost.

**Stable and volatile content are separated.** The catalogue is the same for every
learner, so it belongs in the cached system prefix; the learner's own goal and state
go in the message, after the breakpoint.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from pathwise.services.knowledge.graph import KnowledgeGraph

#: Description text is truncated in the catalogue. The full text is available when a
#: concept is actually being worked on; here it only has to be enough to choose by,
#: and 89 full descriptions would crowd out the learner's own context.
_MAX_DESCRIPTION_CHARS = 140


def render_catalogue(
    graph: KnowledgeGraph,
    *,
    concept_ids: Iterable[UUID] | None = None,
    include_prerequisites: bool = False,
) -> str:
    """Render concepts as a stable, selectable list.

    Args:
        graph: The graph to render from.
        concept_ids: Restrict to these concepts. Defaults to everything.
        include_prerequisites: Append each concept's direct prerequisites. Useful
            when the model needs to reason about ordering, wasteful when it only
            needs to pick a target.
    """
    selected = set(concept_ids) if concept_ids is not None else set(graph.node_ids)
    nodes = sorted(
        (graph.node(cid) for cid in selected if cid in graph),
        key=lambda n: (n.domain, n.slug),
    )

    lines: list[str] = []
    current_domain: str | None = None

    for node in nodes:
        if node.domain != current_domain:
            current_domain = node.domain
            lines.append(f"\n## {current_domain}")

        hours = node.estimated_minutes / 60
        entry = f"- {node.slug} — {node.name} (difficulty {node.difficulty}/5, ~{hours:.1f}h)"

        if include_prerequisites:
            requirements = graph.direct_requirements(node.id)
            if requirements:
                names = ", ".join(graph.node(r.concept_id).slug for r in requirements)
                entry += f"\n    requires: {names}"

        lines.append(entry)

    return "\n".join(lines).strip()


def render_plan_steps(graph: KnowledgeGraph, concept_ids: Iterable[UUID]) -> str:
    """Render an ordered plan for the annotation prompt.

    Deliberately includes each step's position, its prerequisites, and what it
    unlocks — the facts a rationale must be built from. A model given only a list of
    names would have to invent the reasons, which is the failure this design exists
    to prevent.
    """
    ordered = [cid for cid in concept_ids if cid in graph]
    included = set(ordered)
    lines: list[str] = []

    for index, concept_id in enumerate(ordered, start=1):
        node = graph.node(concept_id)
        requirements = [
            graph.node(r.concept_id).slug
            for r in graph.direct_requirements(concept_id)
            if r.concept_id in included
        ]
        unlocks = [
            graph.node(dependent).slug
            for dependent in graph.direct_dependents(concept_id)
            if dependent in included
        ]

        lines.append(f"{index}. {node.slug} — {node.name}")
        lines.append(f"   {_summarise(node.description)}")
        lines.append(f"   difficulty {node.difficulty}/5, ~{node.estimated_minutes / 60:.1f}h")
        if requirements:
            lines.append(f"   builds on: {', '.join(requirements)}")
        if unlocks:
            lines.append(f"   unlocks: {', '.join(unlocks[:4])}")

    return "\n".join(lines)


def render_skipped(skipped: Iterable[tuple[str, float, int]]) -> str:
    """Render excused concepts with the evidence that excused them.

    Given to the annotation prompt so the summary can say what was skipped and why —
    grounded in real mastery figures rather than a guess at what the learner knows.
    """
    entries = list(skipped)
    if not entries:
        return "(nothing — this learner has no prior demonstrated knowledge here)"
    return "\n".join(
        f"- {slug}: mastery {mastery:.2f} across {count} recorded result(s)"
        for slug, mastery, count in entries
    )


def _summarise(description: str) -> str:
    """First sentence of a description, bounded."""
    text = " ".join(description.split())
    if not text:
        return "(no description)"

    sentence_end = text.find(". ")
    if 0 < sentence_end <= _MAX_DESCRIPTION_CHARS:
        return text[: sentence_end + 1]

    if len(text) <= _MAX_DESCRIPTION_CHARS:
        return text
    return text[:_MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0] + "..."
