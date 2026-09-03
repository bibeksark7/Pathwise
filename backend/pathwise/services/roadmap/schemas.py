"""Structured output schemas for the roadmap pipeline.

These are the *only* shapes a model may return. Note what they do and do not contain:

* ``ParsedGoal`` selects concept **slugs from the catalogue**. It never returns a
  free-text topic, because a free-text topic cannot be validated against the graph.
* ``RoadmapAnnotation`` carries a title and prose. It contains no ordering, no
  inclusions, and no time estimates — those are the planner's output, already fixed
  before this schema is ever populated.

The division is the point: the model is asked for language, not decisions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: Slugs are kebab-case identifiers from the catalogue. Constraining the pattern
#: catches a model returning a display name ("Gradient Descent") before the
#: membership validator has to.
SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class ParsedGoal(BaseModel):
    """A learner's free-text goal, resolved onto the knowledge graph."""

    model_config = ConfigDict(extra="forbid")

    #: The concepts that *are* the goal — what the learner wants to be able to do.
    #: Validated for membership in the catalogue before use.
    goal_slugs: list[str] = Field(min_length=1, max_length=6)

    #: The goal restated plainly, shown back to the learner for confirmation. This is
    #: how a misreading gets caught before an entire roadmap is built on it.
    interpreted_goal: str = Field(min_length=10, max_length=300)

    #: Concepts the learner claims to know already. Self-report is the weakest
    #: evidence there is, so these seed a *diagnostic*, they do not set mastery.
    claimed_knowledge: list[str] = Field(default_factory=list, max_length=25)

    #: Anything in the request the model could not map to the graph — a domain
    #: Pathwise does not cover yet. Surfaced rather than silently dropped.
    unmapped_topics: list[str] = Field(default_factory=list, max_length=10)

    #: Set when the request is too vague to plan from. The caller asks a follow-up
    #: question instead of guessing.
    needs_clarification: bool = False
    clarifying_question: str | None = Field(default=None, max_length=200)


class NodeRationale(BaseModel):
    """Why one step is in the roadmap, in the learner's terms."""

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(pattern=SLUG_PATTERN)
    #: One or two sentences. Long enough to say something, short enough to read on a
    #: node detail panel without scrolling.
    rationale: str = Field(min_length=20, max_length=400)


class RoadmapAnnotation(BaseModel):
    """The language layer over an already-decided plan."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=5, max_length=80)
    #: Two or three sentences describing the shape of the path — where it starts,
    #: what it builds towards, and what was skipped.
    summary: str = Field(min_length=40, max_length=800)
    rationales: list[NodeRationale] = Field(default_factory=list)

    def rationale_for(self, slug: str) -> str | None:
        return next((r.rationale for r in self.rationales if r.slug == slug), None)

    @property
    def annotated_slugs(self) -> set[str]:
        return {r.slug for r in self.rationales}
