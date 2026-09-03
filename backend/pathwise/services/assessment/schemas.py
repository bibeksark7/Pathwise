"""Structured output schemas for generated questions.

A generated assessment question is unusual among LLM outputs: it is shown to a learner
as authoritative, and its grading directly moves the mastery estimate that decides
what they study next. A bad question does not merely look bad — it produces false
evidence, and false evidence is worse than none, because the system will act on it
confidently.

So the schema is tight, and the domain validators in `generator.py` are tighter. What
the schema alone cannot express — that exactly one option is correct, that the answer
is not given away in the stem, that the concept is one we asked about — is checked
there.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pathwise.models.enums import QuestionType

#: Four options is the sweet spot: enough that guessing is 25%, few enough that a
#: model can write three genuinely plausible distractors rather than two plus filler.
MIN_OPTIONS = 3
MAX_OPTIONS = 5


class GeneratedOption(BaseModel):
    """One multiple-choice option."""

    model_config = ConfigDict(extra="forbid")

    #: Single lowercase letter, so the answer key and the learner's response have an
    #: unambiguous shared vocabulary.
    id: str = Field(pattern=r"^[a-e]$")
    text: str = Field(min_length=1, max_length=300)
    #: Exactly one option per question must set this. Enforced in `generator.py`.
    is_correct: bool = False
    #: Why this option is wrong, shown after the learner answers. A distractor that
    #: cannot be explained is usually a distractor that is arguably also correct.
    why_wrong: str | None = Field(default=None, max_length=300)


class GeneratedQuestion(BaseModel):
    """One question, bound to the concept and objective it measures."""

    model_config = ConfigDict(extra="forbid")

    #: The concept slug this question probes. Validated against the blueprint — a
    #: question about something we did not ask for measures nothing we can use.
    concept_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    #: Objective ids this question measures. Validated against the concept's declared
    #: objectives, so a wrong answer can name the capability that is missing.
    objective_ids: list[str] = Field(min_length=1, max_length=3)

    question_type: QuestionType = QuestionType.MULTIPLE_CHOICE
    stem: str = Field(min_length=20, max_length=1200)
    options: list[GeneratedOption] = Field(min_length=MIN_OPTIONS, max_length=MAX_OPTIONS)

    #: Shown after answering. This is the part a learner actually reads, so it must
    #: explain the reasoning rather than restate the correct option.
    explanation: str = Field(min_length=20, max_length=800)
    difficulty: int = Field(ge=1, le=5)

    @property
    def correct_options(self) -> list[GeneratedOption]:
        return [option for option in self.options if option.is_correct]

    @property
    def correct_option_id(self) -> str | None:
        correct = self.correct_options
        return correct[0].id if len(correct) == 1 else None


class GeneratedDiagnostic(BaseModel):
    """A whole diagnostic, as returned by the model."""

    model_config = ConfigDict(extra="forbid")

    questions: list[GeneratedQuestion] = Field(min_length=1, max_length=25)

    @property
    def probed_slugs(self) -> set[str]:
        return {question.concept_slug for question in self.questions}
