"""Generating diagnostic questions.

The planner decides what to probe; this writes the questions. Everything structural
is already fixed before the model is called — which concepts, which objectives, what
difficulty — so the model's job is purely to author prose that measures a specified
capability.

The validators below check what the schema cannot. A question can be perfectly
well-formed JSON and still be unusable: two defensible answers, a giveaway in the
stem, a concept nobody asked about. Each of those produces *false evidence*, which is
worse than no evidence — the mastery model will act on it confidently, and the learner
will be routed accordingly.

When generation cannot produce something valid, the diagnostic is skipped rather than
degraded. There is no deterministic fallback here and there should not be: a
hand-rolled template question would measure nothing, and a placement test that
measures nothing is worse than starting with no assumptions at all.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence

from pathwise.ai.client import AIClient
from pathwise.ai.providers.base import Effort
from pathwise.ai.validators import ValidationResult
from pathwise.api.errors import AIError
from pathwise.logging_config import get_logger
from pathwise.models.enums import QuestionType
from pathwise.services.assessment.grading import QuestionSpec
from pathwise.services.assessment.schemas import GeneratedDiagnostic, GeneratedQuestion
from pathwise.services.assessment.selection import DiagnosticBlueprint, ProbeTarget
from pathwise.services.knowledge.graph import KnowledgeGraph

log = get_logger(__name__)

FEATURE = "diagnostic_generate"
PROMPT = "diagnostic_generate"

#: Questions are long — a stem, four options with explanations, and a rationale each.
#: Ten of those needs real headroom or generation truncates mid-document.
GENERATION_MAX_TOKENS = 16_000

#: Phrases that hand the answer to the learner. Not exhaustive — a determined model
#: can still leak — but these are the recurring ones.
_GIVEAWAY_PATTERNS = (
    re.compile(r"\bthe (?:correct )?answer is\b", re.IGNORECASE),
    re.compile(r"\bhint\s*:", re.IGNORECASE),
    re.compile(r"\bobviously\b", re.IGNORECASE),
    re.compile(r"\brecall that\b", re.IGNORECASE),
)

#: Options that test test-taking rather than the concept.
_BANNED_OPTIONS = ("all of the above", "none of the above", "both a and b")


def diagnostic_validator(
    blueprint: DiagnosticBlueprint, graph: KnowledgeGraph
) -> Callable[[GeneratedDiagnostic], ValidationResult]:
    """Every rule a generated diagnostic must satisfy to be usable."""
    targets = {target.slug: target for target in blueprint.targets}

    def validate(diagnostic: GeneratedDiagnostic) -> ValidationResult:
        result = ValidationResult()
        _check_coverage(result, diagnostic, targets)
        for index, question in enumerate(diagnostic.questions, start=1):
            _check_question(result, index, question, targets)
        return result

    return validate


def _check_coverage(
    result: ValidationResult,
    diagnostic: GeneratedDiagnostic,
    targets: dict[str, ProbeTarget],
) -> None:
    """One question per requested concept, and nothing extra."""
    produced = diagnostic.probed_slugs
    expected = set(targets)

    unexpected = sorted(produced - expected)
    if unexpected:
        result.add(
            "questions",
            f"probe {len(unexpected)} concept(s) that were not requested: {unexpected[:5]}",
            "Write one question per listed concept only. Requested: "
            + ", ".join(sorted(expected))
            + ".",
        )

    missing = sorted(expected - produced)
    if missing:
        result.add(
            "questions",
            f"omit {len(missing)} requested concept(s): {missing[:5]}",
            f"Every listed concept needs a question. Missing: {', '.join(missing[:10])}.",
        )

    duplicates = len(diagnostic.questions) - len(produced)
    if duplicates > 0:
        result.add(
            "questions",
            f"contain {duplicates} duplicate concept(s)",
            "Write exactly one question per concept.",
        )


def _check_question(
    result: ValidationResult,
    index: int,
    question: GeneratedQuestion,
    targets: dict[str, ProbeTarget],
) -> None:
    """Everything the schema cannot express about a single question."""
    label = f"question {index} ({question.concept_slug})"
    target = targets.get(question.concept_slug)

    # Exactly one correct answer. Zero makes the question ungradeable; two produces a
    # wrong measurement from a learner who reasoned correctly.
    correct = question.correct_options
    if len(correct) != 1:
        result.add(
            label,
            f"has {len(correct)} correct options, expected exactly 1",
            "Mark exactly one option `is_correct`. If two are defensible, rewrite the "
            "question so only one is.",
        )

    # Objectives must be real, or a score cannot be attributed to a capability.
    if target is not None:
        allowed = set(target.objective_ids)
        unknown = sorted(set(question.objective_ids) - allowed)
        if unknown:
            result.add(
                label,
                f"cites objective(s) this concept does not declare: {unknown}",
                f"Use only these objective ids for {question.concept_slug}: "
                f"{', '.join(sorted(allowed))}.",
            )

    option_texts = [option.text.strip().lower() for option in question.options]

    if len(set(option_texts)) != len(option_texts):
        result.add(
            label,
            "has duplicate options",
            "Every option must be distinct.",
        )

    banned = [text for text in option_texts if text in _BANNED_OPTIONS]
    if banned:
        result.add(
            label,
            f"uses a banned option: {banned}",
            "Do not use 'all of the above', 'none of the above', or similar — they "
            "test test-taking rather than the concept.",
        )

    option_ids = [option.id for option in question.options]
    if len(set(option_ids)) != len(option_ids):
        result.add(label, "has duplicate option ids", "Give each option a distinct id.")

    for pattern in _GIVEAWAY_PATTERNS:
        if pattern.search(question.stem):
            result.add(
                label,
                f"gives the answer away in the stem ({pattern.pattern})",
                "The stem must pose the problem without resolving it.",
            )
            break

    # A correct option markedly longer than the distractors is the classic tell that
    # lets a test-wise learner score without understanding anything.
    if len(correct) == 1 and len(question.options) > 1:
        correct_length = len(correct[0].text)
        others = [len(o.text) for o in question.options if not o.is_correct]
        if others and correct_length > 2.5 * (sum(others) / len(others)):
            result.add(
                label,
                "has a correct option far longer than its distractors",
                "Make all options similar in length and specificity, or the answer is "
                "guessable from shape alone.",
            )

    if target is not None and abs(question.difficulty - target.difficulty) > 1:
        result.add(
            label,
            f"is pitched at difficulty {question.difficulty}, "
            f"but the concept is difficulty {target.difficulty}",
            f"Write this question at difficulty {target.difficulty}.",
        )


class DiagnosticGenerator:
    """Writes the questions for a blueprint."""

    def __init__(self, client: AIClient, graph: KnowledgeGraph) -> None:
        self._client = client
        self._graph = graph

    async def generate(
        self, blueprint: DiagnosticBlueprint, *, user_id: uuid.UUID | None = None
    ) -> tuple[GeneratedQuestion, ...]:
        """Generate one question per probe target.

        Raises:
            AIError: if usable questions could not be produced. Callers should start
                the learner with no assumptions rather than a broken placement test.
        """
        diagnostic = await self._client.generate_structured(
            feature=FEATURE,
            prompt_name=PROMPT,
            schema=GeneratedDiagnostic,
            variables={"targets": render_targets(self._graph, blueprint.targets)},
            effort=Effort.HIGH,
            max_tokens=GENERATION_MAX_TOKENS,
            user_id=user_id,
            validate=diagnostic_validator(blueprint, self._graph),
        )

        # Return in blueprint order so the learner sees an easy question first rather
        # than whatever order generation happened to produce.
        by_slug = {q.concept_slug: q for q in diagnostic.questions}
        return tuple(by_slug[target.slug] for target in blueprint.targets if target.slug in by_slug)

    async def generate_or_skip(
        self, blueprint: DiagnosticBlueprint, *, user_id: uuid.UUID | None = None
    ) -> tuple[GeneratedQuestion, ...]:
        """Generate, returning nothing rather than raising if it cannot be done.

        Deliberately not a fallback to template questions. A placement test that
        measures nothing would still move mastery estimates, and acting on false
        evidence is worse than starting with none.
        """
        try:
            return await self.generate(blueprint, user_id=user_id)
        except AIError as exc:
            log.warning(
                "diagnostic_generation_failed",
                error_type=type(exc).__name__,
                targets=len(blueprint.targets),
            )
            return ()


def render_targets(graph: KnowledgeGraph, targets: Sequence[ProbeTarget]) -> str:
    """Render probe targets for the generation prompt.

    Includes the concept's description and its declared objectives verbatim, so the
    model writes about the concept as the graph defines it rather than as it recalls
    the term.
    """
    lines: list[str] = []
    for index, target in enumerate(targets, start=1):
        node = graph.node(target.concept_id)
        lines.append(f"{index}. {target.slug} — {target.name} (difficulty {target.difficulty}/5)")
        if node.description:
            lines.append(f"   {' '.join(node.description.split())[:300]}")
        lines.append(f"   objectives to measure: {', '.join(target.objective_ids) or 'lo-1'}")
    return "\n".join(lines)


def to_question_specs(
    questions: Sequence[GeneratedQuestion], graph: KnowledgeGraph
) -> tuple[QuestionSpec, ...]:
    """Convert generated questions into the gradeable form.

    Assigns a stable id per question so responses can be matched back, and resolves
    slugs to concept ids for the evidence pipeline.
    """
    specs: list[QuestionSpec] = []
    for question in questions:
        try:
            concept_id = graph.by_slug(question.concept_slug).id
        except Exception:
            log.warning("generated_question_slug_missing", slug=question.concept_slug)
            continue

        specs.append(
            QuestionSpec(
                question_id=uuid.uuid4(),
                concept_ids=(concept_id,),
                objective_ids=tuple(question.objective_ids),
                correct_option=question.correct_option_id,
            )
        )
    return tuple(specs)


def is_multiple_choice(question: GeneratedQuestion) -> bool:
    return question.question_type is QuestionType.MULTIPLE_CHOICE
