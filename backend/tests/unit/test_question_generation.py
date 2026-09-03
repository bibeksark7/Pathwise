"""Question generation tests.

A generated question is shown to a learner as authoritative, and its grading moves the
mastery estimate that decides what they study next. So the bar is not "did the model
return valid JSON" — it is "would this question produce a true measurement".

Every test here corresponds to a way a well-formed question can still be unusable.
"""

from __future__ import annotations

import uuid
from uuid import UUID

import pytest

from pathwise.ai.call_log import InMemoryCallRecorder
from pathwise.ai.client import AIClient
from pathwise.ai.providers.fake_provider import FakeBehaviour, FakeProvider
from pathwise.api.errors import AIProviderError, AIValidationError
from pathwise.config import Settings
from pathwise.models.enums import RelationType
from pathwise.services.assessment.generator import (
    DiagnosticGenerator,
    diagnostic_validator,
    render_targets,
    to_question_specs,
)
from pathwise.services.assessment.schemas import (
    GeneratedDiagnostic,
    GeneratedOption,
    GeneratedQuestion,
)
from pathwise.services.assessment.selection import select_probes
from pathwise.services.knowledge.graph import GraphEdge, GraphNode, KnowledgeGraph


def cid(slug: str) -> UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, slug)


def node(slug: str, difficulty: int = 3) -> GraphNode:
    return GraphNode(
        id=cid(slug),
        slug=slug,
        name=slug.replace("-", " ").title(),
        difficulty=difficulty,
        estimated_minutes=120,
        domain="test",
        description="A concept description used when rendering the generation prompt.",
        objective_ids=("lo-1", "lo-2"),
    )


@pytest.fixture
def graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        [node("easy", 1), node("hard", 5)],
        [GraphEdge(cid("easy"), cid("hard"), RelationType.PREREQUISITE_OF, 1.0)],
    )


@pytest.fixture
def blueprint(graph: KnowledgeGraph):
    return select_probes(graph, graph.node_ids, question_count=2)


def question(
    slug: str = "easy",
    *,
    difficulty: int = 1,
    correct_count: int = 1,
    objective_ids: list[str] | None = None,
    stem: str = "Which statement best describes the behaviour being asked about here?",
    option_texts: list[str] | None = None,
) -> GeneratedQuestion:
    texts = option_texts or ["First option text", "Second option text", "Third option"]
    options = [
        GeneratedOption(
            id="abcde"[index],
            text=text,
            is_correct=index < correct_count,
            why_wrong=None if index < correct_count else "Because of a common misconception.",
        )
        for index, text in enumerate(texts)
    ]
    return GeneratedQuestion(
        concept_slug=slug,
        objective_ids=objective_ids or ["lo-1"],
        stem=stem,
        options=options,
        explanation="An explanation of the reasoning behind the correct answer here.",
        difficulty=difficulty,
    )


def diagnostic(*questions: GeneratedQuestion) -> GeneratedDiagnostic:
    return GeneratedDiagnostic(questions=list(questions))


def complete(blueprint) -> GeneratedDiagnostic:
    """One valid question per target."""
    return diagnostic(*(question(t.slug, difficulty=t.difficulty) for t in blueprint.targets))


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def test_a_complete_diagnostic_is_valid(blueprint, graph: KnowledgeGraph) -> None:
    assert diagnostic_validator(blueprint, graph)(complete(blueprint)).is_valid


def test_a_question_about_an_unrequested_concept_is_rejected(
    blueprint, graph: KnowledgeGraph
) -> None:
    """A question about something we did not ask for measures nothing usable, and
    would attach evidence to a concept the blueprint never selected."""
    bad = diagnostic(*complete(blueprint).questions, question("unrequested-concept"))
    result = diagnostic_validator(blueprint, graph)(bad)
    assert not result.is_valid
    assert "not requested" in result.issues[0].problem


def test_a_missing_concept_is_rejected(blueprint, graph: KnowledgeGraph) -> None:
    partial = diagnostic(complete(blueprint).questions[0])
    result = diagnostic_validator(blueprint, graph)(partial)
    assert not result.is_valid
    assert any("omit" in issue.problem for issue in result.issues)


def test_duplicate_concepts_are_rejected(blueprint, graph: KnowledgeGraph) -> None:
    first = blueprint.targets[0]
    duplicated = diagnostic(
        *complete(blueprint).questions, question(first.slug, difficulty=first.difficulty)
    )
    result = diagnostic_validator(blueprint, graph)(duplicated)
    assert not result.is_valid
    assert any("duplicate" in issue.problem for issue in result.issues)


# --------------------------------------------------------------------------- #
# Answerability
# --------------------------------------------------------------------------- #


def test_two_correct_answers_are_rejected(blueprint, graph: KnowledgeGraph) -> None:
    """A learner who reasons correctly and picks the other defensible option gets
    marked wrong, and the mastery model records a failure that did not happen."""
    target = blueprint.targets[0]
    bad = diagnostic(
        question(target.slug, difficulty=target.difficulty, correct_count=2),
        *complete(blueprint).questions[1:],
    )
    result = diagnostic_validator(blueprint, graph)(bad)
    assert not result.is_valid
    assert any("2 correct options" in issue.problem for issue in result.issues)


def test_no_correct_answer_is_rejected(blueprint, graph: KnowledgeGraph) -> None:
    target = blueprint.targets[0]
    bad = diagnostic(
        question(target.slug, difficulty=target.difficulty, correct_count=0),
        *complete(blueprint).questions[1:],
    )
    result = diagnostic_validator(blueprint, graph)(bad)
    assert not result.is_valid


def test_duplicate_options_are_rejected(blueprint, graph: KnowledgeGraph) -> None:
    target = blueprint.targets[0]
    bad = diagnostic(
        question(
            target.slug,
            difficulty=target.difficulty,
            option_texts=["Same text", "Same text", "Different"],
        ),
        *complete(blueprint).questions[1:],
    )
    result = diagnostic_validator(blueprint, graph)(bad)
    assert not result.is_valid
    assert any("duplicate options" in issue.problem for issue in result.issues)


@pytest.mark.parametrize("banned", ["All of the above", "None of the above"])
def test_banned_options_are_rejected(blueprint, graph: KnowledgeGraph, banned: str) -> None:
    """These test test-taking skill rather than the concept."""
    target = blueprint.targets[0]
    bad = diagnostic(
        question(
            target.slug,
            difficulty=target.difficulty,
            option_texts=["A real option", "Another real option", banned],
        ),
        *complete(blueprint).questions[1:],
    )
    result = diagnostic_validator(blueprint, graph)(bad)
    assert not result.is_valid
    assert any("banned option" in issue.problem for issue in result.issues)


@pytest.mark.parametrize(
    "stem",
    [
        "The correct answer is the first one; which option is it?",
        "Hint: think about gradients. Which option follows?",
        "Recall that the gradient points uphill. Which option is right?",
    ],
)
def test_a_giveaway_in_the_stem_is_rejected(blueprint, graph: KnowledgeGraph, stem: str) -> None:
    target = blueprint.targets[0]
    bad = diagnostic(
        question(target.slug, difficulty=target.difficulty, stem=stem),
        *complete(blueprint).questions[1:],
    )
    result = diagnostic_validator(blueprint, graph)(bad)
    assert not result.is_valid
    assert any("gives the answer away" in issue.problem for issue in result.issues)


def test_a_conspicuously_long_correct_option_is_rejected(blueprint, graph: KnowledgeGraph) -> None:
    """The classic tell that lets a test-wise learner score without understanding."""
    target = blueprint.targets[0]
    bad = diagnostic(
        question(
            target.slug,
            difficulty=target.difficulty,
            option_texts=[
                "A very long and carefully qualified statement that hedges every claim "
                "and therefore reads as the thorough, considered, obviously-correct one",
                "Short",
                "Also short",
            ],
        ),
        *complete(blueprint).questions[1:],
    )
    result = diagnostic_validator(blueprint, graph)(bad)
    assert not result.is_valid
    assert any("far longer" in issue.problem for issue in result.issues)


# --------------------------------------------------------------------------- #
# Objective binding
# --------------------------------------------------------------------------- #


def test_an_invented_objective_is_rejected(blueprint, graph: KnowledgeGraph) -> None:
    """Without a real objective id, a wrong answer cannot name the missing
    capability — the score reverts to a number about a topic."""
    target = blueprint.targets[0]
    bad = diagnostic(
        question(target.slug, difficulty=target.difficulty, objective_ids=["lo-99"]),
        *complete(blueprint).questions[1:],
    )
    result = diagnostic_validator(blueprint, graph)(bad)
    assert not result.is_valid
    assert any("does not declare" in issue.problem for issue in result.issues)


def test_a_declared_objective_is_accepted(blueprint, graph: KnowledgeGraph) -> None:
    target = blueprint.targets[0]
    ok = diagnostic(
        question(target.slug, difficulty=target.difficulty, objective_ids=["lo-2"]),
        *complete(blueprint).questions[1:],
    )
    assert diagnostic_validator(blueprint, graph)(ok).is_valid


# --------------------------------------------------------------------------- #
# Difficulty
# --------------------------------------------------------------------------- #


def test_a_wildly_mispitched_question_is_rejected(blueprint, graph: KnowledgeGraph) -> None:
    """A difficulty-5 question about a difficulty-1 concept measures frustration."""
    target = next(t for t in blueprint.targets if t.difficulty <= 2)
    bad = diagnostic(
        question(target.slug, difficulty=5),
        *(q for q in complete(blueprint).questions if q.concept_slug != target.slug),
    )
    result = diagnostic_validator(blueprint, graph)(bad)
    assert not result.is_valid
    assert any("pitched at difficulty" in issue.problem for issue in result.issues)


def test_one_level_of_drift_is_tolerated(blueprint, graph: KnowledgeGraph) -> None:
    """Difficulty is a judgement call, not a measurement — an exact match would
    reject reasonable questions and burn repair attempts."""
    target = blueprint.targets[0]
    ok = diagnostic(
        question(target.slug, difficulty=min(5, target.difficulty + 1)),
        *complete(blueprint).questions[1:],
    )
    assert diagnostic_validator(blueprint, graph)(ok).is_valid


# --------------------------------------------------------------------------- #
# Prompt rendering and conversion
# --------------------------------------------------------------------------- #


def test_the_prompt_carries_the_concept_definition(blueprint, graph: KnowledgeGraph) -> None:
    """So the model writes about the concept as the graph defines it, rather than as
    it happens to recall the term."""
    rendered = render_targets(graph, blueprint.targets)
    assert "A concept description used when rendering" in rendered
    assert "objectives to measure: lo-1, lo-2" in rendered


def test_generated_questions_convert_to_gradeable_specs(graph: KnowledgeGraph) -> None:
    specs = to_question_specs([question("easy")], graph)
    assert len(specs) == 1
    assert specs[0].concept_ids == (cid("easy"),)
    assert specs[0].correct_option == "a"
    assert specs[0].objective_ids == ("lo-1",)


def test_a_question_for_a_vanished_concept_is_dropped(graph: KnowledgeGraph) -> None:
    assert to_question_specs([question("no-such-concept")], graph) == ()


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_provider="fake", jwt_secret="x" * 40)


async def test_generation_returns_questions_in_blueprint_order(
    blueprint, graph: KnowledgeGraph, settings: Settings
) -> None:
    """Easiest first, so a learner is not met with a difficulty-5 question."""
    recorder = InMemoryCallRecorder()
    provider = FakeProvider(FakeBehaviour(canned_values=[complete(blueprint)]))
    client = AIClient(provider, settings, recorder=recorder)

    questions = await DiagnosticGenerator(client, graph).generate(blueprint)

    assert [q.concept_slug for q in questions] == [t.slug for t in blueprint.targets]
    assert recorder.records[0].feature == "diagnostic_generate"


async def test_an_unrepairable_diagnostic_raises(
    blueprint, graph: KnowledgeGraph, settings: Settings
) -> None:
    broken = diagnostic(question("unrequested-concept"))
    provider = FakeProvider(FakeBehaviour(canned_values=[broken, broken]))
    client = AIClient(provider, settings, recorder=InMemoryCallRecorder())

    with pytest.raises(AIValidationError):
        await DiagnosticGenerator(client, graph).generate(blueprint)


async def test_generation_failure_skips_rather_than_substituting(
    blueprint, graph: KnowledgeGraph, settings: Settings
) -> None:
    """Deliberately no template fallback. A placement test that measures nothing would
    still move mastery estimates, and acting on false evidence is worse than acting on
    none."""
    provider = FakeProvider(FakeBehaviour(fail_with=AIProviderError("outage")))
    client = AIClient(provider, settings, recorder=InMemoryCallRecorder())

    assert await DiagnosticGenerator(client, graph).generate_or_skip(blueprint) == ()
