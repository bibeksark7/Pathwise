"""Goal parsing and roadmap annotation tests.

These two services are where a language model touches the planning pipeline, so the
tests are about *containment*: the model may only name concepts that exist, it may
only annotate steps the planner chose, and when it fails the learner still gets a
working roadmap.

Everything runs against the deterministic fake provider.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from pathwise.ai.call_log import InMemoryCallRecorder
from pathwise.ai.client import AIClient
from pathwise.ai.providers.fake_provider import FakeBehaviour, FakeProvider
from pathwise.api.errors import AIProviderError, AIValidationError, ValidationError
from pathwise.config import Settings
from pathwise.models.enums import LLMCallStatus, RelationType
from pathwise.services.knowledge.catalogue import (
    render_catalogue,
    render_plan_steps,
    render_skipped,
)
from pathwise.services.knowledge.graph import GraphEdge, GraphNode, KnowledgeGraph
from pathwise.services.onboarding.goal import (
    GoalParser,
    ResolvedGoal,
    parse_goal_or_fallback,
)
from pathwise.services.roadmap.annotator import (
    RoadmapAnnotator,
    annotation_validator,
)
from pathwise.services.roadmap.planner import plan_roadmap
from pathwise.services.roadmap.schemas import (
    NodeRationale,
    ParsedGoal,
    RoadmapAnnotation,
)

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def cid(slug: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, slug)


def node(slug: str, **kwargs: object) -> GraphNode:
    return GraphNode(
        id=cid(slug),
        slug=slug,
        name=slug.replace("-", " ").title(),
        domain=str(kwargs.get("domain", "test")),
        estimated_minutes=int(kwargs.get("minutes", 60)),  # type: ignore[call-overload]
        description=str(kwargs.get("description", "A description of the concept.")),
    )


@pytest.fixture
def graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        [node(s) for s in ("basics", "intermediate", "goal")],
        [
            GraphEdge(cid("basics"), cid("intermediate"), RelationType.PREREQUISITE_OF, 1.0),
            GraphEdge(cid("intermediate"), cid("goal"), RelationType.PREREQUISITE_OF, 1.0),
        ],
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_provider="fake", jwt_secret="x" * 40)


@pytest.fixture
def recorder() -> InMemoryCallRecorder:
    return InMemoryCallRecorder()


def make_client(
    settings: Settings, recorder: InMemoryCallRecorder, behaviour: FakeBehaviour | None = None
) -> tuple[AIClient, FakeProvider]:
    provider = FakeProvider(behaviour)
    return AIClient(provider, settings, recorder=recorder), provider


# --------------------------------------------------------------------------- #
# Catalogue rendering
# --------------------------------------------------------------------------- #


def test_the_catalogue_lists_every_concept(graph: KnowledgeGraph) -> None:
    rendered = render_catalogue(graph)
    for slug in ("basics", "intermediate", "goal"):
        assert slug in rendered


def test_catalogue_ordering_is_stable(graph: KnowledgeGraph) -> None:
    """Prompt caching is a prefix match. A catalogue that shuffled between calls
    would invalidate the cache every request and quietly multiply cost."""
    assert render_catalogue(graph) == render_catalogue(graph)


def test_the_catalogue_can_be_restricted(graph: KnowledgeGraph) -> None:
    rendered = render_catalogue(graph, concept_ids=[cid("goal")])
    assert "goal" in rendered
    assert "basics" not in rendered


def test_plan_steps_carry_the_facts_a_rationale_needs(graph: KnowledgeGraph) -> None:
    """A model given only names would have to invent the reasons."""
    rendered = render_plan_steps(graph, [cid("basics"), cid("intermediate")])
    assert "builds on: basics" in rendered
    assert "unlocks: intermediate" in rendered


def test_skipped_rendering_states_the_evidence() -> None:
    assert "mastery 0.92" in render_skipped([("basics", 0.92, 7)])
    assert "7 recorded" in render_skipped([("basics", 0.92, 7)])


def test_nothing_skipped_renders_explicitly() -> None:
    """An empty string would leave the prompt with a dangling heading."""
    assert "nothing" in render_skipped([]).lower()


# --------------------------------------------------------------------------- #
# Goal parsing
# --------------------------------------------------------------------------- #


async def test_a_goal_resolves_to_concept_ids(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    client, _ = make_client(
        settings,
        recorder,
        FakeBehaviour(
            canned_values=[
                ParsedGoal(
                    goal_slugs=["goal"],
                    interpreted_goal="You want to reach the goal concept.",
                )
            ]
        ),
    )
    resolved = await GoalParser(client, graph).parse("I want to reach the goal")

    assert resolved.goal_concept_ids == (cid("goal"),)
    assert resolved.goal_slugs == ("goal",)
    assert resolved.is_actionable


async def test_an_invented_concept_is_rejected(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    """The containment property: a plausible-looking slug that is not in the graph
    must never become a goal, because nothing downstream could join to it."""
    invented = ParsedGoal(
        goal_slugs=["machine-learning-mastery"],
        interpreted_goal="You want to master machine learning.",
    )
    client, _ = make_client(settings, recorder, FakeBehaviour(canned_values=[invented, invented]))

    with pytest.raises(AIValidationError):
        await GoalParser(client, graph).parse("I want to master ML")


async def test_invented_prior_knowledge_is_also_rejected(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    """A fabricated "claimed knowledge" entry would seed a diagnostic for a concept
    that does not exist."""
    bad = ParsedGoal(
        goal_slugs=["goal"],
        interpreted_goal="You want to reach the goal concept.",
        claimed_knowledge=["not-a-real-concept"],
    )
    client, _ = make_client(settings, recorder, FakeBehaviour(canned_values=[bad, bad]))

    with pytest.raises(AIValidationError):
        await GoalParser(client, graph).parse("I want the goal, I know things")


async def test_empty_goal_text_is_rejected_before_any_model_call(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    client, provider = make_client(settings, recorder)
    with pytest.raises(ValidationError):
        await GoalParser(client, graph).parse("   ")
    assert provider.call_count == 0


async def test_a_vague_goal_can_ask_for_clarification(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    """Guessing here misdirects weeks of study, so asking is the better failure."""
    client, _ = make_client(
        settings,
        recorder,
        FakeBehaviour(
            canned_values=[
                ParsedGoal(
                    goal_slugs=["goal"],
                    interpreted_goal="You want to learn something technical.",
                    needs_clarification=True,
                    clarifying_question="Which area are you aiming at?",
                )
            ]
        ),
    )
    resolved = await GoalParser(client, graph).parse("I want to learn coding")

    assert resolved.needs_clarification
    assert not resolved.is_actionable
    assert resolved.clarifying_question


async def test_unmapped_topics_are_surfaced_not_dropped(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    """A domain Pathwise does not cover should be reported, not silently ignored."""
    client, _ = make_client(
        settings,
        recorder,
        FakeBehaviour(
            canned_values=[
                ParsedGoal(
                    goal_slugs=["goal"],
                    interpreted_goal="You want the goal, and also game development.",
                    unmapped_topics=["game development"],
                )
            ]
        ),
    )
    resolved = await GoalParser(client, graph).parse("goal plus gamedev")
    assert resolved.unmapped_topics == ("game development",)


async def test_the_catalogue_goes_in_the_cacheable_system_prefix(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    """It is identical for every learner, so it belongs before the cache breakpoint
    rather than in the per-request message."""
    client, provider = make_client(
        settings,
        recorder,
        FakeBehaviour(
            canned_values=[ParsedGoal(goal_slugs=["goal"], interpreted_goal="You want the goal.")]
        ),
    )
    await GoalParser(client, graph).parse("I want the goal")

    request = provider.last_request
    assert "basics" in request.system
    assert request.cache_system
    assert "basics" not in request.messages[0].content


async def test_the_call_is_attributed_to_the_learner(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    client, _ = make_client(
        settings,
        recorder,
        FakeBehaviour(
            canned_values=[ParsedGoal(goal_slugs=["goal"], interpreted_goal="You want the goal.")]
        ),
    )
    user_id = uuid.uuid4()
    await GoalParser(client, graph).parse("goal", user_id=user_id)

    assert recorder.records[0].user_id == user_id
    assert recorder.records[0].feature == "goal_parse"


async def test_the_fallback_asks_rather_than_guesses(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    """When the goal cannot be grounded, planning from a guess would be worse than
    asking the learner to name their target."""
    invented = ParsedGoal(goal_slugs=["nonexistent"], interpreted_goal="Something vague.")
    client, _ = make_client(settings, recorder, FakeBehaviour(canned_values=[invented, invented]))

    resolved = await parse_goal_or_fallback(GoalParser(client, graph), "mumble")

    assert isinstance(resolved, ResolvedGoal)
    assert resolved.needs_clarification
    assert not resolved.is_actionable
    assert resolved.clarifying_question


# --------------------------------------------------------------------------- #
# Annotation containment
# --------------------------------------------------------------------------- #


def test_an_annotation_covering_every_step_is_valid(graph: KnowledgeGraph) -> None:
    plan = plan_roadmap(graph, [cid("goal")], now=NOW)
    annotation = RoadmapAnnotation(
        title="A Path To The Goal",
        summary="A summary long enough to satisfy the schema minimum length rule here.",
        rationales=[
            NodeRationale(slug=slug, rationale="A rationale of sufficient length here.")
            for slug in plan.slugs
        ],
    )
    assert annotation_validator(plan)(annotation).is_valid


def test_a_smuggled_in_step_is_rejected(graph: KnowledgeGraph) -> None:
    """The planner decides what is in the roadmap. An extra rationale is the model
    trying to add a step, and it must not render."""
    plan = plan_roadmap(graph, [cid("goal")], now=NOW)
    annotation = RoadmapAnnotation(
        title="A Path To The Goal",
        summary="A summary long enough to satisfy the schema minimum length rule here.",
        rationales=[
            *(
                NodeRationale(slug=slug, rationale="A rationale of sufficient length here.")
                for slug in plan.slugs
            ),
            NodeRationale(slug="smuggled-step", rationale="A rationale of sufficient length."),
        ],
    )
    result = annotation_validator(plan)(annotation)
    assert not result.is_valid
    assert "not in this roadmap" in result.issues[0].problem


def test_a_missing_rationale_is_rejected(graph: KnowledgeGraph) -> None:
    plan = plan_roadmap(graph, [cid("goal")], now=NOW)
    annotation = RoadmapAnnotation(
        title="A Path To The Goal",
        summary="A summary long enough to satisfy the schema minimum length rule here.",
        rationales=[
            NodeRationale(slug=plan.slugs[0], rationale="A rationale of sufficient length.")
        ],
    )
    result = annotation_validator(plan)(annotation)
    assert not result.is_valid
    assert any("omit" in issue.problem for issue in result.issues)


def test_duplicate_rationales_are_rejected(graph: KnowledgeGraph) -> None:
    plan = plan_roadmap(graph, [cid("goal")], now=NOW)
    annotation = RoadmapAnnotation(
        title="A Path To The Goal",
        summary="A summary long enough to satisfy the schema minimum length rule here.",
        rationales=[
            *(
                NodeRationale(slug=slug, rationale="A rationale of sufficient length here.")
                for slug in plan.slugs
            ),
            NodeRationale(slug=plan.slugs[0], rationale="A second rationale, same step."),
        ],
    )
    result = annotation_validator(plan)(annotation)
    assert not result.is_valid
    assert any("duplicate" in issue.problem for issue in result.issues)


# --------------------------------------------------------------------------- #
# Annotation behaviour and fallback
# --------------------------------------------------------------------------- #


async def test_a_successful_annotation_is_attached_to_the_plan(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    plan = plan_roadmap(graph, [cid("goal")], now=NOW)
    annotation = RoadmapAnnotation(
        title="A Path To The Goal",
        summary="A summary long enough to satisfy the schema minimum length rule here.",
        rationales=[
            NodeRationale(slug=slug, rationale=f"Why {slug} matters, at sufficient length.")
            for slug in plan.slugs
        ],
    )
    client, _ = make_client(settings, recorder, FakeBehaviour(canned_values=[annotation]))

    result = await RoadmapAnnotator(client, graph).annotate(
        plan, interpreted_goal="You want the goal."
    )

    assert not result.is_fallback
    assert result.title == "A Path To The Goal"
    assert set(result.rationales) == set(plan.slugs)


async def test_a_provider_failure_still_delivers_a_roadmap(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    """Rationales are the part a learner can most afford to lose. An error page is
    a far worse outcome than a plainly-worded plan."""
    client, _ = make_client(
        settings, recorder, FakeBehaviour(fail_with=AIProviderError("upstream down"))
    )
    plan = plan_roadmap(graph, [cid("goal")], now=NOW)

    result = await RoadmapAnnotator(client, graph).annotate(
        plan, interpreted_goal="You want the goal."
    )

    assert result.is_fallback
    assert result.title
    assert set(result.rationales) == set(plan.slugs)
    assert recorder.records[-1].status is LLMCallStatus.PROVIDER_ERROR


async def test_a_refusal_still_delivers_a_roadmap(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    client, _ = make_client(settings, recorder, FakeBehaviour(refuse=True))
    plan = plan_roadmap(graph, [cid("goal")], now=NOW)

    result = await RoadmapAnnotator(client, graph).annotate(
        plan, interpreted_goal="You want the goal."
    )
    assert result.is_fallback
    assert result.rationales


async def test_fallback_rationales_are_built_from_graph_facts(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    """Plainer than generated prose, but every sentence is verifiably true."""
    client, _ = make_client(settings, recorder, FakeBehaviour(fail_with=AIProviderError("down")))
    plan = plan_roadmap(graph, [cid("goal")], now=NOW)

    result = await RoadmapAnnotator(client, graph).annotate(
        plan, interpreted_goal="You want the goal."
    )

    assert "builds on Basics" in result.rationale_for("intermediate")
    assert "needed for" in result.rationale_for("basics")


async def test_an_empty_plan_does_not_call_the_model(
    settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    """Nothing to annotate, so spending a call on it would be pure waste."""
    empty_graph = KnowledgeGraph([node("goal")], [])
    mastery: dict[uuid.UUID, object] = {}
    plan = plan_roadmap(empty_graph, [cid("goal")], mastery, now=NOW)  # type: ignore[arg-type]
    # A goal is never skipped, so build a genuinely empty plan by other means.
    empty_plan = type(plan)(
        nodes=(),
        edges=(),
        skipped=plan.skipped,
        pacing=plan.pacing,
        scope=plan.scope,
    )

    client, provider = make_client(settings, recorder)
    result = await RoadmapAnnotator(client, empty_graph).annotate(
        empty_plan, interpreted_goal="You want the goal."
    )

    assert provider.call_count == 0
    assert result.is_fallback
    assert "nothing left" in result.summary.lower()


async def test_the_annotation_prompt_receives_the_real_plan(
    graph: KnowledgeGraph, settings: Settings, recorder: InMemoryCallRecorder
) -> None:
    """The model must be given graph facts, not asked to recall them."""
    plan = plan_roadmap(graph, [cid("goal")], now=NOW)
    annotation = RoadmapAnnotation(
        title="A Path To The Goal",
        summary="A summary long enough to satisfy the schema minimum length rule here.",
        rationales=[
            NodeRationale(slug=slug, rationale="A rationale of sufficient length here.")
            for slug in plan.slugs
        ],
    )
    client, provider = make_client(settings, recorder, FakeBehaviour(canned_values=[annotation]))

    await RoadmapAnnotator(client, graph).annotate(plan, interpreted_goal="You want the goal.")

    sent = provider.last_request.messages[0].content
    assert "builds on: basics" in sent
    assert "You want the goal." in sent
