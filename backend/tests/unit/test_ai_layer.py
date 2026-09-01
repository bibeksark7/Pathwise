"""AI layer tests.

The claim this layer makes is "the application never blindly trusts a model
response". These tests are what makes that claim checkable: they exercise the repair
round-trip, the refusal path, the accounting that must not be bypassable, and the
hallucination guards — all offline, against the deterministic fake provider.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import BaseModel, Field, model_validator

from pathwise.ai.cache import InMemoryResponseCache, NullResponseCache, cacheable
from pathwise.ai.call_log import CallRecord, InMemoryCallRecorder
from pathwise.ai.client import MAX_REPAIR_ATTEMPTS, AIClient, build_provider
from pathwise.ai.cost import (
    CACHE_READ_MULTIPLIER,
    PRICING,
    breakdown,
    estimate_cost,
    format_usd,
    pricing_for,
)
from pathwise.ai.prompts.registry import ACTIVE_VERSIONS, Prompt, PromptRegistry
from pathwise.ai.providers.base import (
    Effort,
    LLMRequest,
    LLMResponse,
    TokenUsage,
    single,
)
from pathwise.ai.providers.fake_provider import FakeBehaviour, FakeProvider, synthesise
from pathwise.ai.validators import (
    ValidationPipeline,
    ValidationResult,
    acyclic_edges,
    grounded_in_trace,
    known_concepts,
    known_resources,
    non_empty,
)
from pathwise.api.errors import (
    AIProviderError,
    AIRefusalError,
    AIValidationError,
    NotFoundError,
    ValidationError,
)
from pathwise.config import Settings
from pathwise.models.enums import LLMCallStatus


class Plan(BaseModel):
    """A stand-in for a real generated artifact."""

    title: str = Field(min_length=3)
    concept_slugs: list[str] = Field(min_length=1)
    rationale: str = ""


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_provider="fake", jwt_secret="test-secret-of-at-least-32-bytes!!")


@pytest.fixture
def recorder() -> InMemoryCallRecorder:
    return InMemoryCallRecorder()


@pytest.fixture
def registry(tmp_path) -> PromptRegistry:
    """A registry with one throwaway prompt, so tests do not depend on shipped text."""
    directory = tmp_path / "prompts"
    (directory / "demo").mkdir(parents=True)
    (directory / "demo" / "v1.md").write_text("Plan for $goal in $hours hours.", encoding="utf-8")
    (directory / "demo" / "v2.md").write_text("v2: plan for $goal.", encoding="utf-8")
    return PromptRegistry(directory)


def make_client(
    settings: Settings,
    recorder: InMemoryCallRecorder,
    registry: PromptRegistry,
    behaviour: FakeBehaviour | None = None,
    **kwargs: object,
) -> tuple[AIClient, FakeProvider]:
    provider = FakeProvider(behaviour)
    client = AIClient(provider, settings, recorder=recorder, registry=registry, **kwargs)  # type: ignore[arg-type]
    return client, provider


# --------------------------------------------------------------------------- #
# Cost accounting
# --------------------------------------------------------------------------- #


def test_cost_is_computed_from_actual_usage() -> None:
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_cost("claude-opus-5", usage) == pytest.approx(30.0)  # $5 in + $25 out


def test_cached_reads_cost_a_fraction_of_fresh_input() -> None:
    """The whole point of prompt caching; collapsing the two would hide it."""
    fresh = estimate_cost("claude-opus-5", TokenUsage(input_tokens=1_000_000))
    cached = estimate_cost("claude-opus-5", TokenUsage(cache_read_tokens=1_000_000))
    assert cached == pytest.approx(fresh * CACHE_READ_MULTIPLIER)


def test_cache_writes_cost_more_than_fresh_input() -> None:
    """Caching only pays off on reuse — a written-once, never-read prefix loses money."""
    fresh = estimate_cost("claude-opus-5", TokenUsage(input_tokens=1_000_000))
    written = estimate_cost("claude-opus-5", TokenUsage(cache_write_tokens=1_000_000))
    assert written > fresh


def test_an_unpriced_model_does_not_raise() -> None:
    """A missing price is an accounting gap, not a reason to fail a user's request."""
    assert estimate_cost("some-future-model", TokenUsage(input_tokens=1000)) == 0.0
    assert pricing_for("some-future-model") is None


def test_every_configured_model_is_priced() -> None:
    """Catches the real failure: adding a model to config without adding its price,
    which would silently under-report spend for every call that routes to it."""
    settings = Settings(jwt_secret="x" * 40)
    for model in (settings.llm_model, settings.llm_fast_model):
        assert model in PRICING, f"{model} is configured but has no pricing entry"


def test_breakdown_reports_what_caching_saved() -> None:
    saved = breakdown("claude-opus-5", TokenUsage(cache_read_tokens=1_000_000)).cache_savings
    assert saved == pytest.approx(5.0 * (1 - CACHE_READ_MULTIPLIER))


def test_no_cache_reads_means_no_savings() -> None:
    assert breakdown("claude-opus-5", TokenUsage(input_tokens=1000)).cache_savings == 0.0


def test_sub_cent_costs_are_not_rendered_as_zero() -> None:
    """Individual calls cost fractions of a cent; two decimals would show every one
    of them as $0.00 and make the usage dashboard useless."""
    assert format_usd(0.000042) == "$0.000042"
    assert format_usd(0.0) == "$0.00"
    assert format_usd(12.5) == "$12.50"


def test_the_fake_model_is_free_so_tests_do_not_pollute_cost_data() -> None:
    assert estimate_cost("fake-model", TokenUsage(input_tokens=999_999)) == 0.0


# --------------------------------------------------------------------------- #
# Token usage
# --------------------------------------------------------------------------- #


def test_usage_accumulates_across_attempts() -> None:
    """A repair round-trip costs twice; the recorded usage must reflect both calls."""
    combined = TokenUsage(input_tokens=10, output_tokens=5) + TokenUsage(
        input_tokens=20, output_tokens=7
    )
    assert combined.input_tokens == 30
    assert combined.total_tokens == 42


def test_cache_hit_is_detectable() -> None:
    assert not TokenUsage(input_tokens=100).cache_hit
    assert TokenUsage(cache_read_tokens=1).cache_hit


# --------------------------------------------------------------------------- #
# Request identity
# --------------------------------------------------------------------------- #


def test_identical_requests_share_a_cache_key() -> None:
    assert LLMRequest(messages=single("hello")).cache_key() == (
        LLMRequest(messages=single("hello")).cache_key()
    )


def test_effort_changes_the_cache_key() -> None:
    """Serving a cached low-effort answer to a high-effort call would silently
    degrade quality with no signal that it happened."""
    low = LLMRequest(messages=single("hello"), effort=Effort.LOW)
    high = LLMRequest(messages=single("hello"), effort=Effort.HIGH)
    assert low.cache_key() != high.cache_key()


@pytest.mark.parametrize(
    "changed",
    [
        {"system": "different"},
        {"model": "claude-haiku-4-5"},
        {"max_tokens": 500},
        {"messages": single("a different question")},
    ],
)
def test_anything_affecting_the_response_changes_the_key(changed: dict[str, object]) -> None:
    base = LLMRequest(messages=single("hello"))
    assert base.cache_key() != LLMRequest(**{**{"messages": base.messages}, **changed}).cache_key()  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The fake provider
# --------------------------------------------------------------------------- #


async def test_the_fake_is_deterministic_across_instances() -> None:
    """Byte-identical CI runs depend on this."""
    request = LLMRequest(messages=single("explain backpropagation"))
    first = await FakeProvider().complete(request)
    second = await FakeProvider().complete(request)
    assert first.text == second.text


async def test_different_prompts_yield_different_output() -> None:
    provider = FakeProvider()
    a = await provider.complete(LLMRequest(messages=single("question one")))
    b = await provider.complete(LLMRequest(messages=single("question two")))
    assert a.text != b.text


async def test_the_fake_synthesises_schema_valid_output() -> None:
    """The reason this is a real provider and not a mock: it exercises the actual
    Pydantic schema, so a broken schema fails in CI rather than on the first live call."""
    result = await FakeProvider().complete_structured(
        LLMRequest(messages=single("plan something")), Plan
    )
    assert isinstance(result.value, Plan)
    assert len(result.value.title) >= 3
    assert result.value.concept_slugs


def test_synthesis_respects_schema_constraints() -> None:
    class Constrained(BaseModel):
        score: int = Field(ge=10, le=20)
        label: str = Field(pattern=r"^lo-\d+$")
        items: list[str] = Field(min_length=2)

    value = synthesise(Constrained, seed="deterministic")
    assert 10 <= value.score <= 20
    assert value.label == "lo-1"
    assert len(value.items) >= 2


def test_synthesis_fails_loudly_on_a_constraint_it_cannot_model() -> None:
    """The synthesiser walks the JSON schema, so a rule expressed only in Python —
    a cross-field validator — is invisible to it. Better to fail here than to hand
    back output the real schema would reject at the call site."""

    class CrossFieldRule(BaseModel):
        low: int
        high: int

        @model_validator(mode="after")
        def _ordered(self) -> CrossFieldRule:
            if self.low >= self.high:
                raise ValueError("low must be below high")
            return self

    with pytest.raises(AIProviderError, match="could not synthesise"):
        synthesise(CrossFieldRule, seed="x")


async def test_the_fake_records_what_it_was_asked() -> None:
    provider = FakeProvider()
    await provider.complete(LLMRequest(messages=single("first")))
    await provider.complete(LLMRequest(messages=single("second")))
    assert provider.call_count == 2
    assert provider.last_request.messages[-1].content == "second"


async def test_canned_values_let_a_test_script_a_flow() -> None:
    plan = Plan(title="Scripted", concept_slugs=["python"])
    provider = FakeProvider(FakeBehaviour(canned_values=[plan]))
    result = await provider.complete_structured(LLMRequest(messages=single("x")), Plan)
    assert result.value.title == "Scripted"


# --------------------------------------------------------------------------- #
# Prompt registry
# --------------------------------------------------------------------------- #


def test_a_prompt_renders_its_variables(registry: PromptRegistry) -> None:
    rendered = registry.get("demo", "v1").render(goal="ML engineering", hours=8)
    assert "ML engineering" in rendered
    assert "$" not in rendered


def test_a_missing_variable_is_rejected(registry: PromptRegistry) -> None:
    """Rendering the literal `$goal` into the prompt would send the placeholder to
    the model as if it were content."""
    with pytest.raises(ValidationError, match="missing required variables"):
        registry.get("demo", "v1").render(goal="ML")


def test_declared_variables_are_discoverable(registry: PromptRegistry) -> None:
    assert registry.get("demo", "v1").variables == {"goal", "hours"}


def test_dollar_templating_survives_literal_json_braces() -> None:
    """Prompts contain JSON examples; `str.format` would crash or mangle them."""
    prompt = Prompt(name="t", version="v1", template='Return {"key": "value"} for $topic.')
    assert prompt.render(topic="graphs") == 'Return {"key": "value"} for graphs.'


def test_editing_a_prompt_changes_its_checksum() -> None:
    """Catches what a version number misses: text edited under a live version."""
    first = Prompt(name="t", version="v1", template="original")
    second = Prompt(name="t", version="v1", template="edited")
    assert first.checksum != second.checksum


def test_an_unpinned_prompt_is_an_error(registry: PromptRegistry) -> None:
    with pytest.raises(NotFoundError, match="No active version"):
        registry.get("never_registered")


def test_a_missing_version_file_is_an_error(registry: PromptRegistry) -> None:
    with pytest.raises(NotFoundError, match="not found"):
        registry.get("demo", "v99")


def test_adding_a_version_file_does_not_activate_it(registry: PromptRegistry) -> None:
    """Promotion is an explicit, reviewable diff — not a consequence of adding a file."""
    assert registry.versions("demo") == ("v1", "v2")
    assert "v2" in registry.get("demo", "v2").template
    # ACTIVE_VERSIONS has no entry for this fixture prompt, so there is no default.
    with pytest.raises(NotFoundError):
        registry.get("demo")


def test_every_shipped_prompt_is_pinned_and_present() -> None:
    """Guards the real registry: a prompt file with no pin, or a pin with no file."""
    real = PromptRegistry()
    for name in real.discovered_names():
        assert name in ACTIVE_VERSIONS, f"prompt '{name}' exists but is not in ACTIVE_VERSIONS"
    for name, version in ACTIVE_VERSIONS.items():
        assert real.get(name, version).template.strip(), f"{name}@{version} is empty"


def test_the_decision_explain_prompt_forbids_inventing_numbers() -> None:
    """This prompt is the load-bearing half of the explainability claim; if its
    grounding instruction is ever edited away, this fails."""
    template = PromptRegistry().get("decision_explain").template.lower()
    assert "only" in template
    assert "trace" in template


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #


def test_invented_concept_slugs_are_rejected() -> None:
    validate = known_concepts({"gradient-descent"}, lambda p: p.concept_slugs)
    result = validate(Plan(title="Plan", concept_slugs=["gradient-descent", "made-up-topic"]))
    assert not result.is_valid


def test_a_near_miss_gets_a_suggestion() -> None:
    """The repair hint has to be actionable — "invalid slug" repairs far worse than
    "did you mean 'gradient-descent'?"."""
    validate = known_concepts({"gradient-descent"}, lambda p: p.concept_slugs)
    result = validate(Plan(title="Plan", concept_slugs=["gradient-decent"]))
    assert "gradient-descent" in result.issues[0].repair_hint


def test_invented_urls_are_rejected() -> None:
    """The rule enforcing 'never let the model invent a resource URL'."""

    class Rec(BaseModel):
        urls: list[str]

    validate = known_resources({"https://real.example/course"}, lambda r: r.urls)
    result = validate(Rec(urls=["https://plausible-but-fake.example/course"]))
    assert not result.is_valid
    assert "Never write a URL of your own" in result.issues[0].repair_hint


def test_cyclic_prerequisites_are_rejected() -> None:
    class Edges(BaseModel):
        pairs: list[tuple[str, str]]

    validate = acyclic_edges(lambda e: e.pairs)
    assert validate(Edges(pairs=[("a", "b"), ("b", "c")])).is_valid
    assert not validate(Edges(pairs=[("a", "b"), ("b", "c"), ("c", "a")])).is_valid


def test_empty_fields_are_rejected() -> None:
    """A schema typed `str` accepts `""`, which is a silently degraded product."""
    validate = non_empty("rationale")
    assert not validate(Plan(title="Plan", concept_slugs=["x"], rationale="   ")).is_valid
    assert validate(Plan(title="Plan", concept_slugs=["x"], rationale="Because.")).is_valid


def test_prose_may_cite_numbers_from_the_trace() -> None:
    validate = grounded_in_trace([0.48, 0.88])
    assert validate("You scored 48% while your calculus sits at 0.88.").is_valid


def test_prose_may_not_invent_numbers() -> None:
    """The hallucination guard. An invented "72%" is indistinguishable from a real
    one to the reader, and corrodes trust in every other number shown."""
    result = grounded_in_trace([0.48])("You scored 72% on that assessment.")
    assert not result.is_valid
    assert "72" in result.issues[0].problem


def test_small_counting_words_are_not_treated_as_statistics() -> None:
    """ "the first 2 topics" must not be flagged as a fabricated figure."""
    assert grounded_in_trace([0.5])("The next 2 topics build on 1 idea.").is_valid


def test_a_pipeline_reports_every_issue_at_once() -> None:
    """Fixing issues one at a time costs a call each and lets the model reintroduce
    an earlier problem while addressing a later one."""
    pipeline = ValidationPipeline(
        known_concepts({"real"}, lambda p: p.concept_slugs),
        non_empty("rationale"),
    )
    result = pipeline(Plan(title="Plan", concept_slugs=["fake"], rationale=""))
    assert len(result.issues) == 2


def test_repair_instructions_name_every_problem() -> None:
    result = ValidationResult()
    result.add("slugs", "unknown concept", "Use the catalogue.")
    result.add("rationale", "is empty", "Explain the choice.")
    instructions = result.repair_instructions()
    assert "1." in instructions
    assert "2." in instructions
    assert "Use the catalogue." in instructions


def test_enforce_raises_on_invalid_output() -> None:
    pipeline = ValidationPipeline(non_empty("rationale"))
    with pytest.raises(AIValidationError):
        pipeline.enforce(Plan(title="Plan", concept_slugs=["x"], rationale=""))


# --------------------------------------------------------------------------- #
# The client: repair, refusal, accounting
# --------------------------------------------------------------------------- #


async def test_a_valid_response_is_returned_and_recorded(
    settings: Settings, recorder: InMemoryCallRecorder, registry: PromptRegistry
) -> None:
    client, _ = make_client(settings, recorder, registry)
    value = await client.generate_structured(
        feature="planning",
        prompt_name="demo",
        prompt_version="v1",
        schema=Plan,
        variables={"goal": "ML", "hours": 8},
    )
    assert isinstance(value, Plan)
    assert len(recorder.records) == 1
    assert recorder.records[0].status is LLMCallStatus.SUCCESS
    assert recorder.records[0].feature == "planning"


async def test_invalid_output_triggers_exactly_one_repair(
    settings: Settings, recorder: InMemoryCallRecorder, registry: PromptRegistry
) -> None:
    """A model that fails twice will not succeed on a third try, and each attempt
    costs money — so the cap is one, and then the deterministic fallback."""
    client, provider = make_client(settings, recorder, registry)

    always_invalid = known_concepts(set(), lambda p: p.concept_slugs)

    with pytest.raises(AIValidationError):
        await client.generate_structured(
            feature="planning",
            prompt_name="demo",
            prompt_version="v1",
            schema=Plan,
            variables={"goal": "ML", "hours": 8},
            validate=always_invalid,
        )

    assert provider.call_count == 1 + MAX_REPAIR_ATTEMPTS


async def test_the_repair_message_carries_the_specific_errors(
    settings: Settings, recorder: InMemoryCallRecorder, registry: PromptRegistry
) -> None:
    """The model is shown what was wrong, not just asked to try again."""
    client, provider = make_client(settings, recorder, registry)

    with pytest.raises(AIValidationError):
        await client.generate_structured(
            feature="planning",
            prompt_name="demo",
            prompt_version="v1",
            schema=Plan,
            variables={"goal": "ML", "hours": 8},
            validate=known_concepts({"only-this"}, lambda p: p.concept_slugs),
        )

    repair_prompt = provider.last_request.messages[-1].content
    assert "problems" in repair_prompt.lower()
    assert "only-this" in repair_prompt


async def test_a_failed_validation_is_recorded_as_such(
    settings: Settings, recorder: InMemoryCallRecorder, registry: PromptRegistry
) -> None:
    """The validation-failure rate is only meaningful if failures are in the
    denominator."""
    client, _ = make_client(settings, recorder, registry)

    with pytest.raises(AIValidationError):
        await client.generate_structured(
            feature="planning",
            prompt_name="demo",
            prompt_version="v1",
            schema=Plan,
            variables={"goal": "ML", "hours": 8},
            validate=known_concepts(set(), lambda p: p.concept_slugs),
        )

    final = recorder.records[-1]
    assert final.status is LLMCallStatus.VALIDATION_FAILED
    assert not final.validation_passed
    assert final.validation_errors
    assert final.repair_attempts == MAX_REPAIR_ATTEMPTS


async def test_a_refusal_becomes_an_error_not_an_empty_answer(
    settings: Settings, recorder: InMemoryCallRecorder, registry: PromptRegistry
) -> None:
    """A refusal arrives as HTTP 200. Unhandled, it reads as a successful blank
    response and would be stored as a learner's roadmap."""
    client, _ = make_client(settings, recorder, registry, FakeBehaviour(refuse=True))

    with pytest.raises(AIRefusalError):
        await client.generate(
            feature="planning",
            prompt_name="demo",
            prompt_version="v1",
            variables={"goal": "ML", "hours": 8},
        )

    assert recorder.records[-1].status is LLMCallStatus.REFUSED


async def test_provider_failures_are_recorded_before_propagating(
    settings: Settings, recorder: InMemoryCallRecorder, registry: PromptRegistry
) -> None:
    client, _ = make_client(
        settings, recorder, registry, FakeBehaviour(fail_with=AIProviderError("upstream down"))
    )

    with pytest.raises(AIProviderError):
        await client.generate(
            feature="planning",
            prompt_name="demo",
            prompt_version="v1",
            variables={"goal": "ML", "hours": 8},
        )

    record = recorder.records[-1]
    assert record.status is LLMCallStatus.PROVIDER_ERROR
    assert record.error_type == "AIProviderError"


async def test_every_record_carries_the_prompt_it_used(
    settings: Settings, recorder: InMemoryCallRecorder, registry: PromptRegistry
) -> None:
    """Provenance: any stored output traces to the exact prompt text behind it."""
    client, _ = make_client(settings, recorder, registry)
    await client.generate(
        feature="planning",
        prompt_name="demo",
        prompt_version="v1",
        variables={"goal": "ML", "hours": 8},
    )
    record = recorder.records[0]
    assert record.prompt_name == "demo"
    assert record.prompt_version == "v1"
    assert len(record.prompt_hash) == 16


async def test_the_user_is_attributed_when_supplied(
    settings: Settings, recorder: InMemoryCallRecorder, registry: PromptRegistry
) -> None:
    client, _ = make_client(settings, recorder, registry)
    user_id = uuid.uuid4()
    await client.generate(
        feature="planning",
        prompt_name="demo",
        prompt_version="v1",
        variables={"goal": "ML", "hours": 8},
        user_id=user_id,
    )
    assert recorder.records[0].user_id == user_id


async def test_streaming_yields_chunks_and_still_records(
    settings: Settings, recorder: InMemoryCallRecorder, registry: PromptRegistry
) -> None:
    client, _ = make_client(settings, recorder, registry)
    chunks = [
        chunk
        async for chunk in client.stream(
            feature="tutor",
            prompt_name="demo",
            prompt_version="v1",
            variables={"goal": "ML", "hours": 8},
        )
    ]
    assert len(chunks) > 1
    assert recorder.records[-1].feature == "tutor"


# --------------------------------------------------------------------------- #
# Response caching
# --------------------------------------------------------------------------- #


async def test_caching_is_off_unless_requested(
    settings: Settings, recorder: InMemoryCallRecorder, registry: PromptRegistry
) -> None:
    """The key does not include who asked, so caching learner-specific output would
    serve one person another's answer. It has to be opt-in."""
    cache = InMemoryResponseCache()
    client, provider = make_client(settings, recorder, registry, cache=cache)

    for _ in range(2):
        await client.generate(
            feature="planning",
            prompt_name="demo",
            prompt_version="v1",
            variables={"goal": "ML", "hours": 8},
        )
    assert provider.call_count == 2


async def test_an_identical_request_is_served_from_cache(
    settings: Settings, recorder: InMemoryCallRecorder, registry: PromptRegistry
) -> None:
    cache = InMemoryResponseCache()
    client, provider = make_client(settings, recorder, registry, cache=cache)

    for _ in range(2):
        await client.generate(
            feature="planning",
            prompt_name="demo",
            prompt_version="v1",
            variables={"goal": "ML", "hours": 8},
            use_cache=True,
        )

    assert provider.call_count == 1
    assert cache.hits == 1
    assert recorder.records[-1].status is LLMCallStatus.CACHED


def test_refusals_and_truncations_are_not_cached() -> None:
    """Caching a refusal makes a transient safety decision permanent; caching a
    truncated document serves broken JSON forever."""
    assert cacheable(LLMResponse(text="fine", model="m"))
    assert not cacheable(LLMResponse(text="", model="m", stop_reason="refusal"))
    assert not cacheable(LLMResponse(text="{partial", model="m", stop_reason="max_tokens"))


async def test_the_null_cache_never_returns_anything() -> None:
    cache = NullResponseCache()
    await cache.set("k", LLMResponse(text="x", model="m"), ttl_seconds=60)
    assert await cache.get("k") is None


# --------------------------------------------------------------------------- #
# Accounting roll-ups
# --------------------------------------------------------------------------- #


def test_the_recorder_totals_cost_and_tokens() -> None:
    recorder = InMemoryCallRecorder()
    recorder.records.append(
        CallRecord(
            feature="planning",
            provider="anthropic",
            model="claude-opus-5",
            prompt_name="p",
            prompt_version="v1",
            prompt_hash="abc",
            status=LLMCallStatus.SUCCESS,
            usage=TokenUsage(input_tokens=1_000_000, output_tokens=0),
        )
    )
    assert recorder.total_cost_usd == pytest.approx(5.0)
    assert recorder.total_tokens == 1_000_000
    assert len(recorder.by_feature("planning")) == 1


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #


def test_the_fake_provider_is_selectable_by_configuration() -> None:
    """One environment variable puts the whole suite offline."""
    provider = build_provider(Settings(llm_provider="fake", jwt_secret="x" * 40))
    assert provider.name == "fake"


def test_an_unimplemented_provider_fails_clearly() -> None:
    with pytest.raises(AIProviderError, match="not implemented"):
        build_provider(Settings(llm_provider="openai", jwt_secret="x" * 40))
