"""A deterministic LLM for tests, CI, and local development without a key.

This is not a mock in the usual sense — it is a full `LLMProvider` implementation
that produces *valid, schema-conforming* output without a network call. That matters
more than it sounds:

* CI runs the whole suite offline, for free, with byte-identical results every time.
* A structured-output test exercises the real Pydantic schema, so a bad schema fails
  in CI rather than on the first real call.
* Failure modes that are awkward to provoke against a live API — refusals,
  truncation, malformed output, transport errors — are one line to arrange here.

Output is derived from a hash of the request, so the same prompt always yields the
same answer, and two different prompts yield different ones. That gives tests
something stable to assert on without pinning them to a fixed string.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from pathwise.ai.providers.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    SchemaT,
    StructuredResponse,
    TokenUsage,
)
from pathwise.api.errors import AIProviderError, AIRefusalError, AIValidationError

FAKE_MODEL = "fake-model"


@dataclass(slots=True)
class RecordedCall:
    """One call the fake received, for assertions about how it was invoked."""

    request: LLMRequest
    schema: type[BaseModel] | None = None


@dataclass(slots=True)
class FakeBehaviour:
    """How the fake should respond. Everything defaults to success.

    Set exactly one failure flag to exercise a specific path; setting several is
    resolved in the order they are checked in ``_apply_behaviour``.
    """

    #: Raise a transport-style failure, as a provider outage would.
    fail_with: Exception | None = None
    #: Return `stop_reason="refusal"` rather than content.
    refuse: bool = False
    #: Return `stop_reason="max_tokens"`, as an over-long generation would.
    truncate: bool = False
    #: Return text that is not valid JSON, to exercise the repair round-trip.
    emit_invalid_json: bool = False
    #: Fail this many times before succeeding. Exercises retry logic.
    fail_times: int = 0
    #: Fixed text to return, bypassing hash-derived output.
    canned_text: str | None = None
    #: Fixed structured values, popped in order. Lets a test script a whole flow.
    canned_values: list[BaseModel] = field(default_factory=list)
    #: Report cached input tokens, so cache-accounting paths can be tested.
    cache_read_tokens: int = 0
    #: Artificial delay, for testing timeout and concurrency behaviour.
    latency_seconds: float = 0.0


class FakeProvider(LLMProvider):
    """A deterministic provider. Never makes a network call."""

    name: ClassVar[str] = "fake"

    def __init__(
        self,
        behaviour: FakeBehaviour | None = None,
        *,
        responder: Callable[[LLMRequest], str] | None = None,
    ) -> None:
        self.behaviour = behaviour or FakeBehaviour()
        #: Optional hook for a test that needs output to depend on the prompt.
        self._responder = responder
        self.calls: list[RecordedCall] = []
        self._failures_remaining = self.behaviour.fail_times

    @property
    def default_model(self) -> str:
        return FAKE_MODEL

    # --- introspection for tests ---------------------------------------------- #

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_request(self) -> LLMRequest:
        if not self.calls:
            raise AssertionError("FakeProvider received no calls")
        return self.calls[-1].request

    def reset(self) -> None:
        self.calls.clear()
        self._failures_remaining = self.behaviour.fail_times

    # --- the provider interface ------------------------------------------------ #

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(RecordedCall(request))
        await self._apply_behaviour()

        if self.behaviour.refuse:
            return self._response(request, text="", stop_reason="refusal")

        text = (
            self.behaviour.canned_text
            if self.behaviour.canned_text is not None
            else self._deterministic_text(request)
        )
        stop_reason = "max_tokens" if self.behaviour.truncate else "end_turn"
        return self._response(request, text=text, stop_reason=stop_reason)

    async def complete_structured(
        self, request: LLMRequest, schema: type[SchemaT]
    ) -> StructuredResponse[SchemaT]:
        self.calls.append(RecordedCall(request, schema))
        await self._apply_behaviour()

        if self.behaviour.refuse:
            raise AIRefusalError("The fake provider was configured to refuse.")

        if self.behaviour.emit_invalid_json:
            raise AIValidationError(
                "Model output was not valid JSON.", raw_output="{not json at all"
            )

        if self.behaviour.canned_values:
            value = self.behaviour.canned_values.pop(0)
            if not isinstance(value, schema):
                raise AIProviderError(
                    f"Canned value is a {type(value).__name__}, not {schema.__name__}."
                )
            return StructuredResponse(
                value=value, raw=self._response(request, text=value.model_dump_json())
            )

        instance = synthesise(schema, seed=request.cache_key())
        return StructuredResponse(
            value=instance, raw=self._response(request, text=instance.model_dump_json())
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Yield the deterministic response word by word."""
        self.calls.append(RecordedCall(request))
        await self._apply_behaviour()

        if self.behaviour.refuse:
            raise AIRefusalError("The fake provider was configured to refuse.")

        text = self.behaviour.canned_text or self._deterministic_text(request)
        for word in text.split(" "):
            yield word + " "

    async def count_tokens(self, request: LLMRequest) -> int:
        """A stable word-count approximation.

        Not calibrated against any real tokeniser — nothing that matters should
        depend on the exact number, and a test asserting an exact count against the
        fake would be asserting nothing about production.
        """
        words = len(request.system.split()) + sum(len(m.content.split()) for m in request.messages)
        return int(words * 1.3) + 8

    # --- internals ------------------------------------------------------------- #

    async def _apply_behaviour(self) -> None:
        if self.behaviour.latency_seconds:
            await asyncio.sleep(self.behaviour.latency_seconds)

        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise AIProviderError(
                "Simulated transient provider failure.",
                attempts_remaining=self._failures_remaining,
            )

        if self.behaviour.fail_with is not None:
            raise self.behaviour.fail_with

    def _deterministic_text(self, request: LLMRequest) -> str:
        if self._responder is not None:
            return self._responder(request)
        digest = request.cache_key()[:8]
        return f"Fake response {digest} to: {request.messages[-1].content[:80]}"

    def _response(
        self, request: LLMRequest, *, text: str, stop_reason: str = "end_turn"
    ) -> LLMResponse:
        prompt_words = sum(len(m.content.split()) for m in request.messages)
        return LLMResponse(
            text=text,
            model=self.resolve_model(request),
            usage=TokenUsage(
                input_tokens=prompt_words + len(request.system.split()),
                output_tokens=len(text.split()),
                cache_read_tokens=self.behaviour.cache_read_tokens,
            ),
            stop_reason=stop_reason,
            latency_ms=1.0,
            provider_request_id=f"fake_{request.cache_key()[:12]}",
            refusal_category="fake_refusal" if stop_reason == "refusal" else None,
        )


# --------------------------------------------------------------------------- #
# Schema synthesis
# --------------------------------------------------------------------------- #


def synthesise(schema: type[SchemaT], *, seed: str) -> SchemaT:
    """Build a valid instance of ``schema`` deterministically from ``seed``.

    Walks the model's JSON schema and fills each field with a plausible value that
    satisfies its constraints — respecting enums, string patterns and lengths,
    numeric bounds, and nesting.

    The alternative, hand-written fixtures per schema, rots: a field added to a
    Pydantic model breaks every fixture at once and tempts people to loosen the
    schema instead of updating them. Synthesising from the schema means the fake
    tracks the model automatically, and a schema that cannot be satisfied fails here
    — which is itself worth knowing.
    """
    rng = random.Random(hashlib.sha256(f"{schema.__name__}:{seed}".encode()).hexdigest())
    json_schema = schema.model_json_schema()
    payload = _synthesise_object(json_schema, json_schema, rng)

    try:
        return schema.model_validate(payload)
    except PydanticValidationError as exc:
        raise AIProviderError(
            f"FakeProvider could not synthesise a valid {schema.__name__}. "
            "The schema likely has a constraint the synthesiser does not model "
            "(a custom validator, or a cross-field rule).",
            schema=schema.__name__,
            errors=str(exc)[:500],
        ) from exc


def _resolve_ref(node: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Follow a local ``$ref`` into the schema's ``$defs``."""
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return node
    resolved = root.get("$defs", {}).get(ref.removeprefix("#/$defs/"), {})
    return resolved if isinstance(resolved, dict) else {}


def _synthesise_object(
    node: dict[str, Any], root: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    node = _resolve_ref(node, root)
    properties: dict[str, Any] = node.get("properties", {})
    required = set(node.get("required", []))

    result: dict[str, Any] = {}
    for name, spec in properties.items():
        # Optional fields with a default are left out, so the synthesised value
        # exercises the defaults the real schema declares.
        if name not in required and "default" in spec:
            continue
        result[name] = _synthesise_value(name, spec, root, rng)
    return result


def _synthesise_value(
    name: str, spec: dict[str, Any], root: dict[str, Any], rng: random.Random
) -> Any:
    spec = _resolve_ref(spec, root)

    if "const" in spec:
        return spec["const"]
    if enum_values := spec.get("enum"):
        return enum_values[rng.randrange(len(enum_values))]

    # A union (`str | None`, or a discriminated union) — take the first concrete
    # branch so the value is deterministic.
    for key in ("anyOf", "oneOf"):
        if branches := spec.get(key):
            concrete = [b for b in branches if b.get("type") != "null"] or branches
            return _synthesise_value(name, concrete[0], root, rng)

    schema_type = spec.get("type")

    if schema_type == "object" or "properties" in spec:
        return _synthesise_object(spec, root, rng)

    if schema_type == "array":
        item_spec = spec.get("items", {"type": "string"})
        count = max(int(spec.get("minItems", 1)), 1)
        count = min(count, int(spec.get("maxItems", count)))
        return [_synthesise_value(name, item_spec, root, rng) for _ in range(count)]

    if schema_type == "integer":
        low: int = int(spec.get("minimum", int(spec.get("exclusiveMinimum", -1)) + 1))
        high: int = int(spec.get("maximum", int(spec.get("exclusiveMaximum", low + 101)) - 1))
        return rng.randint(low, max(low, high))

    if schema_type == "number":
        lower = float(spec.get("minimum", 0.0))
        upper = float(spec.get("maximum", max(lower + 1.0, 1.0)))
        return round(rng.uniform(lower, upper), 4)

    if schema_type == "boolean":
        return rng.random() > 0.5

    if schema_type == "null":
        return None

    return _synthesise_string(name, spec, rng)


#: Patterns the seed and prompt schemas use, mapped to a satisfying value. A pattern
#: not listed here falls through to a generic string, which will fail validation —
#: loudly, in a test, which is the intent.
_PATTERN_SAMPLES: dict[str, str] = {
    r"^lo-\d+$": "lo-1",
    r"^[a-z0-9]+(?:-[a-z0-9]+)*$": "a-valid-slug",
    r"^[a-z0-9-]+$": "a-valid-slug",
    "^(remember|understand|apply|analyze|evaluate|create)$": "understand",
}


def _synthesise_string(name: str, spec: dict[str, Any], rng: random.Random) -> str:
    if pattern := spec.get("pattern"):
        if sample := _PATTERN_SAMPLES.get(pattern):
            return sample
        return "a-valid-slug"

    fmt = spec.get("format")
    if fmt == "email":
        return "learner@example.com"
    if fmt == "uri":
        return "https://example.com/resource"
    if fmt in {"date-time", "date"}:
        return "2026-01-01T00:00:00Z" if fmt == "date-time" else "2026-01-01"
    if fmt == "uuid":
        return f"00000000-0000-4000-8000-{rng.randrange(16**12):012x}"

    minimum = int(spec.get("minLength", 0))
    maximum = int(spec.get("maxLength", 10_000))

    base = f"Synthesised {name.replace('_', ' ')} for deterministic testing."
    if len(base) < minimum:
        base = base + " " + "detail " * ((minimum - len(base)) // 7 + 1)
    return base[:maximum] if len(base) > maximum else base


def json_responder(payload: dict[str, Any]) -> Callable[[LLMRequest], str]:
    """A responder that always returns one fixed JSON document."""
    encoded = json.dumps(payload)
    return lambda _request: encoded
