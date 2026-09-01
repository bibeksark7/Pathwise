"""The Anthropic provider.

Wraps the official SDK and translates it into Pathwise's own vocabulary, so nothing
above this file imports `anthropic`.

Four API details are load-bearing and easy to get wrong:

* **Structured output uses ``messages.parse(output_format=Model)``**, which constrains
  generation to the schema. Asking for JSON in the prompt and parsing hopefully is a
  different, much worse thing — it fails silently and intermittently.
* **Adaptive thinking, not a token budget.** ``budget_tokens`` returns a 400 on
  current models; depth is controlled by ``output_config.effort``.
* **No sampling parameters.** ``temperature``/``top_p``/``top_k`` are rejected.
* **A refusal is an HTTP 200**, not an exception. ``stop_reason`` must be checked
  before reading content, or a refusal reads as an empty successful response.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import anthropic

from pathwise.ai.providers.base import (
    Effort,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    SchemaT,
    StructuredResponse,
    TokenUsage,
    ToolCall,
)
from pathwise.api.errors import AIProviderError, AIRefusalError, AIValidationError
from pathwise.config import Settings
from pathwise.logging_config import get_logger

log = get_logger(__name__)

#: Efforts at which thinking may not be disabled — the API returns a 400.
_NO_DISABLE_EFFORTS = frozenset({Effort.XHIGH, Effort.MAX})


class AnthropicProvider(LLMProvider):
    """Claude, via the official SDK."""

    name: ClassVar[str] = "anthropic"

    def __init__(self, settings: Settings, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._settings = settings
        # The SDK retries connection errors, 408, 409, 429 and 5xx with exponential
        # backoff on its own. Our retry logic sits a layer up and handles a different
        # failure — output that parsed but failed a domain rule — so the two do not
        # overlap or compound.
        self._client = client or anthropic.AsyncAnthropic(
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    @property
    def default_model(self) -> str:
        return self._settings.llm_model

    # --- request construction --------------------------------------------------- #

    def _build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        """Translate an `LLMRequest` into SDK arguments."""
        kwargs: dict[str, Any] = {
            "model": self.resolve_model(request),
            "max_tokens": request.max_tokens,
            "messages": [{"role": str(m.role), "content": m.content} for m in request.messages],
            "output_config": {"effort": str(request.effort)},
        }

        if request.system:
            kwargs["system"] = self._system_blocks(request)

        kwargs["thinking"] = self._thinking_config(request, kwargs)

        if request.tools:
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "strict": tool.strict,
                }
                for tool in request.tools
            ]

        return kwargs

    def _system_blocks(self, request: LLMRequest) -> list[dict[str, Any]] | str:
        """The system prompt, with a cache breakpoint when caching is requested.

        Caching is a *prefix* match, so anything volatile — a timestamp, a request
        id, an unsorted dict — placed before this breakpoint silently invalidates it
        on every call. There is also a model-dependent minimum prefix length
        (512-4096 tokens) below which nothing is cached and no error is raised, which
        is why `usage.cache_read_input_tokens` is worth asserting on in tests.
        """
        if not request.cache_system:
            return request.system
        return [
            {
                "type": "text",
                "text": request.system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _thinking_config(self, request: LLMRequest, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Adaptive thinking, or an explicit disable where that is permitted.

        Disabling thinking is discouraged and only offered for latency-critical
        paths: with it off, the model occasionally writes a tool call into visible
        text instead of emitting a `tool_use` block — the turn succeeds, the call
        never runs, and no error is raised. Lowering `effort` is nearly always the
        better lever, so that is what happens if the combination is invalid.
        """
        if request.thinking:
            return {"type": "adaptive"}

        if request.effort in _NO_DISABLE_EFFORTS:
            # `disabled` + xhigh/max is a 400. Step effort down rather than fail.
            log.warning(
                "thinking_disable_downgraded",
                effort=str(request.effort),
                reason="disabled thinking is rejected above effort=high",
            )
            kwargs["output_config"] = {"effort": str(Effort.HIGH)}

        return {"type": "disabled"}

    # --- the provider interface -------------------------------------------------- #

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = await self._client.messages.create(**self._build_kwargs(request))
        except Exception as exc:
            raise self._translate(exc) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        return self._to_response(response, elapsed_ms)

    async def complete_structured(
        self, request: LLMRequest, schema: type[SchemaT]
    ) -> StructuredResponse[SchemaT]:
        """Generate output constrained to ``schema``.

        `output_format` makes the schema a constraint on generation, so a
        well-formed response is guaranteed to parse. What it does *not* guarantee is
        that the content is *true* — a schema-valid roadmap can still cite a concept
        that does not exist. That is what the domain validators are for.
        """
        started = time.perf_counter()
        try:
            response = await self._client.messages.parse(
                **self._build_kwargs(request), output_format=schema
            )
        except Exception as exc:
            raise self._translate(exc) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        raw = self._to_response(response, elapsed_ms)

        if raw.was_refused:
            raise AIRefusalError("The model declined this request.", category=raw.refusal_category)

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            # Reachable when generation is cut off mid-document: the constraint holds
            # for a complete response, not a truncated one.
            raise AIValidationError(
                "Structured output was empty or could not be parsed.",
                schema=schema.__name__,
                stop_reason=raw.stop_reason,
                truncated=raw.was_truncated,
            )

        return StructuredResponse(value=parsed, raw=raw)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream text deltas. Used by the tutor so answers appear as they form."""
        try:
            async with self._client.messages.stream(**self._build_kwargs(request)) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as exc:
            raise self._translate(exc) from exc

    async def count_tokens(self, request: LLMRequest) -> int:
        """Count input tokens with the provider's own tokeniser.

        Never `tiktoken` or a words-times-1.3 heuristic: both disagree with what is
        actually billed, and the disagreement grows with exactly the long contexts
        where the number matters.
        """
        kwargs = self._build_kwargs(request)
        try:
            result = await self._client.messages.count_tokens(
                model=kwargs["model"],
                messages=kwargs["messages"],
                **({"system": kwargs["system"]} if "system" in kwargs else {}),
            )
        except Exception as exc:
            raise self._translate(exc) from exc
        return int(result.input_tokens)

    # --- translation ------------------------------------------------------------- #

    def _to_response(self, response: Any, elapsed_ms: float) -> LLMResponse:
        """Convert an SDK message into an `LLMResponse`."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        usage = getattr(response, "usage", None)
        stop_reason = getattr(response, "stop_reason", "end_turn") or "end_turn"

        # `stop_details` is populated only on a refusal and is None otherwise, so it
        # must be guarded before reading.
        refusal_category: str | None = None
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            refusal_category = getattr(details, "category", None)

        return LLMResponse(
            text="".join(text_parts),
            model=getattr(response, "model", self.default_model),
            usage=TokenUsage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            ),
            stop_reason=stop_reason,
            latency_ms=elapsed_ms,
            provider_request_id=getattr(response, "_request_id", None),
            tool_calls=tuple(tool_calls),
            refusal_category=refusal_category,
        )

    def _translate(self, exc: Exception) -> Exception:
        """Map SDK exceptions onto Pathwise's own hierarchy.

        Handled most-specific-first. Collapsing these into one broad `except` would
        lose the distinction that matters operationally: a 429 or 5xx is worth
        retrying, a 400 or 404 never is.
        """
        if isinstance(exc, AIProviderError | AIRefusalError | AIValidationError):
            return exc

        if isinstance(exc, anthropic.AuthenticationError):
            return AIProviderError("The Anthropic API key is missing or invalid.", retryable=False)
        if isinstance(exc, anthropic.PermissionDeniedError):
            return AIProviderError("The API key lacks permission for this call.", retryable=False)
        if isinstance(exc, anthropic.NotFoundError):
            return AIProviderError(
                f"Unknown model or endpoint: {exc}", retryable=False, model=self.default_model
            )
        if isinstance(exc, anthropic.BadRequestError):
            # Almost always our bug — a malformed schema, or a parameter the model
            # rejects. Never retry it; the same request will fail identically.
            return AIProviderError(f"Malformed request: {exc}", retryable=False)
        if isinstance(exc, anthropic.RateLimitError):
            retry_after = exc.response.headers.get("retry-after") if exc.response else None
            return AIProviderError(
                "Rate limited by the Anthropic API.",
                retryable=True,
                retry_after_seconds=retry_after,
            )
        if isinstance(exc, anthropic.APITimeoutError):
            return AIProviderError("The request to Anthropic timed out.", retryable=True)
        if isinstance(exc, anthropic.APIConnectionError):
            return AIProviderError("Could not reach the Anthropic API.", retryable=True)
        if isinstance(exc, anthropic.APIStatusError):
            return AIProviderError(
                f"Anthropic returned {exc.status_code}: {exc.message}",
                retryable=exc.status_code >= 500,
                status_code=exc.status_code,
            )

        return AIProviderError(f"Unexpected provider failure: {exc}", retryable=False)
