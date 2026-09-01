"""The AI client — the single door every model call goes through.

Services never touch a provider directly. They call `AIClient`, which:

1. renders a versioned prompt from the registry,
2. checks the response cache (opt-in per call),
3. calls the provider,
4. runs domain validators over the result,
5. on failure, sends the errors back for **one** repair attempt,
6. records the call — cost, tokens, latency, outcome — whatever happened,
7. returns a validated value, or raises so the caller can take its deterministic path.

Step 6 happens on every path, including exceptions. That is what makes the cost and
validation-failure numbers real rather than optimistic: nothing can quietly skip the
recorder.

Step 5 is capped at one attempt on purpose. A model that produced an invalid slug
usually fixes it when shown the error; one that fails twice will not succeed on the
third try, and each attempt costs money and latency. Beyond that, the answer is the
deterministic fallback — which is why every AI feature here has one.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import replace

from pathwise.ai.cache import NullResponseCache, ResponseCache, cacheable
from pathwise.ai.call_log import CallRecord, CallRecorder, NullCallRecorder
from pathwise.ai.prompts.registry import Prompt, PromptRegistry, get_registry
from pathwise.ai.providers.base import (
    Effort,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    Message,
    Role,
    SchemaT,
    user,
)
from pathwise.ai.validators import ValidationResult
from pathwise.api.errors import AIProviderError, AIRefusalError, AIValidationError
from pathwise.config import Settings
from pathwise.logging_config import get_logger
from pathwise.models.enums import LLMCallStatus

log = get_logger(__name__)

#: One repair attempt. See the module docstring for why not more.
MAX_REPAIR_ATTEMPTS = 1

#: Recorded for calls made without a registered prompt (ad-hoc internal use).
_UNVERSIONED = "unversioned"


class AIClient:
    """Orchestrates prompting, validation, repair, caching, and accounting."""

    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings,
        *,
        recorder: CallRecorder | None = None,
        cache: ResponseCache | None = None,
        registry: PromptRegistry | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._recorder = recorder or NullCallRecorder()
        # Caching is off unless supplied: it is unsafe for anything learner-specific,
        # so it must be an explicit decision rather than an inherited default.
        self._cache = cache or NullResponseCache()
        self._registry = registry or get_registry()

    @property
    def provider_name(self) -> str:
        return self._provider.name

    # --- free-form generation ---------------------------------------------------- #

    async def generate(
        self,
        *,
        feature: str,
        prompt_name: str,
        variables: dict[str, object] | None = None,
        system: str = "",
        effort: Effort = Effort.HIGH,
        max_tokens: int | None = None,
        user_id: uuid.UUID | None = None,
        use_cache: bool = False,
        prompt_version: str | None = None,
    ) -> LLMResponse:
        """Render a prompt, call the model, and return the text response."""
        prompt = self._registry.get(prompt_name, prompt_version)
        rendered = prompt.render(**(variables or {}))

        request = LLMRequest(
            messages=(user(rendered),),
            system=system,
            effort=effort,
            max_tokens=max_tokens or self._settings.llm_max_tokens,
        )

        if use_cache:
            cached = await self._cache.get(request.cache_key())
            if cached is not None:
                await self._record(
                    feature, prompt, cached, status=LLMCallStatus.CACHED, user_id=user_id
                )
                return cached

        response = await self._call(feature, prompt, request, user_id=user_id)

        if use_cache and cacheable(response):
            await self._cache.set(
                request.cache_key(), response, ttl_seconds=self._settings.llm_cache_ttl_seconds
            )
        return response

    # --- structured generation ---------------------------------------------------- #

    async def generate_structured(
        self,
        *,
        feature: str,
        prompt_name: str,
        schema: type[SchemaT],
        variables: dict[str, object] | None = None,
        system: str = "",
        effort: Effort = Effort.HIGH,
        max_tokens: int | None = None,
        user_id: uuid.UUID | None = None,
        validate: Callable[[SchemaT], ValidationResult] | None = None,
        prompt_version: str | None = None,
    ) -> SchemaT:
        """Generate a schema-conforming, domain-validated value.

        Raises:
            AIValidationError: if the output still fails domain validation after the
                repair attempt. Callers are expected to catch this and fall back to
                deterministic behaviour rather than surface it.
            AIRefusalError: if the model declined.
            AIProviderError: on a transport or configuration failure.
        """
        prompt = self._registry.get(prompt_name, prompt_version)
        rendered = prompt.render(**(variables or {}))

        request = LLMRequest(
            messages=(user(rendered),),
            system=system,
            effort=effort,
            max_tokens=max_tokens or self._settings.llm_max_tokens,
        )

        attempt = 0
        last_issues: list[str] = []

        while True:
            started = time.perf_counter()
            try:
                structured = await self._provider.complete_structured(request, schema)
            except (AIRefusalError, AIValidationError, AIProviderError) as exc:
                await self._record_failure(
                    feature, prompt, request, exc, attempt, time.perf_counter() - started, user_id
                )
                raise

            if validate is None:
                await self._record(
                    feature,
                    prompt,
                    structured.raw,
                    status=LLMCallStatus.SUCCESS,
                    user_id=user_id,
                    repair_attempts=attempt,
                )
                return structured.value

            result = validate(structured.value)
            if result.is_valid:
                await self._record(
                    feature,
                    prompt,
                    structured.raw,
                    status=LLMCallStatus.SUCCESS,
                    user_id=user_id,
                    repair_attempts=attempt,
                )
                return structured.value

            last_issues = result.summary()
            log.warning(
                "ai_output_invalid",
                feature=feature,
                prompt=prompt.name,
                version=prompt.version,
                attempt=attempt,
                issues=last_issues,
            )

            if attempt >= MAX_REPAIR_ATTEMPTS:
                await self._record(
                    feature,
                    prompt,
                    structured.raw,
                    status=LLMCallStatus.VALIDATION_FAILED,
                    user_id=user_id,
                    validation_passed=False,
                    validation_errors=last_issues,
                    repair_attempts=attempt,
                )
                raise AIValidationError(
                    "Model output failed domain validation after repair.",
                    feature=feature,
                    prompt=f"{prompt.name}@{prompt.version}",
                    issues=last_issues,
                )

            # Repair: replay the exchange with the model's own answer and the
            # specific corrections, so it edits rather than starts over.
            attempt += 1
            request = replace(
                request,
                messages=(
                    *request.messages,
                    Message(role=Role.ASSISTANT, content=structured.raw.text),
                    user(result.repair_instructions()),
                ),
            )

    # --- streaming ---------------------------------------------------------------- #

    async def stream(
        self,
        *,
        feature: str,
        prompt_name: str,
        variables: dict[str, object] | None = None,
        system: str = "",
        effort: Effort = Effort.HIGH,
        history: tuple[Message, ...] = (),
        user_id: uuid.UUID | None = None,
        prompt_version: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a response. Used by the tutor.

        Token usage is not available until a stream completes, so the recorded row
        carries latency and outcome but zero tokens. The alternative — buffering the
        whole response to count it — would defeat the point of streaming.
        """
        prompt = self._registry.get(prompt_name, prompt_version)
        rendered = prompt.render(**(variables or {}))

        request = LLMRequest(
            messages=(*history, user(rendered)),
            system=system,
            effort=effort,
            max_tokens=self._settings.llm_max_tokens,
        )

        started = time.perf_counter()
        status = LLMCallStatus.SUCCESS
        try:
            async for chunk in self._provider.stream(request):
                yield chunk
        except (AIRefusalError, AIProviderError):
            status = LLMCallStatus.PROVIDER_ERROR
            raise
        finally:
            await self._record(
                feature,
                prompt,
                LLMResponse(
                    text="",
                    model=self._provider.resolve_model(request),
                    latency_ms=(time.perf_counter() - started) * 1000,
                ),
                status=status,
                user_id=user_id,
            )

    async def count_tokens(self, text: str, *, system: str = "") -> int:
        """Input-token count for a prospective prompt."""
        return await self._provider.count_tokens(LLMRequest(messages=(user(text),), system=system))

    # --- internals ----------------------------------------------------------------- #

    async def _call(
        self,
        feature: str,
        prompt: Prompt,
        request: LLMRequest,
        *,
        user_id: uuid.UUID | None,
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = await self._provider.complete(request)
        except (AIRefusalError, AIProviderError) as exc:
            await self._record_failure(
                feature, prompt, request, exc, 0, time.perf_counter() - started, user_id
            )
            raise

        if response.was_refused:
            # A refusal arrives as a successful HTTP response, so it has to be turned
            # into an error here or it reads as an empty answer.
            await self._record(
                feature, prompt, response, status=LLMCallStatus.REFUSED, user_id=user_id
            )
            raise AIRefusalError(
                "The model declined this request.", category=response.refusal_category
            )

        await self._record(feature, prompt, response, status=LLMCallStatus.SUCCESS, user_id=user_id)
        return response

    async def _record(
        self,
        feature: str,
        prompt: Prompt,
        response: LLMResponse,
        *,
        status: LLMCallStatus,
        user_id: uuid.UUID | None,
        validation_passed: bool = True,
        validation_errors: list[str] | None = None,
        repair_attempts: int = 0,
    ) -> None:
        await self._recorder.record(
            CallRecord(
                feature=feature,
                provider=self._provider.name,
                model=response.model,
                prompt_name=prompt.name,
                prompt_version=prompt.version,
                prompt_hash=prompt.checksum,
                status=status,
                usage=response.usage,
                latency_ms=response.latency_ms,
                user_id=user_id,
                validation_passed=validation_passed,
                validation_errors=validation_errors or [],
                repair_attempts=repair_attempts,
                stop_reason=response.stop_reason,
                provider_request_id=response.provider_request_id,
            )
        )

    async def _record_failure(
        self,
        feature: str,
        prompt: Prompt,
        request: LLMRequest,
        exc: Exception,
        attempt: int,
        elapsed_seconds: float,
        user_id: uuid.UUID | None,
    ) -> None:
        """Record a call that raised.

        Failures are the calls most worth having in the table: they are what the
        validation-failure rate and the provider-error rate are computed from.
        """
        status = (
            LLMCallStatus.REFUSED
            if isinstance(exc, AIRefusalError)
            else LLMCallStatus.VALIDATION_FAILED
            if isinstance(exc, AIValidationError)
            else LLMCallStatus.PROVIDER_ERROR
        )
        await self._recorder.record(
            CallRecord(
                feature=feature,
                provider=self._provider.name,
                model=self._provider.resolve_model(request),
                prompt_name=prompt.name,
                prompt_version=prompt.version,
                prompt_hash=prompt.checksum,
                status=status,
                latency_ms=elapsed_seconds * 1000,
                user_id=user_id,
                validation_passed=status is not LLMCallStatus.VALIDATION_FAILED,
                repair_attempts=attempt,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        )


def build_provider(settings: Settings) -> LLMProvider:
    """Construct the configured provider.

    The one place a provider is chosen. Switching to the deterministic fake for a
    whole environment is a single environment variable, which is how the test suite
    and CI run offline.
    """
    if settings.llm_provider == "fake":
        from pathwise.ai.providers.fake_provider import FakeProvider

        return FakeProvider()

    if settings.llm_provider == "anthropic":
        from pathwise.ai.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)

    if settings.llm_provider == "openai":
        raise AIProviderError(
            "The OpenAI provider is not implemented yet. "
            "Set PATHWISE_LLM_PROVIDER to 'anthropic' or 'fake'.",
            retryable=False,
        )

    raise AIProviderError(f"Unknown LLM provider: {settings.llm_provider}", retryable=False)
