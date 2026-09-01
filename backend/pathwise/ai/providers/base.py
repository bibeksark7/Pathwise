"""The LLM provider interface.

Everything above this layer — roadmap generation, tutoring, grading, explanation —
talks to `LLMProvider` and never to a vendor SDK. That buys three things:

* **A deterministic test double.** `FakeProvider` implements the same interface, so
  the entire suite and all of CI run offline, for free, with reproducible output.
* **Provider substitution.** Anthropic today; the OpenAI adapter is a file, not a
  rewrite.
* **One place to enforce policy.** Token accounting, cost, latency, refusal handling,
  and validation all wrap this single surface, so nothing can bypass them.

Note what this interface deliberately does *not* expose: **sampling parameters**.
`temperature`, `top_p`, and `top_k` are rejected outright by current Claude models,
and offering knobs that raise a 400 would be worse than not offering them. Output
variability is controlled by `effort` instead.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class Effort(StrEnum):
    """How much thinking to spend on a request.

    The first real cost lever, and the one to reach for before considering a cheaper
    model: lower effort on a strong model often beats higher effort on a weaker one,
    and staying on one model keeps a single prompt-cache namespace.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    """One conversational turn."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """What a call consumed.

    Cached tokens are tracked separately from fresh ones because they are billed at
    very different rates — a read is roughly a tenth of the input price, a write
    roughly 1.25x — so collapsing them would misreport spend in both directions.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def cache_hit(self) -> bool:
        """Whether any of the prompt was served from cache.

        Worth asserting in tests: a silent cache invalidator (a timestamp in the
        system prompt, an unsorted dict) shows up here as a permanent zero.
        """
        return self.cache_read_tokens > 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Accumulate usage across a retry or a multi-turn tool loop."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the model may call.

    ``strict`` turns on schema enforcement so arguments are guaranteed to validate,
    which removes a whole class of defensive parsing at the call site. It requires
    ``additionalProperties: false`` and a complete ``required`` list in the schema.
    """

    name: str
    description: str
    input_schema: dict[str, object]
    strict: bool = True


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Everything needed to make one model call.

    Immutable, and hashable via :meth:`cache_key`, so an identical request can be
    served from our own response cache without re-deriving what "identical" means at
    each call site.
    """

    messages: tuple[Message, ...]
    system: str = ""
    max_tokens: int = 16_000
    effort: Effort = Effort.HIGH
    thinking: bool = True
    #: Cache the system prefix. Worth it whenever the system prompt is large and
    #: stable — the learner-state block for the tutor, the concept catalogue for
    #: roadmap generation.
    cache_system: bool = True
    #: Overrides the provider default. Used to route cheap classification work to a
    #: smaller model without threading configuration through every caller.
    model: str | None = None
    tools: tuple[ToolSpec, ...] = ()

    def cache_key(self) -> str:
        """A stable digest of everything that affects the response.

        Deliberately includes the model and effort: the same prompt at a different
        effort is a different request, and serving a cached low-effort answer to a
        high-effort call would silently degrade quality.
        """
        parts = [
            self.model or "",
            self.system,
            str(self.max_tokens),
            str(self.effort),
            str(self.thinking),
            *(f"{m.role}:{m.content}" for m in self.messages),
            *(t.name for t in self.tools),
        ]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]


StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """The result of one model call."""

    text: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    stop_reason: str = "end_turn"
    latency_ms: float = 0.0
    provider_request_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    #: Populated only on a refusal; names the safety category.
    refusal_category: str | None = None

    @property
    def was_refused(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def was_truncated(self) -> bool:
        """Output hit the token ceiling mid-thought.

        Important to surface rather than swallow: truncated JSON fails validation
        with a confusing parse error, when the real fix is a larger `max_tokens`.
        """
        return self.stop_reason == "max_tokens"


@dataclass(frozen=True, slots=True)
class StructuredResponse(Generic[SchemaT]):
    """A validated, typed model output alongside the raw call it came from.

    Keeping the raw response attached is what makes the call auditable — cost,
    latency, and stop reason stay available after parsing.
    """

    value: SchemaT
    raw: LLMResponse

    @property
    def usage(self) -> TokenUsage:
        return self.raw.usage


class LLMProvider(ABC):
    """A language model backend.

    Implementations must be safe to share across concurrent requests and must not
    raise vendor-specific exceptions past this boundary — everything becomes an
    ``AIProviderError``, ``AIRefusalError``, or ``AIValidationError`` so callers can
    handle failures without importing an SDK.
    """

    #: Short identifier recorded on every logged call.
    name: ClassVar[str] = "base"

    @property
    @abstractmethod
    def default_model(self) -> str:
        """The model used when a request does not name one."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate free-form text."""

    @abstractmethod
    async def complete_structured(
        self, request: LLMRequest, schema: type[SchemaT]
    ) -> StructuredResponse[SchemaT]:
        """Generate output conforming to a Pydantic schema.

        Implementations must use the provider's native structured-output support
        rather than asking for JSON in the prompt and parsing hopefully. The schema
        is a constraint on generation, not a suggestion.
        """

    @abstractmethod
    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Yield text deltas as they arrive. Used by the tutor."""

    @abstractmethod
    async def count_tokens(self, request: LLMRequest) -> int:
        """Count input tokens without generating.

        Used to guard against oversized context before spending on a call, and to
        estimate cost up front. Always the provider's own tokeniser — never
        `tiktoken` or a heuristic, which disagree with billing.
        """

    def resolve_model(self, request: LLMRequest) -> str:
        return request.model or self.default_model


def user(content: str) -> Message:
    """Shorthand for a user turn."""
    return Message(role=Role.USER, content=content)


def assistant(content: str) -> Message:
    """Shorthand for an assistant turn."""
    return Message(role=Role.ASSISTANT, content=content)


def conversation(*contents: str) -> tuple[Message, ...]:
    """Build an alternating user/assistant sequence, starting with the user."""
    return tuple(
        Message(role=Role.USER if index % 2 == 0 else Role.ASSISTANT, content=content)
        for index, content in enumerate(contents)
    )


def single(prompt: str) -> tuple[Message, ...]:
    """A one-turn conversation — the shape most Pathwise calls take."""
    return (user(prompt),)


__all__ = [
    "Effort",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "Role",
    "SchemaT",
    "StopReason",
    "StructuredResponse",
    "TokenUsage",
    "ToolCall",
    "ToolSpec",
    "assistant",
    "conversation",
    "single",
    "user",
]
