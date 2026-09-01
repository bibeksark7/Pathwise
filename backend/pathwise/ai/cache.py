"""Response caching.

Distinct from Anthropic's prompt caching, which discounts the *input* tokens of a
repeated prefix. This caches whole *responses*, so an identical request costs nothing
and returns instantly.

Only safe for deterministic, idempotent generation — parsing a learning goal, drafting
a roadmap for a given profile, explaining a decision trace. Never for the tutor: two
learners asking the same question have different learner state, and serving one the
other's answer would be both wrong and a privacy leak. The cache key covers the
rendered prompt, model, and effort, but *not* who asked, which is exactly why callers
must opt in per feature rather than getting caching by default.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Protocol

from pathwise.ai.providers.base import LLMResponse, TokenUsage
from pathwise.logging_config import get_logger

log = get_logger(__name__)

CACHE_PREFIX = "pathwise:llm:"


class ResponseCache(Protocol):
    """Storage for whole model responses."""

    async def get(self, key: str) -> LLMResponse | None: ...

    async def set(self, key: str, response: LLMResponse, *, ttl_seconds: int) -> None: ...


class NullResponseCache:
    """Caches nothing. The default, so caching is always a deliberate choice."""

    async def get(self, key: str) -> LLMResponse | None:
        return None

    async def set(self, key: str, response: LLMResponse, *, ttl_seconds: int) -> None:
        return None


class InMemoryResponseCache:
    """A process-local dict. For tests, and for single-process development.

    No eviction, so it is not suitable for a long-running server — which is what
    `RedisResponseCache` is for.
    """

    def __init__(self) -> None:
        self._entries: dict[str, LLMResponse] = {}
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> LLMResponse | None:
        hit = self._entries.get(key)
        if hit is None:
            self.misses += 1
        else:
            self.hits += 1
        return hit

    async def set(self, key: str, response: LLMResponse, *, ttl_seconds: int) -> None:
        self._entries[key] = response

    def clear(self) -> None:
        self._entries.clear()
        self.hits = self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class RedisResponseCache:
    """Redis-backed, shared across processes and workers."""

    def __init__(self, redis: object, *, prefix: str = CACHE_PREFIX) -> None:
        # Typed loosely so this module does not import redis at definition time;
        # the client is supplied by the composition root.
        self._redis = redis
        self._prefix = prefix

    async def get(self, key: str) -> LLMResponse | None:
        """Fetch a cached response.

        Any failure — Redis down, a stale entry whose shape no longer matches
        `LLMResponse` after a code change — is a miss, not an error. A cache that can
        break the application is worse than no cache.
        """
        try:
            raw = await self._redis.get(self._prefix + key)  # type: ignore[attr-defined]
        except Exception as exc:
            log.warning("llm_cache_read_failed", error=str(exc))
            return None

        if raw is None:
            return None

        try:
            payload = json.loads(raw)
            return LLMResponse(
                text=payload["text"],
                model=payload["model"],
                usage=TokenUsage(**payload["usage"]),
                stop_reason=payload.get("stop_reason", "end_turn"),
                latency_ms=payload.get("latency_ms", 0.0),
                provider_request_id=payload.get("provider_request_id"),
            )
        except Exception as exc:
            log.warning("llm_cache_entry_unreadable", error=str(exc))
            return None

    async def set(self, key: str, response: LLMResponse, *, ttl_seconds: int) -> None:
        """Store a response. Never raises — a failed write is just a future miss."""
        try:
            payload = json.dumps(
                {
                    "text": response.text,
                    "model": response.model,
                    "usage": asdict(response.usage),
                    "stop_reason": response.stop_reason,
                    "latency_ms": response.latency_ms,
                    "provider_request_id": response.provider_request_id,
                }
            )
            await self._redis.setex(self._prefix + key, ttl_seconds, payload)  # type: ignore[attr-defined]
        except Exception as exc:
            log.warning("llm_cache_write_failed", error=str(exc))


def cacheable(response: LLMResponse) -> bool:
    """Whether a response is worth storing.

    Refusals and truncated generations are excluded: caching a refusal would make a
    transient safety decision permanent for every future caller, and caching a
    truncated document would serve broken JSON forever.
    """
    return not response.was_refused and not response.was_truncated and bool(response.text)
