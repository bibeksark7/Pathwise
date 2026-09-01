"""Recording every model call.

A cost figure is only trustworthy if nothing can bypass the recorder, and a
validation-failure rate is only meaningful if the denominator includes the calls that
failed. So every call is recorded — successes, refusals, provider errors, validation
failures, and cache hits alike.

Recording is deliberately behind an interface rather than wired straight to the
database. The AI client is then testable without Postgres, a worker can record
through the same path as a request, and a recorder outage degrades to a log line
instead of failing the user's request.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from pathwise.ai.cost import estimate_cost
from pathwise.ai.providers.base import TokenUsage
from pathwise.logging_config import get_logger
from pathwise.models.enums import LLMCallStatus
from pathwise.models.observability import LLMCall

log = get_logger(__name__)


@dataclass(slots=True)
class CallRecord:
    """Everything worth knowing about one model call.

    ``prompt_hash`` alongside ``prompt_version`` catches what a version number alone
    misses: a prompt edited in place without a version bump. Two records claiming the
    same version with different hashes means the text moved underneath a live version.
    """

    feature: str
    provider: str
    model: str
    prompt_name: str
    prompt_version: str
    prompt_hash: str
    status: LLMCallStatus
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    user_id: uuid.UUID | None = None
    validation_passed: bool = True
    validation_errors: Sequence[str] = ()
    repair_attempts: int = 0
    stop_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    provider_request_id: str | None = None

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.usage)


class CallRecorder(Protocol):
    """Somewhere to put call records."""

    async def record(self, record: CallRecord) -> None: ...


class NullCallRecorder:
    """Discards records. For scripts and one-off CLI work."""

    async def record(self, record: CallRecord) -> None:
        return None


class InMemoryCallRecorder:
    """Collects records in a list, for assertions in tests."""

    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    async def record(self, record: CallRecord) -> None:
        self.records.append(record)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.usage.total_tokens for r in self.records)

    def by_feature(self, feature: str) -> list[CallRecord]:
        return [r for r in self.records if r.feature == feature]

    def clear(self) -> None:
        self.records.clear()


class DatabaseCallRecorder:
    """Writes records to the ``llm_calls`` table.

    A failure to record is logged and swallowed. Observability must never take down
    the thing it observes: losing one audit row is a far smaller problem than failing
    a learner's request because the audit write hit a constraint.
    """

    def __init__(self, session: AsyncSession, *, autoflush: bool = True) -> None:
        self._session = session
        self._autoflush = autoflush

    async def record(self, record: CallRecord) -> None:
        try:
            row = LLMCall(
                user_id=record.user_id,
                feature=record.feature,
                provider=record.provider,
                model=record.model,
                prompt_name=record.prompt_name,
                prompt_version=record.prompt_version,
                prompt_hash=record.prompt_hash,
                input_tokens=record.usage.input_tokens,
                output_tokens=record.usage.output_tokens,
                cache_read_tokens=record.usage.cache_read_tokens,
                cache_write_tokens=record.usage.cache_write_tokens,
                cost_usd=record.cost_usd,
                latency_ms=record.latency_ms,
                status=record.status,
                validation_passed=record.validation_passed,
                validation_errors=list(record.validation_errors),
                repair_attempts=record.repair_attempts,
                stop_reason=record.stop_reason,
                error_type=record.error_type,
                # Truncated: an error message can be an entire API response body, and
                # a runaway one would bloat every row of the audit table.
                error_message=(record.error_message or None) and record.error_message[:2000],
                provider_request_id=record.provider_request_id,
            )
            self._session.add(row)
            if self._autoflush:
                await self._session.flush()
        except Exception as exc:
            log.error(
                "llm_call_record_failed",
                feature=record.feature,
                model=record.model,
                error=str(exc),
            )
