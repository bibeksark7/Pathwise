"""AI observability: every model call, and every evaluation run.

``LLMCall`` is written for *every* call, including failures, refusals, and cache hits.
That completeness is the point — a validation-failure rate is only meaningful if the
denominator includes the calls that failed, and a cost figure is only trustworthy if
nothing bypasses the recorder.

``EvalRun`` / ``EvalResult`` store scored evaluation runs so a prompt or model change
can be compared against a stored baseline rather than judged by impression.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pathwise.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from pathwise.models.enums import LLMCallStatus
from pathwise.models.types import pg_enum


class LLMCall(UUIDPrimaryKeyMixin, Base):
    """One request to a language model, with its cost and validation outcome.

    ``prompt_name`` + ``prompt_version`` + ``prompt_hash`` together identify exactly
    which prompt text produced this output. The hash catches the case that version
    numbers miss: a prompt edited without bumping its version.
    """

    __tablename__ = "llm_calls"
    __table_args__ = (
        CheckConstraint("cost_usd >= 0", name="cost_non_negative"),
        Index("ix_llm_calls_feature_created_at", "feature", "created_at"),
        Index("ix_llm_calls_user_id_created_at", "user_id", "created_at"),
        Index("ix_llm_calls_prompt_name_prompt_version", "prompt_name", "prompt_version"),
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    # The application capability this call served: "roadmap_generate", "tutor_reply",
    # "grade_short_answer", ... Cost and latency are reported per feature.
    feature: Mapped[str] = mapped_column(String(60), nullable=False)

    provider: Mapped[str] = mapped_column(String(30), nullable=False)
    model: Mapped[str] = mapped_column(String(80), nullable=False)
    prompt_name: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(16), nullable=False)

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    status: Mapped[LLMCallStatus] = mapped_column(
        pg_enum(LLMCallStatus, "llm_call_status"), nullable=False
    )
    # False when the response parsed but failed a domain rule (unknown concept slug,
    # cyclic edge, invented URL). Tracked separately from provider errors because it
    # measures prompt quality, not infrastructure.
    validation_passed: Mapped[bool] = mapped_column(nullable=False, default=True)
    validation_errors: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    repair_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    stop_reason: Mapped[str | None] = mapped_column(String(30))
    error_type: Mapped[str | None] = mapped_column(String(60))
    error_message: Mapped[str | None] = mapped_column(Text)
    provider_request_id: Mapped[str | None] = mapped_column(String(80))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    def __repr__(self) -> str:
        return f"<LLMCall {self.feature} {self.model} ${self.cost_usd:.4f}>"


class EvalRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One execution of one evaluation suite against one prompt/model combination."""

    __tablename__ = "eval_runs"
    __table_args__ = (Index("ix_eval_runs_suite_created_at", "suite", "created_at"),)

    suite: Mapped[str] = mapped_column(String(60), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(20), nullable=False, default="v1")
    provider: Mapped[str] = mapped_column(String(30), nullable=False)
    model: Mapped[str] = mapped_column(String(80), nullable=False)
    prompt_name: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(20))
    git_sha: Mapped[str | None] = mapped_column(String(40))

    case_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    passed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # {scorer_name: mean_score} — the numbers a regression gate compares.
    aggregate_scores: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False, default=dict)
    total_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    mean_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Set when this run is the accepted reference for future comparisons.
    is_baseline: Mapped[bool] = mapped_column(nullable=False, default=False)
    notes: Mapped[str | None] = mapped_column(Text)

    results: Mapped[list[EvalResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<EvalRun {self.suite} {self.passed_count}/{self.case_count}>"


class EvalResult(UUIDPrimaryKeyMixin, Base):
    """The outcome of one evaluation case.

    Inputs and outputs are stored verbatim so a regression can be read directly —
    the point of an eval failure is seeing *what* the model said, not only that a
    number dropped.
    """

    __tablename__ = "eval_results"
    __table_args__ = (Index("ix_eval_results_run_id_case_id", "run_id", "case_id"),)

    run_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False
    )
    case_id: Mapped[str] = mapped_column(String(80), nullable=False)

    input_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    expected: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    actual: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # {scorer_name: score} for this case.
    scores: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False, default=dict)
    passed: Mapped[bool] = mapped_column(nullable=False, default=False)
    failure_reason: Mapped[str | None] = mapped_column(Text)

    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    run: Mapped[EvalRun] = relationship(back_populates="results")
