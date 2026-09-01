"""Assessments: how Pathwise finds out what a learner actually understands.

The design point that makes assessments useful to the adaptive engine is that a
question is bound to **concepts and learning objectives**, not just to a topic. A
score of 48% is not evidence; "missed both questions targeting the chain-rule
objective, passed the intuition objectives" is. ``Answer.objective_scores`` is what
turns a submission into per-objective evidence, and per-objective evidence is what
blame attribution needs to name a prerequisite.
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
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pathwise.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from pathwise.models.enums import AttemptStatus, QuestionType
from pathwise.models.types import pg_enum


class Assessment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A generated set of questions targeting specific concepts.

    ``purpose`` distinguishes the diagnostic that seeds the initial knowledge estimate
    from a checkpoint after a topic and from targeted remediation practice — they
    carry different evidence weights and different pass thresholds.
    """

    __tablename__ = "assessments"
    __table_args__ = (Index("ix_assessments_user_id_created_at", "user_id", "created_at"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    purpose: Mapped[str] = mapped_column(String(30), nullable=False, default="checkpoint")
    concept_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(PGUUID(as_uuid=True)), nullable=False, default=list
    )
    roadmap_node_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("roadmap_nodes.id", ondelete="SET NULL")
    )
    estimated_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=15)

    # Provenance for the evaluation framework.
    prompt_name: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(20))

    questions: Mapped[list[Question]] = relationship(
        back_populates="assessment",
        cascade="all, delete-orphan",
        order_by="Question.order_index",
    )
    attempts: Mapped[list[Attempt]] = relationship(
        back_populates="assessment", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Assessment {self.title!r} ({self.purpose})>"


class Question(UUIDPrimaryKeyMixin, Base):
    """One question, bound to the concepts and objectives it measures.

    ``answer_key`` holds the correct option id for multiple choice, or the reference
    answer and rubric for open responses. It is never sent to the client — the API
    schema for a question deliberately omits it.
    """

    __tablename__ = "questions"
    __table_args__ = (
        CheckConstraint("difficulty BETWEEN 1 AND 5", name="difficulty_range"),
        CheckConstraint("points > 0", name="points_positive"),
        Index("ix_questions_assessment_id_order_index", "assessment_id", "order_index"),
    )

    assessment_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False
    )
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    question_type: Mapped[QuestionType] = mapped_column(
        pg_enum(QuestionType, "question_type"), nullable=False
    )
    stem: Mapped[str] = mapped_column(Text, nullable=False)

    # [{"id": "a", "text": "..."}, ...] — empty for open-response types.
    options: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    answer_key: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    explanation: Mapped[str | None] = mapped_column(Text)

    concept_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(PGUUID(as_uuid=True)), nullable=False, default=list
    )
    objective_ids: Mapped[list[str]] = mapped_column(
        ARRAY(String(40)), nullable=False, default=list
    )
    difficulty: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    points: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    assessment: Mapped[Assessment] = relationship(back_populates="questions")

    def __repr__(self) -> str:
        return f"<Question {self.question_type} #{self.order_index}>"


class Attempt(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One learner's run at one assessment."""

    __tablename__ = "attempts"
    __table_args__ = (
        CheckConstraint("score IS NULL OR score BETWEEN 0 AND 1", name="score_range"),
        Index("ix_attempts_user_id_assessment_id", "user_id", "assessment_id"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[AttemptStatus] = mapped_column(
        pg_enum(AttemptStatus, "attempt_status"), nullable=False, default=AttemptStatus.IN_PROGRESS
    )
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    score: Mapped[float | None] = mapped_column(Float)
    # {concept_id: score} and {objective_id: score} — the per-target breakdown that
    # becomes evidence. A single overall score is not enough to update the model.
    concept_scores: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False, default=dict)
    objective_scores: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False, default=dict)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    graded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[int | None] = mapped_column(Integer)

    assessment: Mapped[Assessment] = relationship(back_populates="attempts")
    answers: Mapped[list[Answer]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Attempt {self.status} score={self.score}>"


class Answer(UUIDPrimaryKeyMixin, Base):
    """A learner's response to one question, with its grading.

    ``grader`` records how the score was produced — ``deterministic`` for multiple
    choice and executed code, ``llm_rubric`` for open responses. Only the latter is
    subject to grader drift, so the evaluation suite can measure it separately.
    """

    __tablename__ = "answers"
    __table_args__ = (
        CheckConstraint("score IS NULL OR score BETWEEN 0 AND 1", name="score_range"),
        Index("ix_answers_attempt_id_question_id", "attempt_id", "question_id"),
    )

    attempt_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("attempts.id", ondelete="CASCADE"), nullable=False
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("questions.id", ondelete="CASCADE"), nullable=False
    )
    response: Mapped[str] = mapped_column(Text, nullable=False, default="")
    score: Mapped[float | None] = mapped_column(Float)
    grader: Mapped[str | None] = mapped_column(String(30))

    objective_scores: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False, default=dict)
    # [{"objective_id": "...", "description": "...", "evidence": "<quoted response>"}]
    # Quoting the learner's own words keeps a misconception claim checkable.
    misconceptions: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    feedback: Mapped[str | None] = mapped_column(Text)
    time_spent_seconds: Mapped[int | None] = mapped_column(Integer)

    attempt: Mapped[Attempt] = relationship(back_populates="answers")
    question: Mapped[Question] = relationship()
