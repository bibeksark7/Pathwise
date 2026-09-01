"""Practical projects and their submissions.

Projects are the highest-weight evidence source in the mastery model: building a
working thing demonstrates transfer in a way that answering questions about it does
not. ``rubric`` is stored as data so grading is reproducible and so the evaluation
suite can check grader consistency against the same criteria.
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


class Project(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A practical build that exercises several concepts at once.

    Projects are global and reusable rather than per learner: a curated, evaluated
    project is worth more than a freshly invented one, and reuse lets difficulty
    calibration accumulate across everyone who has attempted it.
    """

    __tablename__ = "projects"
    __table_args__ = (
        CheckConstraint("difficulty BETWEEN 1 AND 5", name="difficulty_range"),
        CheckConstraint("expected_hours > 0", name="expected_hours_positive"),
        Index("ix_projects_domain_difficulty", "domain", "difficulty"),
    )

    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    domain: Mapped[str] = mapped_column(String(60), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)

    # ["Reads a log file from stdin", "Groups by status code", ...]
    requirements: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    # Concepts this project is evidence *for* — the skills it actually tests.
    concept_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(PGUUID(as_uuid=True)), nullable=False, default=list
    )
    prerequisite_concept_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(PGUUID(as_uuid=True)), nullable=False, default=list
    )

    difficulty: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    expected_hours: Mapped[float] = mapped_column(Float, nullable=False, default=4.0)

    # [{"id": "c1", "criterion": "...", "weight": 0.3, "levels": {...}}, ...]
    rubric: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    extensions: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    starter_notes: Mapped[str | None] = mapped_column(Text)

    submissions: Mapped[list[ProjectSubmission]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Project {self.slug}>"


class ProjectSubmission(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One learner's attempt at one project."""

    __tablename__ = "project_submissions"
    __table_args__ = (
        CheckConstraint("score IS NULL OR score BETWEEN 0 AND 1", name="score_range"),
        Index("ix_project_submissions_user_id_project_id", "user_id", "project_id"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    roadmap_node_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("roadmap_nodes.id", ondelete="SET NULL")
    )

    repository_url: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reflection: Mapped[str | None] = mapped_column(Text)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="submitted")
    score: Mapped[float | None] = mapped_column(Float)
    # {criterion_id: score} against the project's stored rubric.
    criterion_scores: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False, default=dict)
    concept_scores: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False, default=dict)
    feedback: Mapped[str | None] = mapped_column(Text)

    hours_spent: Mapped[float | None] = mapped_column(Float)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    graded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    project: Mapped[Project] = relationship(back_populates="submissions")
