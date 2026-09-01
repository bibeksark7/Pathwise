"""The knowledge graph and the per-learner knowledge state.

Two halves that must not be confused:

* ``Concept`` / ``ConceptEdge`` are **global** — one curated graph shared by everyone.
* ``MasteryState`` / ``EvidenceEvent`` are **per user** — what this learner has shown.

``EvidenceEvent`` is an append-only log. ``MasteryState`` is a materialised view of it,
kept for query speed; it can always be rebuilt by replaying the log, which is what
makes a change to the mastery algorithm safe to deploy.
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
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pathwise.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from pathwise.models.enums import ConceptSource, ConceptStatus, EvidenceSource, RelationType
from pathwise.models.types import embedding_column, pg_enum


class Concept(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A unit of knowledge — the node type of the graph.

    Granularity target: something a learner could plausibly study in one to six hours
    and be assessed on. "Linear algebra" is too coarse to blame for a failure;
    "matrix multiplication" is the right size.
    """

    __tablename__ = "concepts"
    __table_args__ = (
        CheckConstraint("difficulty BETWEEN 1 AND 5", name="difficulty_range"),
        CheckConstraint("estimated_minutes > 0", name="estimated_minutes_positive"),
        Index("ix_concepts_domain_status", "domain", "status"),
    )

    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    domain: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    difficulty: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    estimated_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=120)

    # [{"id": "lo-1", "text": "...", "bloom": "apply"}, ...]
    # Assessment questions bind to these ids, which is how a score becomes evidence
    # about a specific objective rather than a vague number about a whole topic.
    learning_objectives: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    tags: Mapped[list[str]] = mapped_column(ARRAY(String(50)), nullable=False, default=list)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(String(120)), nullable=False, default=list)

    source: Mapped[ConceptSource] = mapped_column(
        pg_enum(ConceptSource, "concept_source"), nullable=False, default=ConceptSource.SEED
    )
    status: Mapped[ConceptStatus] = mapped_column(
        pg_enum(ConceptStatus, "concept_status"), nullable=False, default=ConceptStatus.APPROVED
    )
    embedding: Mapped[list[float] | None] = mapped_column(embedding_column())

    outgoing_edges: Mapped[list[ConceptEdge]] = relationship(
        back_populates="source_concept",
        foreign_keys="ConceptEdge.source_id",
        cascade="all, delete-orphan",
    )
    incoming_edges: Mapped[list[ConceptEdge]] = relationship(
        back_populates="target_concept",
        foreign_keys="ConceptEdge.target_id",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Concept {self.slug}>"


class ConceptEdge(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A typed, weighted relationship between two concepts.

    ``strength`` is how hard the requirement is: 1.0 means you genuinely cannot
    proceed without it, 0.5 means it helps. The decision engine uses it as the
    prerequisite threshold multiplier, and blame attribution weights candidates by it,
    so a weak edge produces a weak accusation rather than a false one.
    """

    __tablename__ = "concept_edges"
    __table_args__ = (
        UniqueConstraint("source_id", "target_id", "relation", name="uq_edge"),
        CheckConstraint("source_id <> target_id", name="no_self_loops"),
        CheckConstraint("strength BETWEEN 0 AND 1", name="strength_range"),
        Index("ix_concept_edges_target_relation", "target_id", "relation"),
        Index("ix_concept_edges_source_relation", "source_id", "relation"),
    )

    source_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    relation: Mapped[RelationType] = mapped_column(
        pg_enum(RelationType, "relation_type"), nullable=False
    )
    strength: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    source: Mapped[ConceptSource] = mapped_column(
        pg_enum(ConceptSource, "concept_source"), nullable=False, default=ConceptSource.SEED
    )
    rationale: Mapped[str | None] = mapped_column(Text)

    source_concept: Mapped[Concept] = relationship(
        back_populates="outgoing_edges", foreign_keys=[source_id]
    )
    target_concept: Mapped[Concept] = relationship(
        back_populates="incoming_edges", foreign_keys=[target_id]
    )

    def __repr__(self) -> str:
        return f"<ConceptEdge {self.source_id} -{self.relation}-> {self.target_id}>"


class MasteryState(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """The learner's estimated command of one concept.

    Mastery is a Beta(alpha, beta) posterior, not a bare percentage. Two learners can
    both sit at 0.80 while one has answered a single question and the other has been
    assessed twelve times; ``confidence`` is what separates them, and the decision
    engine refuses to skip material on a high-mastery, low-confidence estimate.
    """

    __tablename__ = "mastery_states"
    __table_args__ = (
        UniqueConstraint("user_id", "concept_id", name="uq_user_concept"),
        CheckConstraint("alpha > 0 AND beta > 0", name="beta_params_positive"),
        CheckConstraint("mastery BETWEEN 0 AND 1", name="mastery_range"),
        Index("ix_mastery_states_user_id_mastery", "user_id", "mastery"),
        Index("ix_mastery_states_user_id_review_due_at", "user_id", "review_due_at"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    concept_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )

    alpha: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    beta: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    mastery: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    evidence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_evidence_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    review_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    concept: Mapped[Concept] = relationship()

    def __repr__(self) -> str:
        return f"<MasteryState concept={self.concept_id} m={self.mastery:.2f}>"


class EvidenceEvent(UUIDPrimaryKeyMixin, Base):
    """One observation about what a learner knows. Append-only; never updated.

    Keeping the raw log rather than only the derived score is what makes the mastery
    model auditable: any state can be explained by the events that produced it, and a
    change to the weighting can be replayed over history instead of applying only to
    the future.
    """

    __tablename__ = "evidence_events"
    __table_args__ = (
        CheckConstraint("score BETWEEN 0 AND 1", name="score_range"),
        CheckConstraint("weight >= 0", name="weight_non_negative"),
        Index(
            "ix_evidence_events_user_concept_time",
            "user_id",
            "concept_id",
            "occurred_at",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    concept_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    source: Mapped[EvidenceSource] = mapped_column(
        pg_enum(EvidenceSource, "evidence_source"), nullable=False
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Provenance: which attempt, submission, or tutor turn produced this.
    origin_type: Mapped[str | None] = mapped_column(String(40))
    origin_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    def __repr__(self) -> str:
        return f"<EvidenceEvent {self.source} concept={self.concept_id} score={self.score:.2f}>"
