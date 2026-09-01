"""Learning resources and their concept bindings.

Every row here originates from a curated catalogue and has had its URL fetched and
checked before insert. The LLM ranks and explains resources; it never produces one.
``last_validated_at`` and ``http_status`` exist so a link that rots is detectable
rather than silently recommended forever.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Date,
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
from pathwise.models.enums import ResourceType
from pathwise.models.knowledge import Concept
from pathwise.models.types import embedding_column, pg_enum


class Resource(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A single external learning resource.

    ``canonical_url`` is the deduplication key — URLs are normalised (scheme, host
    case, tracking parameters, trailing slash) before hashing, so the same page
    submitted three different ways lands on one row.
    """

    __tablename__ = "resources"
    __table_args__ = (
        CheckConstraint("difficulty BETWEEN 1 AND 5", name="difficulty_range"),
        CheckConstraint("quality_prior BETWEEN 0 AND 1", name="quality_prior_range"),
        Index("ix_resources_resource_type_difficulty", "resource_type", "difficulty"),
    )

    title: Mapped[str] = mapped_column(String(300), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_url: Mapped[str] = mapped_column(Text, unique=True, nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    resource_type: Mapped[ResourceType] = mapped_column(
        pg_enum(ResourceType, "resource_type"), nullable=False
    )
    difficulty: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    duration_minutes: Mapped[int | None] = mapped_column(Integer)

    publisher: Mapped[str | None] = mapped_column(String(150), index=True)
    authors: Mapped[list[str]] = mapped_column(ARRAY(String(150)), nullable=False, default=list)
    published_at: Mapped[date | None] = mapped_column(Date)
    updated_at_source: Mapped[date | None] = mapped_column(Date)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")
    is_free: Mapped[bool] = mapped_column(nullable=False, default=True)

    # Reputation of the publisher, not of the individual page. Set from a curated
    # allowlist at ingest; a page from python.org starts above a random blog post.
    quality_prior: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    # Link health, written by the validation worker.
    last_validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    http_status: Mapped[int | None] = mapped_column(Integer)
    is_reachable: Mapped[bool] = mapped_column(nullable=False, default=True)

    embedding: Mapped[list[float] | None] = mapped_column(embedding_column())

    concept_links: Mapped[list[ResourceConcept]] = relationship(
        back_populates="resource", cascade="all, delete-orphan"
    )
    chunks: Mapped[list[ResourceChunk]] = relationship(
        back_populates="resource", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Resource {self.title!r} ({self.resource_type})>"


class ResourceConcept(UUIDPrimaryKeyMixin, Base):
    """How relevant one resource is to one concept.

    ``covers_objectives`` narrows it further: a resource can be highly relevant to
    "gradient descent" while covering only the intuition objective and none of the
    derivation, which is exactly the distinction that makes a recommendation useful
    to a learner who failed the derivation questions.
    """

    __tablename__ = "resource_concepts"
    __table_args__ = (
        UniqueConstraint("resource_id", "concept_id", name="uq_resource_concept"),
        CheckConstraint("relevance BETWEEN 0 AND 1", name="relevance_range"),
        Index("ix_resource_concepts_concept_id_relevance", "concept_id", "relevance"),
    )

    resource_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("resources.id", ondelete="CASCADE"), nullable=False
    )
    concept_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    relevance: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    covers_objectives: Mapped[list[str]] = mapped_column(
        ARRAY(String(40)), nullable=False, default=list
    )

    resource: Mapped[Resource] = relationship(back_populates="concept_links")
    concept: Mapped[Concept] = relationship()


class ResourceChunk(UUIDPrimaryKeyMixin, Base):
    """A retrievable passage of resource text, for tutor RAG.

    Only populated for resources whose text we may legitimately store (documentation
    excerpts, our own summaries). This is what grounds tutor explanations in something
    citable instead of in the model's recollection.
    """

    __tablename__ = "resource_chunks"
    __table_args__ = (
        UniqueConstraint("resource_id", "chunk_index", name="uq_resource_chunk_index"),
    )

    resource_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("resources.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    heading: Mapped[str | None] = mapped_column(String(300))
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(embedding_column())
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    resource: Mapped[Resource] = relationship(back_populates="chunks")


class ResourceInteraction(UUIDPrimaryKeyMixin, Base):
    """What a learner did with a recommended resource.

    Feeds two things: a per-learner history so the same video is not recommended
    twice, and a slow-moving usefulness signal that adjusts ranking over time.
    """

    __tablename__ = "resource_interactions"
    __table_args__ = (
        Index("ix_resource_interactions_user_id_resource_id", "user_id", "resource_id"),
        CheckConstraint("rating IS NULL OR rating BETWEEN 1 AND 5", name="rating_range"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    resource_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("resources.id", ondelete="CASCADE"), nullable=False
    )
    concept_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("concepts.id", ondelete="SET NULL")
    )
    recommended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rating: Mapped[int | None] = mapped_column(Integer)
    # The generated justification shown when this was recommended, kept so the
    # explanation can be audited against what the learner actually experienced.
    recommendation_reason: Mapped[str | None] = mapped_column(Text)
