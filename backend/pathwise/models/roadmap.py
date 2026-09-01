"""The learner's roadmap: a projection of the knowledge graph, plus its edit history.

A roadmap is not a list. It is a subgraph of the global knowledge graph, selected and
ordered for one learner's goal, with per-node progress state layered on top.

``RoadmapRevision`` is the feature that makes the adaptation visible: every structural
change is recorded with the evidence that triggered it and the explanation shown to
the learner, so "why did my roadmap change?" is answered from stored data rather than
regenerated after the fact.
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
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pathwise.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from pathwise.models.enums import NodeStatus, NodeType, RelationType, RoadmapStatus
from pathwise.models.knowledge import Concept
from pathwise.models.types import pg_enum
from pathwise.models.user import User


class Roadmap(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One learning path towards one goal."""

    __tablename__ = "roadmaps"
    __table_args__ = (Index("ix_roadmaps_user_id_status", "user_id", "status"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    goal_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[RoadmapStatus] = mapped_column(
        pg_enum(RoadmapStatus, "roadmap_status"), nullable=False, default=RoadmapStatus.GENERATING
    )
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    generation_error: Mapped[str | None] = mapped_column(Text)

    user: Mapped[User] = relationship(back_populates="roadmaps")
    nodes: Mapped[list[RoadmapNode]] = relationship(
        back_populates="roadmap", cascade="all, delete-orphan"
    )
    edges: Mapped[list[RoadmapEdge]] = relationship(
        back_populates="roadmap", cascade="all, delete-orphan"
    )
    revisions: Mapped[list[RoadmapRevision]] = relationship(
        back_populates="roadmap",
        cascade="all, delete-orphan",
        order_by="RoadmapRevision.revision_no",
    )

    def __repr__(self) -> str:
        return f"<Roadmap {self.title!r} rev={self.revision_no}>"


class RoadmapNode(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One step in a roadmap, bound to a concept in the global graph.

    ``status`` is presentation state and is recomputed from mastery and prerequisite
    satisfaction — it is cached here so the graph endpoint is a single query, not a
    per-node calculation. ``LOCKED`` and ``RECOMMENDED`` in particular are always
    derived, never set by the learner.
    """

    __tablename__ = "roadmap_nodes"
    __table_args__ = (
        UniqueConstraint("roadmap_id", "concept_id", "node_type", name="uq_roadmap_concept_type"),
        CheckConstraint("estimated_minutes > 0", name="estimated_minutes_positive"),
        Index("ix_roadmap_nodes_roadmap_id_order_index", "roadmap_id", "order_index"),
        Index("ix_roadmap_nodes_roadmap_id_status", "roadmap_id", "status"),
    )

    roadmap_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("roadmaps.id", ondelete="CASCADE"), nullable=False
    )
    concept_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("concepts.id", ondelete="RESTRICT"), nullable=False
    )
    node_type: Mapped[NodeType] = mapped_column(
        pg_enum(NodeType, "node_type"), nullable=False, default=NodeType.TOPIC
    )
    status: Mapped[NodeStatus] = mapped_column(
        pg_enum(NodeStatus, "node_status"), nullable=False, default=NodeStatus.NOT_STARTED
    )

    order_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=120)
    minutes_spent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Why this node is in the path at all — shown in the node detail panel.
    rationale: Mapped[str | None] = mapped_column(Text)
    # Set when the adaptation engine inserted this node; names the revision.
    added_by_revision: Mapped[int | None] = mapped_column(Integer)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    roadmap: Mapped[Roadmap] = relationship(back_populates="nodes")
    concept: Mapped[Concept] = relationship()

    def __repr__(self) -> str:
        return f"<RoadmapNode {self.concept_id} {self.status}>"


class RoadmapEdge(UUIDPrimaryKeyMixin, Base):
    """A prerequisite link between two nodes of the same roadmap.

    This is a *projection* of the global graph, not a duplicate of it: only edges
    whose endpoints are both in the roadmap are materialised, so the frontend can lay
    the path out without loading the whole knowledge graph.
    """

    __tablename__ = "roadmap_edges"
    __table_args__ = (
        UniqueConstraint("roadmap_id", "source_node_id", "target_node_id", name="uq_roadmap_edge"),
        CheckConstraint("source_node_id <> target_node_id", name="no_self_loops"),
        Index("ix_roadmap_edges_roadmap_id", "roadmap_id"),
    )

    roadmap_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("roadmaps.id", ondelete="CASCADE"), nullable=False
    )
    source_node_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("roadmap_nodes.id", ondelete="CASCADE"), nullable=False
    )
    target_node_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("roadmap_nodes.id", ondelete="CASCADE"), nullable=False
    )
    relation: Mapped[RelationType] = mapped_column(
        pg_enum(RelationType, "relation_type"), nullable=False, default=RelationType.PREREQUISITE_OF
    )
    strength: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    roadmap: Mapped[Roadmap] = relationship(back_populates="edges")


class RoadmapRevision(UUIDPrimaryKeyMixin, Base):
    """An immutable record of one structural change to a roadmap.

    Three fields carry the whole "explain the adaptation" feature:

    * ``trigger`` — the evidence that caused it, as data (scores, concept ids, the
      blame ranking), so the claim can be checked.
    * ``mutations`` — the typed operations applied.
    * ``explanation`` — the learner-facing prose, generated from the two above and
      validated against them before it was stored.
    """

    __tablename__ = "roadmap_revisions"
    __table_args__ = (
        UniqueConstraint("roadmap_id", "revision_no", name="uq_roadmap_revision_no"),
        Index("ix_roadmap_revisions_roadmap_id_created_at", "roadmap_id", "created_at"),
    )

    roadmap_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("roadmaps.id", ondelete="CASCADE"), nullable=False
    )
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False)

    mutations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    trigger: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    explanation: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    roadmap: Mapped[Roadmap] = relationship(back_populates="revisions")

    def __repr__(self) -> str:
        return f"<RoadmapRevision {self.revision_no} ({len(self.mutations)} mutations)>"
