"""AI tutor conversations.

``retrieved_context`` on each assistant turn records exactly what the tutor was shown
— which mastery states, which prerequisite subgraph, which resource chunks. Without
it a bad answer cannot be diagnosed: you cannot tell whether the model reasoned badly
or was handed the wrong context.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pathwise.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from pathwise.models.enums import TutorRole
from pathwise.models.types import pg_enum


class TutorSession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A tutoring conversation, usually anchored to a concept the learner is stuck on."""

    __tablename__ = "tutor_sessions"
    __table_args__ = (Index("ix_tutor_sessions_user_id_created_at", "user_id", "created_at"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    concept_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("concepts.id", ondelete="SET NULL")
    )
    roadmap_node_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("roadmap_nodes.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="Tutoring session")

    # Running signals harvested from the conversation. These are weak evidence on
    # their own, but a repeated misconception across turns is a strong trigger for
    # the adaptation engine.
    struggle_signals: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    flagged_misconceptions: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    messages: Mapped[list[TutorMessage]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="TutorMessage.turn_index",
    )

    def __repr__(self) -> str:
        return f"<TutorSession {self.title!r}>"


class TutorMessage(UUIDPrimaryKeyMixin, Base):
    """A single turn in a tutoring conversation."""

    __tablename__ = "tutor_messages"
    __table_args__ = (Index("ix_tutor_messages_session_id_turn_index", "session_id", "turn_index"),)

    session_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tutor_sessions.id", ondelete="CASCADE"), nullable=False
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    role: Mapped[TutorRole] = mapped_column(pg_enum(TutorRole, "tutor_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Tool calls the tutor made this turn (get_mastery, search_resources, ...) and
    # their results, so the reasoning path is inspectable.
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    # Everything the tutor was shown: mastery snapshot, prerequisite subgraph,
    # retrieved resource chunk ids.
    retrieved_context: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    cited_resource_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(PGUUID(as_uuid=True)), nullable=False, default=list
    )

    llm_call_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("llm_calls.id", ondelete="SET NULL")
    )
    latency_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    session: Mapped[TutorSession] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        return f"<TutorMessage {self.role} #{self.turn_index}>"
