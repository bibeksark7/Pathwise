"""Identity, credentials, and the learner profile."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pathwise.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from pathwise.models.enums import LearningStyle
from pathwise.models.types import pg_enum

if TYPE_CHECKING:
    from pathwise.models.roadmap import Roadmap


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An account.

    Deliberately minimal: an email, a password hash, and a display name. No
    demographics, no analytics identifiers — the spec asks for no unnecessary
    personal information, and the learning model needs none of it.
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    profile: Mapped[LearningProfile | None] = relationship(
        back_populates="user", cascade="all, delete-orphan", uselist=False
    )
    refresh_tokens: Mapped[list[RefreshToken]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    roadmaps: Mapped[list[Roadmap]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.email}>"


class RefreshToken(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A rotating refresh token.

    Only the hash is stored, so a database leak does not yield usable tokens. Rotation
    is enforced by `replaced_by`: presenting an already-rotated token is treated as
    theft and revokes the whole chain.
    """

    __tablename__ = "refresh_tokens"
    __table_args__ = (Index("ix_refresh_tokens_user_id_revoked_at", "user_id", "revoked_at"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    replaced_by: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    user_agent: Mapped[str | None] = mapped_column(String(255))

    user: Mapped[User] = relationship(back_populates="refresh_tokens")


class LearningProfile(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """What the learner told us at onboarding.

    `goal_concept_ids` is the parsed, graph-grounded form of `goal_text`: the concepts
    the learner is aiming at. Every downstream calculation — goal relevance in the
    decision engine, roadmap scope, deadline pacing — reads these, not the free text.
    """

    __tablename__ = "learning_profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    goal_text: Mapped[str] = mapped_column(Text, nullable=False)
    goal_concept_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(PGUUID(as_uuid=True)), nullable=False, default=list
    )
    experience_summary: Mapped[str | None] = mapped_column(Text)
    learning_style: Mapped[LearningStyle] = mapped_column(
        pg_enum(LearningStyle, "learning_style"), nullable=False, default=LearningStyle.MIXED
    )
    hours_per_week: Mapped[float] = mapped_column(nullable=False, default=5.0)
    deadline: Mapped[date | None] = mapped_column(Date)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="UTC")
    diagnostic_completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    onboarding_step: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    user: Mapped[User] = relationship(back_populates="profile")
