"""Declarative base and shared ORM mixins.

The naming convention matters: without it Alembic autogenerate emits migrations that
drop and recreate constraints with server-assigned names, which makes downgrades
unreliable. Every constraint gets a deterministic name derived from its columns.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Shared declarative base for every Pathwise ORM model."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class UUIDPrimaryKeyMixin:
    """Adds a client-generatable UUID primary key.

    UUIDs are generated in Python rather than by the database so a full object graph
    (roadmap + nodes + edges) can be built and cross-referenced before a single INSERT
    is issued, then flushed in one batch.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )


class TimestampMixin:
    """Adds created/updated timestamps maintained by the database."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


def utcnow() -> datetime:
    """Timezone-aware current time. Never use `datetime.utcnow()` — it is naive."""
    return datetime.now(UTC)
