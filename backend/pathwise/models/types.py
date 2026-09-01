"""Reusable column types and helpers."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB

from pathwise.models.enums import EMBEDDING_DIM

E = TypeVar("E", bound=StrEnum)


def pg_enum(enum_cls: type[E], name: str) -> SAEnum:
    """A native PostgreSQL enum storing member *values*, not member names.

    Without `values_callable`, SQLAlchemy persists `PREREQUISITE_OF` rather than
    `prerequisite_of`, which then disagrees with every JSON payload and seed file.
    """
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=True,
        create_constraint=False,
        values_callable=lambda cls: [member.value for member in cls],
    )


def embedding_column() -> Vector:
    """A pgvector column matching the configured embedding dimensionality."""
    return Vector(EMBEDDING_DIM)


# JSONB is used for schema-flexible payloads (learning objectives, decision traces,
# rubric definitions). Anything queried or joined on gets a real column instead.
JsonDict = JSONB
Json: Any = JSONB
