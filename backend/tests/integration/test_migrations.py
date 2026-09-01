"""Migrations must reproduce the models exactly, on a real PostgreSQL.

This is the authoritative check that `pathwise/models/` and
`pathwise/database/migrations/` have not drifted apart. It needs a live database with
the `vector` extension available, so it is marked `integration` and skipped when no
`PATHWISE_DATABASE_URL` points at a reachable server. CI provides one
(`pgvector/pgvector:pg16`), which is where this is expected to run.

The drift it catches is the common and expensive one: a column added to a model with
no accompanying migration. Everything passes locally and in tests — which build
tables from metadata — and then fails in production, where the table came from
migrations.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

import pathwise.models  # noqa: F401  — registers every table
from pathwise.config import get_settings
from pathwise.database.base import Base

pytestmark = pytest.mark.integration

ALEMBIC_INI = "alembic.ini"

#: Differences that are expected and must not fail the comparison. pgvector's index
#: options and PostgreSQL's own rendering of server defaults are not round-trippable
#: through reflection, so a raw diff reports them on a schema that is in fact correct.
IGNORED_DIFF_KINDS = frozenset({"modify_default"})


async def _database_is_reachable() -> bool:
    engine = create_async_engine(str(get_settings().database_url), pool_pre_ping=True)
    try:
        async with engine.connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        await engine.dispose()


@pytest.fixture
async def migrated_connection() -> AsyncIterator[AsyncConnection]:
    """Apply every migration from scratch, then yield a connection to the result."""
    if not await _database_is_reachable():
        pytest.skip("no reachable PATHWISE_DATABASE_URL; run `make up` or use CI")

    config = Config(ALEMBIC_INI)
    config.set_main_option("sqlalchemy.url", str(get_settings().database_url))

    command.downgrade(config, "base")
    command.upgrade(config, "head")

    engine = create_async_engine(str(get_settings().database_url))
    try:
        async with engine.connect() as conn:
            yield conn
    finally:
        await engine.dispose()


async def test_migrations_apply_from_scratch(migrated_connection: AsyncConnection) -> None:
    """`alembic upgrade head` on an empty database must succeed.

    Specifically exercises the shared enum types: `concept_source` and `relation_type`
    are each used by two tables, and an inline CREATE TYPE would fail on the second.
    """
    result = await migrated_connection.execute(
        sa.text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    )
    tables = {row[0] for row in result}
    expected = {table.name for table in Base.metadata.sorted_tables}
    assert expected <= tables


async def test_the_vector_extension_is_enabled(migrated_connection: AsyncConnection) -> None:
    result = await migrated_connection.execute(
        sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    )
    assert result.first() is not None


async def test_every_enum_type_exists_exactly_once(
    migrated_connection: AsyncConnection,
) -> None:
    result = await migrated_connection.execute(
        sa.text("SELECT typname, count(*) FROM pg_type WHERE typtype = 'e' GROUP BY typname")
    )
    counts = {row[0]: row[1] for row in result}
    assert counts, "no enum types were created"
    assert all(count == 1 for count in counts.values()), counts


async def test_no_drift_between_models_and_migrations(
    migrated_connection: AsyncConnection,
) -> None:
    """The check that matters: autogenerate must find nothing left to do.

    A non-empty diff means a model was changed without a migration, and the schema in
    production would not match the one the tests build.
    """

    def _diff(sync_conn: sa.Connection) -> list[tuple[object, ...]]:
        context = MigrationContext.configure(
            sync_conn, opts={"compare_type": True, "compare_server_default": False}
        )
        return compare_metadata(context, Base.metadata)

    differences = await migrated_connection.run_sync(_diff)
    significant = [
        diff for diff in differences if not (diff and str(diff[0]) in IGNORED_DIFF_KINDS)
    ]
    assert not significant, f"models and migrations have drifted: {significant}"


async def test_downgrade_removes_everything(migrated_connection: AsyncConnection) -> None:
    """A migration that cannot be reversed is a migration you cannot roll back."""
    config = Config(ALEMBIC_INI)
    config.set_main_option("sqlalchemy.url", str(get_settings().database_url))
    command.downgrade(config, "base")

    engine = create_async_engine(str(get_settings().database_url))
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                sa.text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' AND tablename <> 'alembic_version'"
                )
            )
            assert {row[0] for row in result} == set()
    finally:
        await engine.dispose()
        # Leave the database migrated for any test that follows.
        command.upgrade(config, "head")


def test_the_database_url_is_never_stored_in_alembic_ini() -> None:
    """Credentials belong in the environment, not in a committed config file."""
    with open(ALEMBIC_INI, encoding="utf-8") as handle:
        contents = handle.read()
    assert "sqlalchemy.url =" not in contents
    assert "password" not in contents.lower()


def test_alembic_ini_exists_outside_the_integration_gate() -> None:
    """Runs without a database so a missing config is caught locally, not only in CI."""
    assert os.path.exists(ALEMBIC_INI)
