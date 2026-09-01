"""Initial schema.

Creates the whole Pathwise schema: identity, the knowledge graph, roadmaps and their
revision history, mastery state and the evidence log, resources, assessments,
projects, tutor conversations, and AI observability.

Two things are done explicitly rather than left to table creation:

* The ``vector`` extension is enabled before any table that uses it.
* Native enum types are created up front and referenced with ``create_type=False``.
  ``concept_source`` and ``relation_type`` are each used by two tables, and inline
  creation would emit ``CREATE TYPE`` twice and fail on the second table.

Revision ID: 0001
Revises:
Create Date: 2026-09-01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Every native enum, created before the tables that reference them.
ENUM_TYPES: dict[str, tuple[str, ...]] = {
    "concept_source": ("seed", "llm", "user"),
    "concept_status": ("approved", "pending", "rejected"),
    "resource_type": (
        "documentation",
        "book",
        "course",
        "tutorial",
        "video",
        "article",
        "paper",
        "interactive",
        "exercise",
    ),
    "relation_type": ("prerequisite_of", "depends_on", "builds_on", "related_to", "alternative_to"),
    "evidence_source": (
        "project",
        "assessment",
        "quiz",
        "lesson",
        "tutor",
        "time_on_task",
        "self_report",
        "propagated",
    ),
    "learning_style": ("video", "reading", "interactive", "project_based", "mixed"),
    "llm_call_status": ("success", "validation_failed", "provider_error", "refused", "cached"),
    "roadmap_status": ("generating", "active", "completed", "archived", "failed"),
    "node_type": ("topic", "practice", "assessment", "project", "review"),
    "node_status": (
        "not_started",
        "in_progress",
        "completed",
        "needs_review",
        "locked",
        "recommended",
    ),
    "attempt_status": ("in_progress", "submitted", "graded", "abandoned"),
    "question_type": (
        "multiple_choice",
        "short_answer",
        "conceptual",
        "coding",
        "problem_solving",
        "scenario",
    ),
    "tutor_role": ("user", "assistant", "tool"),
}


def upgrade() -> None:
    bind = op.get_bind()

    # pgvector powers semantic search over concepts, resources, and resource chunks.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    postgresql.ENUM("seed", "llm", "user", name="concept_source").create(bind, checkfirst=True)
    postgresql.ENUM("approved", "pending", "rejected", name="concept_status").create(
        bind, checkfirst=True
    )
    postgresql.ENUM(
        "documentation",
        "book",
        "course",
        "tutorial",
        "video",
        "article",
        "paper",
        "interactive",
        "exercise",
        name="resource_type",
    ).create(bind, checkfirst=True)
    postgresql.ENUM(
        "prerequisite_of",
        "depends_on",
        "builds_on",
        "related_to",
        "alternative_to",
        name="relation_type",
    ).create(bind, checkfirst=True)
    postgresql.ENUM(
        "project",
        "assessment",
        "quiz",
        "lesson",
        "tutor",
        "time_on_task",
        "self_report",
        "propagated",
        name="evidence_source",
    ).create(bind, checkfirst=True)
    postgresql.ENUM(
        "video", "reading", "interactive", "project_based", "mixed", name="learning_style"
    ).create(bind, checkfirst=True)
    postgresql.ENUM(
        "success",
        "validation_failed",
        "provider_error",
        "refused",
        "cached",
        name="llm_call_status",
    ).create(bind, checkfirst=True)
    postgresql.ENUM(
        "generating", "active", "completed", "archived", "failed", name="roadmap_status"
    ).create(bind, checkfirst=True)
    postgresql.ENUM(
        "topic", "practice", "assessment", "project", "review", name="node_type"
    ).create(bind, checkfirst=True)
    postgresql.ENUM(
        "not_started",
        "in_progress",
        "completed",
        "needs_review",
        "locked",
        "recommended",
        name="node_status",
    ).create(bind, checkfirst=True)
    postgresql.ENUM(
        "in_progress", "submitted", "graded", "abandoned", name="attempt_status"
    ).create(bind, checkfirst=True)
    postgresql.ENUM(
        "multiple_choice",
        "short_answer",
        "conceptual",
        "coding",
        "problem_solving",
        "scenario",
        name="question_type",
    ).create(bind, checkfirst=True)
    postgresql.ENUM("user", "assistant", "tool", name="tutor_role").create(bind, checkfirst=True)

    op.create_table(
        "concepts",
        sa.Column("slug", sa.String(length=120), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("domain", sa.String(length=60), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("difficulty", sa.Integer(), nullable=False),
        sa.Column("estimated_minutes", sa.Integer(), nullable=False),
        sa.Column("learning_objectives", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.String(length=50)), nullable=False),
        sa.Column("aliases", postgresql.ARRAY(sa.String(length=120)), nullable=False),
        sa.Column(
            "source", postgresql.ENUM(name="concept_source", create_type=False), nullable=False
        ),
        sa.Column(
            "status", postgresql.ENUM(name="concept_status", create_type=False), nullable=False
        ),
        sa.Column("embedding", Vector(dim=384), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("difficulty BETWEEN 1 AND 5", name=op.f("ck_concepts_difficulty_range")),
        sa.CheckConstraint(
            "estimated_minutes > 0", name=op.f("ck_concepts_estimated_minutes_positive")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_concepts")),
    )
    op.create_table(
        "eval_runs",
        sa.Column("suite", sa.String(length=60), nullable=False),
        sa.Column("dataset_version", sa.String(length=20), nullable=False),
        sa.Column("provider", sa.String(length=30), nullable=False),
        sa.Column("model", sa.String(length=80), nullable=False),
        sa.Column("prompt_name", sa.String(length=100), nullable=True),
        sa.Column("prompt_version", sa.String(length=20), nullable=True),
        sa.Column("git_sha", sa.String(length=40), nullable=True),
        sa.Column("case_count", sa.Integer(), nullable=False),
        sa.Column("passed_count", sa.Integer(), nullable=False),
        sa.Column("aggregate_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("total_cost_usd", sa.Float(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("mean_latency_ms", sa.Float(), nullable=False),
        sa.Column("is_baseline", sa.Boolean(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_eval_runs")),
    )
    op.create_table(
        "projects",
        sa.Column("slug", sa.String(length=120), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("domain", sa.String(length=60), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("requirements", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("concept_ids", postgresql.ARRAY(sa.UUID()), nullable=False),
        sa.Column("prerequisite_concept_ids", postgresql.ARRAY(sa.UUID()), nullable=False),
        sa.Column("difficulty", sa.Integer(), nullable=False),
        sa.Column("expected_hours", sa.Float(), nullable=False),
        sa.Column("rubric", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("extensions", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("starter_notes", sa.Text(), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("difficulty BETWEEN 1 AND 5", name=op.f("ck_projects_difficulty_range")),
        sa.CheckConstraint("expected_hours > 0", name=op.f("ck_projects_expected_hours_positive")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_projects")),
    )
    op.create_table(
        "resources",
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("canonical_url", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "resource_type",
            postgresql.ENUM(name="resource_type", create_type=False),
            nullable=False,
        ),
        sa.Column("difficulty", sa.Integer(), nullable=False),
        sa.Column("duration_minutes", sa.Integer(), nullable=True),
        sa.Column("publisher", sa.String(length=150), nullable=True),
        sa.Column("authors", postgresql.ARRAY(sa.String(length=150)), nullable=False),
        sa.Column("published_at", sa.Date(), nullable=True),
        sa.Column("updated_at_source", sa.Date(), nullable=True),
        sa.Column("language", sa.String(length=10), nullable=False),
        sa.Column("is_free", sa.Boolean(), nullable=False),
        sa.Column("quality_prior", sa.Float(), nullable=False),
        sa.Column("last_validated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("is_reachable", sa.Boolean(), nullable=False),
        sa.Column("embedding", Vector(dim=384), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "difficulty BETWEEN 1 AND 5", name=op.f("ck_resources_difficulty_range")
        ),
        sa.CheckConstraint(
            "quality_prior BETWEEN 0 AND 1", name=op.f("ck_resources_quality_prior_range")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_resources")),
    )
    op.create_table(
        "users",
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
    )
    op.create_table(
        "concept_edges",
        sa.Column("source_id", sa.UUID(), nullable=False),
        sa.Column("target_id", sa.UUID(), nullable=False),
        sa.Column(
            "relation", postgresql.ENUM(name="relation_type", create_type=False), nullable=False
        ),
        sa.Column("strength", sa.Float(), nullable=False),
        sa.Column(
            "source", postgresql.ENUM(name="concept_source", create_type=False), nullable=False
        ),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("source_id <> target_id", name=op.f("ck_concept_edges_no_self_loops")),
        sa.CheckConstraint(
            "strength BETWEEN 0 AND 1", name=op.f("ck_concept_edges_strength_range")
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["concepts.id"],
            name=op.f("fk_concept_edges_source_id_concepts"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_id"],
            ["concepts.id"],
            name=op.f("fk_concept_edges_target_id_concepts"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_concept_edges")),
        sa.UniqueConstraint("source_id", "target_id", "relation", name="uq_edge"),
    )
    op.create_table(
        "eval_results",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("case_id", sa.String(length=80), nullable=False),
        sa.Column("input_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("expected", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("actual", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["eval_runs.id"],
            name=op.f("fk_eval_results_run_id_eval_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_eval_results")),
    )
    op.create_table(
        "evidence_events",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("concept_id", sa.UUID(), nullable=False),
        sa.Column(
            "source", postgresql.ENUM(name="evidence_source", create_type=False), nullable=False
        ),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("origin_type", sa.String(length=40), nullable=True),
        sa.Column("origin_id", sa.UUID(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.CheckConstraint("score BETWEEN 0 AND 1", name=op.f("ck_evidence_events_score_range")),
        sa.CheckConstraint("weight >= 0", name=op.f("ck_evidence_events_weight_non_negative")),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concepts.id"],
            name=op.f("fk_evidence_events_concept_id_concepts"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_evidence_events_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evidence_events")),
    )
    op.create_table(
        "learning_profiles",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("goal_text", sa.Text(), nullable=False),
        sa.Column("goal_concept_ids", postgresql.ARRAY(sa.UUID()), nullable=False),
        sa.Column("experience_summary", sa.Text(), nullable=True),
        sa.Column(
            "learning_style",
            postgresql.ENUM(name="learning_style", create_type=False),
            nullable=False,
        ),
        sa.Column("hours_per_week", sa.Float(), nullable=False),
        sa.Column("deadline", sa.Date(), nullable=True),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("diagnostic_completed", sa.Boolean(), nullable=False),
        sa.Column("onboarding_step", sa.Integer(), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_learning_profiles_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_learning_profiles")),
        sa.UniqueConstraint("user_id", name=op.f("uq_learning_profiles_user_id")),
    )
    op.create_table(
        "llm_calls",
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("feature", sa.String(length=60), nullable=False),
        sa.Column("provider", sa.String(length=30), nullable=False),
        sa.Column("model", sa.String(length=80), nullable=False),
        sa.Column("prompt_name", sa.String(length=100), nullable=False),
        sa.Column("prompt_version", sa.String(length=20), nullable=False),
        sa.Column("prompt_hash", sa.String(length=16), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_write_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column(
            "status", postgresql.ENUM(name="llm_call_status", create_type=False), nullable=False
        ),
        sa.Column("validation_passed", sa.Boolean(), nullable=False),
        sa.Column("validation_errors", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("repair_attempts", sa.Integer(), nullable=False),
        sa.Column("stop_reason", sa.String(length=30), nullable=True),
        sa.Column("error_type", sa.String(length=60), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("provider_request_id", sa.String(length=80), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.CheckConstraint("cost_usd >= 0", name=op.f("ck_llm_calls_cost_non_negative")),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_llm_calls_user_id_users"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_llm_calls")),
    )
    op.create_table(
        "mastery_states",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("concept_id", sa.UUID(), nullable=False),
        sa.Column("alpha", sa.Float(), nullable=False),
        sa.Column("beta", sa.Float(), nullable=False),
        sa.Column("mastery", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("evidence_count", sa.Integer(), nullable=False),
        sa.Column("last_evidence_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_count", sa.Integer(), nullable=False),
        sa.Column("review_due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "alpha > 0 AND beta > 0", name=op.f("ck_mastery_states_beta_params_positive")
        ),
        sa.CheckConstraint("mastery BETWEEN 0 AND 1", name=op.f("ck_mastery_states_mastery_range")),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concepts.id"],
            name=op.f("fk_mastery_states_concept_id_concepts"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_mastery_states_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mastery_states")),
        sa.UniqueConstraint("user_id", "concept_id", name="uq_user_concept"),
    )
    op.create_table(
        "refresh_tokens",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_by", sa.UUID(), nullable=True),
        sa.Column("user_agent", sa.String(length=255), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_refresh_tokens_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_refresh_tokens")),
    )
    op.create_table(
        "resource_chunks",
        sa.Column("resource_id", sa.UUID(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("heading", sa.String(length=300), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(dim=384), nullable=True),
        sa.Column("extra", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["resource_id"],
            ["resources.id"],
            name=op.f("fk_resource_chunks_resource_id_resources"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_resource_chunks")),
        sa.UniqueConstraint("resource_id", "chunk_index", name="uq_resource_chunk_index"),
    )
    op.create_table(
        "resource_concepts",
        sa.Column("resource_id", sa.UUID(), nullable=False),
        sa.Column("concept_id", sa.UUID(), nullable=False),
        sa.Column("relevance", sa.Float(), nullable=False),
        sa.Column("covers_objectives", postgresql.ARRAY(sa.String(length=40)), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.CheckConstraint(
            "relevance BETWEEN 0 AND 1", name=op.f("ck_resource_concepts_relevance_range")
        ),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concepts.id"],
            name=op.f("fk_resource_concepts_concept_id_concepts"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["resource_id"],
            ["resources.id"],
            name=op.f("fk_resource_concepts_resource_id_resources"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_resource_concepts")),
        sa.UniqueConstraint("resource_id", "concept_id", name="uq_resource_concept"),
    )
    op.create_table(
        "resource_interactions",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("resource_id", sa.UUID(), nullable=False),
        sa.Column("concept_id", sa.UUID(), nullable=True),
        sa.Column("recommended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("recommendation_reason", sa.Text(), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.CheckConstraint(
            "rating IS NULL OR rating BETWEEN 1 AND 5",
            name=op.f("ck_resource_interactions_rating_range"),
        ),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concepts.id"],
            name=op.f("fk_resource_interactions_concept_id_concepts"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["resource_id"],
            ["resources.id"],
            name=op.f("fk_resource_interactions_resource_id_resources"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_resource_interactions_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_resource_interactions")),
    )
    op.create_table(
        "roadmaps",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("goal_text", sa.Text(), nullable=False),
        sa.Column(
            "status", postgresql.ENUM(name="roadmap_status", create_type=False), nullable=False
        ),
        sa.Column("revision_no", sa.Integer(), nullable=False),
        sa.Column("generation_error", sa.Text(), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_roadmaps_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_roadmaps")),
    )
    op.create_table(
        "roadmap_nodes",
        sa.Column("roadmap_id", sa.UUID(), nullable=False),
        sa.Column("concept_id", sa.UUID(), nullable=False),
        sa.Column(
            "node_type", postgresql.ENUM(name="node_type", create_type=False), nullable=False
        ),
        sa.Column("status", postgresql.ENUM(name="node_status", create_type=False), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column("estimated_minutes", sa.Integer(), nullable=False),
        sa.Column("minutes_spent", sa.Integer(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("added_by_revision", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "estimated_minutes > 0", name=op.f("ck_roadmap_nodes_estimated_minutes_positive")
        ),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concepts.id"],
            name=op.f("fk_roadmap_nodes_concept_id_concepts"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["roadmap_id"],
            ["roadmaps.id"],
            name=op.f("fk_roadmap_nodes_roadmap_id_roadmaps"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_roadmap_nodes")),
        sa.UniqueConstraint(
            "roadmap_id", "concept_id", "node_type", name="uq_roadmap_concept_type"
        ),
    )
    op.create_table(
        "roadmap_revisions",
        sa.Column("roadmap_id", sa.UUID(), nullable=False),
        sa.Column("revision_no", sa.Integer(), nullable=False),
        sa.Column("mutations", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("trigger", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["roadmap_id"],
            ["roadmaps.id"],
            name=op.f("fk_roadmap_revisions_roadmap_id_roadmaps"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_roadmap_revisions")),
        sa.UniqueConstraint("roadmap_id", "revision_no", name="uq_roadmap_revision_no"),
    )
    op.create_table(
        "assessments",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("purpose", sa.String(length=30), nullable=False),
        sa.Column("concept_ids", postgresql.ARRAY(sa.UUID()), nullable=False),
        sa.Column("roadmap_node_id", sa.UUID(), nullable=True),
        sa.Column("estimated_minutes", sa.Integer(), nullable=False),
        sa.Column("prompt_name", sa.String(length=100), nullable=True),
        sa.Column("prompt_version", sa.String(length=20), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["roadmap_node_id"],
            ["roadmap_nodes.id"],
            name=op.f("fk_assessments_roadmap_node_id_roadmap_nodes"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_assessments_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_assessments")),
    )
    op.create_table(
        "project_submissions",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("roadmap_node_id", sa.UUID(), nullable=True),
        sa.Column("repository_url", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("reflection", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("criterion_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("concept_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("feedback", sa.Text(), nullable=True),
        sa.Column("hours_spent", sa.Float(), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("graded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "score IS NULL OR score BETWEEN 0 AND 1",
            name=op.f("ck_project_submissions_score_range"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_project_submissions_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["roadmap_node_id"],
            ["roadmap_nodes.id"],
            name=op.f("fk_project_submissions_roadmap_node_id_roadmap_nodes"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_project_submissions_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_project_submissions")),
    )
    op.create_table(
        "roadmap_edges",
        sa.Column("roadmap_id", sa.UUID(), nullable=False),
        sa.Column("source_node_id", sa.UUID(), nullable=False),
        sa.Column("target_node_id", sa.UUID(), nullable=False),
        sa.Column(
            "relation", postgresql.ENUM(name="relation_type", create_type=False), nullable=False
        ),
        sa.Column("strength", sa.Float(), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.CheckConstraint(
            "source_node_id <> target_node_id", name=op.f("ck_roadmap_edges_no_self_loops")
        ),
        sa.ForeignKeyConstraint(
            ["roadmap_id"],
            ["roadmaps.id"],
            name=op.f("fk_roadmap_edges_roadmap_id_roadmaps"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_node_id"],
            ["roadmap_nodes.id"],
            name=op.f("fk_roadmap_edges_source_node_id_roadmap_nodes"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_node_id"],
            ["roadmap_nodes.id"],
            name=op.f("fk_roadmap_edges_target_node_id_roadmap_nodes"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_roadmap_edges")),
        sa.UniqueConstraint(
            "roadmap_id", "source_node_id", "target_node_id", name="uq_roadmap_edge"
        ),
    )
    op.create_table(
        "tutor_sessions",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("concept_id", sa.UUID(), nullable=True),
        sa.Column("roadmap_node_id", sa.UUID(), nullable=True),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("struggle_signals", sa.Integer(), nullable=False),
        sa.Column(
            "flagged_misconceptions", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concepts.id"],
            name=op.f("fk_tutor_sessions_concept_id_concepts"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["roadmap_node_id"],
            ["roadmap_nodes.id"],
            name=op.f("fk_tutor_sessions_roadmap_node_id_roadmap_nodes"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_tutor_sessions_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tutor_sessions")),
    )
    op.create_table(
        "attempts",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("assessment_id", sa.UUID(), nullable=False),
        sa.Column(
            "status", postgresql.ENUM(name="attempt_status", create_type=False), nullable=False
        ),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("concept_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("objective_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("graded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "score IS NULL OR score BETWEEN 0 AND 1", name=op.f("ck_attempts_score_range")
        ),
        sa.ForeignKeyConstraint(
            ["assessment_id"],
            ["assessments.id"],
            name=op.f("fk_attempts_assessment_id_assessments"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_attempts_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_attempts")),
    )
    op.create_table(
        "questions",
        sa.Column("assessment_id", sa.UUID(), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column(
            "question_type",
            postgresql.ENUM(name="question_type", create_type=False),
            nullable=False,
        ),
        sa.Column("stem", sa.Text(), nullable=False),
        sa.Column("options", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("answer_key", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("concept_ids", postgresql.ARRAY(sa.UUID()), nullable=False),
        sa.Column("objective_ids", postgresql.ARRAY(sa.String(length=40)), nullable=False),
        sa.Column("difficulty", sa.Integer(), nullable=False),
        sa.Column("points", sa.Float(), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.CheckConstraint(
            "difficulty BETWEEN 1 AND 5", name=op.f("ck_questions_difficulty_range")
        ),
        sa.CheckConstraint("points > 0", name=op.f("ck_questions_points_positive")),
        sa.ForeignKeyConstraint(
            ["assessment_id"],
            ["assessments.id"],
            name=op.f("fk_questions_assessment_id_assessments"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_questions")),
    )
    op.create_table(
        "tutor_messages",
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("role", postgresql.ENUM(name="tutor_role", create_type=False), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tool_calls", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("retrieved_context", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("cited_resource_ids", postgresql.ARRAY(sa.UUID()), nullable=False),
        sa.Column("llm_call_id", sa.UUID(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["llm_call_id"],
            ["llm_calls.id"],
            name=op.f("fk_tutor_messages_llm_call_id_llm_calls"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["tutor_sessions.id"],
            name=op.f("fk_tutor_messages_session_id_tutor_sessions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tutor_messages")),
    )
    op.create_table(
        "answers",
        sa.Column("attempt_id", sa.UUID(), nullable=False),
        sa.Column("question_id", sa.UUID(), nullable=False),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("grader", sa.String(length=30), nullable=True),
        sa.Column("objective_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("misconceptions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("feedback", sa.Text(), nullable=True),
        sa.Column("time_spent_seconds", sa.Integer(), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.CheckConstraint(
            "score IS NULL OR score BETWEEN 0 AND 1", name=op.f("ck_answers_score_range")
        ),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["attempts.id"],
            name=op.f("fk_answers_attempt_id_attempts"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["question_id"],
            ["questions.id"],
            name=op.f("fk_answers_question_id_questions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_answers")),
    )
    op.create_index(op.f("ix_concepts_domain"), "concepts", ["domain"], unique=False)
    op.create_index("ix_concepts_domain_status", "concepts", ["domain", "status"], unique=False)
    op.create_index(op.f("ix_concepts_slug"), "concepts", ["slug"], unique=True)
    op.create_index(
        "ix_eval_runs_suite_created_at", "eval_runs", ["suite", "created_at"], unique=False
    )
    op.create_index(
        "ix_projects_domain_difficulty", "projects", ["domain", "difficulty"], unique=False
    )
    op.create_index(op.f("ix_projects_slug"), "projects", ["slug"], unique=True)
    op.create_index(op.f("ix_resources_canonical_url"), "resources", ["canonical_url"], unique=True)
    op.create_index(op.f("ix_resources_publisher"), "resources", ["publisher"], unique=False)
    op.create_index(
        "ix_resources_resource_type_difficulty",
        "resources",
        ["resource_type", "difficulty"],
        unique=False,
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(
        "ix_concept_edges_source_relation", "concept_edges", ["source_id", "relation"], unique=False
    )
    op.create_index(
        "ix_concept_edges_target_relation", "concept_edges", ["target_id", "relation"], unique=False
    )
    op.create_index(
        "ix_eval_results_run_id_case_id", "eval_results", ["run_id", "case_id"], unique=False
    )
    op.create_index(
        "ix_evidence_events_user_concept_time",
        "evidence_events",
        ["user_id", "concept_id", "occurred_at"],
        unique=False,
    )
    op.create_index(op.f("ix_llm_calls_created_at"), "llm_calls", ["created_at"], unique=False)
    op.create_index(
        "ix_llm_calls_feature_created_at", "llm_calls", ["feature", "created_at"], unique=False
    )
    op.create_index(
        "ix_llm_calls_prompt_name_prompt_version",
        "llm_calls",
        ["prompt_name", "prompt_version"],
        unique=False,
    )
    op.create_index(
        "ix_llm_calls_user_id_created_at", "llm_calls", ["user_id", "created_at"], unique=False
    )
    op.create_index(
        "ix_mastery_states_user_id_mastery", "mastery_states", ["user_id", "mastery"], unique=False
    )
    op.create_index(
        "ix_mastery_states_user_id_review_due_at",
        "mastery_states",
        ["user_id", "review_due_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_refresh_tokens_token_hash"), "refresh_tokens", ["token_hash"], unique=True
    )
    op.create_index(
        "ix_refresh_tokens_user_id_revoked_at",
        "refresh_tokens",
        ["user_id", "revoked_at"],
        unique=False,
    )
    op.create_index(
        "ix_resource_concepts_concept_id_relevance",
        "resource_concepts",
        ["concept_id", "relevance"],
        unique=False,
    )
    op.create_index(
        "ix_resource_interactions_user_id_resource_id",
        "resource_interactions",
        ["user_id", "resource_id"],
        unique=False,
    )
    op.create_index("ix_roadmaps_user_id_status", "roadmaps", ["user_id", "status"], unique=False)
    op.create_index(
        "ix_roadmap_nodes_roadmap_id_order_index",
        "roadmap_nodes",
        ["roadmap_id", "order_index"],
        unique=False,
    )
    op.create_index(
        "ix_roadmap_nodes_roadmap_id_status",
        "roadmap_nodes",
        ["roadmap_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_roadmap_revisions_roadmap_id_created_at",
        "roadmap_revisions",
        ["roadmap_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_assessments_user_id_created_at", "assessments", ["user_id", "created_at"], unique=False
    )
    op.create_index(
        "ix_project_submissions_user_id_project_id",
        "project_submissions",
        ["user_id", "project_id"],
        unique=False,
    )
    op.create_index("ix_roadmap_edges_roadmap_id", "roadmap_edges", ["roadmap_id"], unique=False)
    op.create_index(
        "ix_tutor_sessions_user_id_created_at",
        "tutor_sessions",
        ["user_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_attempts_user_id_assessment_id", "attempts", ["user_id", "assessment_id"], unique=False
    )
    op.create_index(
        "ix_questions_assessment_id_order_index",
        "questions",
        ["assessment_id", "order_index"],
        unique=False,
    )
    op.create_index(
        "ix_tutor_messages_session_id_turn_index",
        "tutor_messages",
        ["session_id", "turn_index"],
        unique=False,
    )
    op.create_index(
        "ix_answers_attempt_id_question_id", "answers", ["attempt_id", "question_id"], unique=False
    )


def downgrade() -> None:
    bind = op.get_bind()

    op.drop_table("answers")
    op.drop_table("tutor_messages")
    op.drop_table("questions")
    op.drop_table("attempts")
    op.drop_table("tutor_sessions")
    op.drop_table("roadmap_edges")
    op.drop_table("project_submissions")
    op.drop_table("assessments")
    op.drop_table("roadmap_revisions")
    op.drop_table("roadmap_nodes")
    op.drop_table("roadmaps")
    op.drop_table("resource_interactions")
    op.drop_table("resource_concepts")
    op.drop_table("resource_chunks")
    op.drop_table("refresh_tokens")
    op.drop_table("mastery_states")
    op.drop_table("llm_calls")
    op.drop_table("learning_profiles")
    op.drop_table("evidence_events")
    op.drop_table("eval_results")
    op.drop_table("concept_edges")
    op.drop_table("users")
    op.drop_table("resources")
    op.drop_table("projects")
    op.drop_table("eval_runs")
    op.drop_table("concepts")

    postgresql.ENUM(name="tutor_role").drop(bind, checkfirst=True)
    postgresql.ENUM(name="question_type").drop(bind, checkfirst=True)
    postgresql.ENUM(name="attempt_status").drop(bind, checkfirst=True)
    postgresql.ENUM(name="node_status").drop(bind, checkfirst=True)
    postgresql.ENUM(name="node_type").drop(bind, checkfirst=True)
    postgresql.ENUM(name="roadmap_status").drop(bind, checkfirst=True)
    postgresql.ENUM(name="llm_call_status").drop(bind, checkfirst=True)
    postgresql.ENUM(name="learning_style").drop(bind, checkfirst=True)
    postgresql.ENUM(name="evidence_source").drop(bind, checkfirst=True)
    postgresql.ENUM(name="relation_type").drop(bind, checkfirst=True)
    postgresql.ENUM(name="resource_type").drop(bind, checkfirst=True)
    postgresql.ENUM(name="concept_status").drop(bind, checkfirst=True)
    postgresql.ENUM(name="concept_source").drop(bind, checkfirst=True)

    # The `vector` extension is deliberately left in place: other schemas in the
    # same database may depend on it, and dropping it is not ours to decide.
