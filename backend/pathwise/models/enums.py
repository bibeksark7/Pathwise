"""Domain enumerations.

These are stored as native PostgreSQL enums. Adding a member requires a migration;
removing one is a breaking change, so members are only ever appended.
"""

from __future__ import annotations

from enum import StrEnum

# Embedding dimensionality. Changing this requires a migration that rewrites every
# `vector` column and rebuilds its HNSW index — it is not a runtime setting.
EMBEDDING_DIM = 384


class ConceptSource(StrEnum):
    """Where a concept or edge came from, which determines how much we trust it."""

    SEED = "seed"  # hand-authored, curated
    LLM = "llm"  # model-proposed, validated, held pending review
    USER = "user"


class ConceptStatus(StrEnum):
    APPROVED = "approved"
    PENDING = "pending"
    REJECTED = "rejected"


class RelationType(StrEnum):
    """Edge semantics in the knowledge graph.

    Only PREREQUISITE_OF and DEPENDS_ON are *ordering* edges — they are the ones
    constrained to form a DAG and the ones traversed for prerequisite closure and
    blame attribution. The rest are associative and may form cycles freely.
    """

    PREREQUISITE_OF = "prerequisite_of"  # src must be learned before dst
    DEPENDS_ON = "depends_on"  # dst is required by src (inverse-flavoured ordering)
    BUILDS_ON = "builds_on"  # src extends dst conceptually
    RELATED_TO = "related_to"
    ALTERNATIVE_TO = "alternative_to"


ORDERING_RELATIONS: frozenset[RelationType] = frozenset(
    {RelationType.PREREQUISITE_OF, RelationType.DEPENDS_ON}
)


class LearningStyle(StrEnum):
    VIDEO = "video"
    READING = "reading"
    INTERACTIVE = "interactive"
    PROJECT_BASED = "project_based"
    MIXED = "mixed"


class NodeType(StrEnum):
    TOPIC = "topic"
    PRACTICE = "practice"
    ASSESSMENT = "assessment"
    PROJECT = "project"
    REVIEW = "review"


class NodeStatus(StrEnum):
    """The six visual states the roadmap graph renders."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    LOCKED = "locked"  # prerequisites unmet
    RECOMMENDED = "recommended"  # the decision engine's current pick


class RoadmapStatus(StrEnum):
    GENERATING = "generating"
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    FAILED = "failed"


class EvidenceSource(StrEnum):
    """Signals that move a mastery estimate, ordered loosely by reliability."""

    PROJECT = "project"
    ASSESSMENT = "assessment"
    QUIZ = "quiz"
    LESSON = "lesson"
    TUTOR = "tutor"
    TIME_ON_TASK = "time_on_task"
    SELF_REPORT = "self_report"
    PROPAGATED = "propagated"  # inferred from a prerequisite relationship, not observed


class ResourceType(StrEnum):
    DOCUMENTATION = "documentation"
    BOOK = "book"
    COURSE = "course"
    TUTORIAL = "tutorial"
    VIDEO = "video"
    ARTICLE = "article"
    PAPER = "paper"
    INTERACTIVE = "interactive"
    EXERCISE = "exercise"


class QuestionType(StrEnum):
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    CONCEPTUAL = "conceptual"
    CODING = "coding"
    PROBLEM_SOLVING = "problem_solving"
    SCENARIO = "scenario"


class AttemptStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    GRADED = "graded"
    ABANDONED = "abandoned"


class MutationType(StrEnum):
    """How the adaptation engine may rewrite a roadmap."""

    INSERT_REMEDIATION = "insert_remediation"
    ADD_PRACTICE = "add_practice"
    ADD_REVIEW = "add_review"
    COMPRESS = "compress"
    SKIP = "skip"
    REORDER = "reorder"
    SPLIT_NODE = "split_node"
    EXPAND = "expand"


class TutorRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMCallStatus(StrEnum):
    SUCCESS = "success"
    VALIDATION_FAILED = "validation_failed"
    PROVIDER_ERROR = "provider_error"
    REFUSED = "refused"
    CACHED = "cached"
