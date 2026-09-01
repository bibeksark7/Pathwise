"""SQLAlchemy ORM models.

Importing this package registers every table on ``Base.metadata``, which is what
Alembic autogenerate reflects against. **A new model module must be imported here or
its table is silently omitted from migrations.**
"""

from pathwise.database.base import Base
from pathwise.models.assessment import Answer, Assessment, Attempt, Question
from pathwise.models.knowledge import Concept, ConceptEdge, EvidenceEvent, MasteryState
from pathwise.models.observability import EvalResult, EvalRun, LLMCall
from pathwise.models.project import Project, ProjectSubmission
from pathwise.models.resource import (
    Resource,
    ResourceChunk,
    ResourceConcept,
    ResourceInteraction,
)
from pathwise.models.roadmap import Roadmap, RoadmapEdge, RoadmapNode, RoadmapRevision
from pathwise.models.tutor import TutorMessage, TutorSession
from pathwise.models.user import LearningProfile, RefreshToken, User

__all__ = [
    "Answer",
    "Assessment",
    "Attempt",
    "Base",
    "Concept",
    "ConceptEdge",
    "EvalResult",
    "EvalRun",
    "EvidenceEvent",
    "LLMCall",
    "LearningProfile",
    "MasteryState",
    "Project",
    "ProjectSubmission",
    "Question",
    "RefreshToken",
    "Resource",
    "ResourceChunk",
    "ResourceConcept",
    "ResourceInteraction",
    "Roadmap",
    "RoadmapEdge",
    "RoadmapNode",
    "RoadmapRevision",
    "TutorMessage",
    "TutorSession",
    "User",
]
