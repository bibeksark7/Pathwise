"""Domain exceptions and their HTTP mapping.

Services raise these; they know nothing about HTTP. The single handler registered in
`main.py` translates them into RFC 9457 problem responses, so a service can be reused
from a worker or the CLI without dragging FastAPI along.
"""

from __future__ import annotations

from typing import Any


class PathwiseError(Exception):
    """Base class for every expected Pathwise failure."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, /, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class NotFoundError(PathwiseError):
    status_code = 404
    error_code = "not_found"


class ValidationError(PathwiseError):
    """Input failed a domain rule (as opposed to a schema rule)."""

    status_code = 422
    error_code = "validation_error"


class ConflictError(PathwiseError):
    status_code = 409
    error_code = "conflict"


class AuthenticationError(PathwiseError):
    status_code = 401
    error_code = "authentication_failed"


class ForbiddenError(PathwiseError):
    status_code = 403
    error_code = "permission_denied"


class RateLimitError(PathwiseError):
    status_code = 429
    error_code = "rate_limited"


# --- AI-layer failures -------------------------------------------------------- #


class AIError(PathwiseError):
    """Base for failures originating in the LLM layer."""

    status_code = 502
    error_code = "ai_error"


class AIValidationError(AIError):
    """The model returned output that failed schema or domain validation.

    Raised only after the repair round-trip has also failed; callers are expected to
    fall back to a deterministic path rather than surface this to the user.
    """

    error_code = "ai_validation_failed"


class AIProviderError(AIError):
    """The provider itself failed — transport, auth, rate limit, or refusal."""

    error_code = "ai_provider_error"


class AIRefusalError(AIError):
    """The model declined the request on safety grounds."""

    error_code = "ai_refusal"


# --- Graph invariants --------------------------------------------------------- #


class CycleError(ValidationError):
    """A prerequisite edge would introduce a cycle in the knowledge graph.

    The graph must remain a DAG: topological ordering, prerequisite closure, and
    blame attribution are all undefined on a cyclic graph.
    """

    error_code = "graph_cycle"
