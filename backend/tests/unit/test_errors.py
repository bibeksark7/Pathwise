"""Domain errors must map to the right HTTP status without importing FastAPI."""

from __future__ import annotations

import pytest

from pathwise.api.errors import (
    AIRefusalError,
    AIValidationError,
    ConflictError,
    CycleError,
    NotFoundError,
    PathwiseError,
    ValidationError,
)


@pytest.mark.parametrize(
    ("error_class", "status"),
    [
        (NotFoundError, 404),
        (ValidationError, 422),
        (ConflictError, 409),
        (CycleError, 422),
        (AIValidationError, 502),
        (AIRefusalError, 502),
    ],
)
def test_status_codes(error_class: type[PathwiseError], status: int) -> None:
    assert error_class("boom").status_code == status


def test_details_are_carried_onto_the_response_body() -> None:
    error = NotFoundError("no such concept", concept_slug="gradient-descent")
    assert error.details == {"concept_slug": "gradient-descent"}
    assert error.message == "no such concept"


def test_cycle_error_is_a_validation_error() -> None:
    """A cyclic prerequisite edge is bad input, not a server fault."""
    assert issubclass(CycleError, ValidationError)
