"""Request and response schemas for authentication.

Note what is absent from every response model: ``password_hash`` never appears, and
``UserResponse`` is built by explicit field selection rather than
``from_attributes`` over the whole ORM row. A response schema that mirrors a table
leaks whatever column is added to that table next.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from pathwise.services.auth.passwords import MAX_PASSWORD_LENGTH, MIN_PASSWORD_LENGTH


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    display_name: str | None = Field(default=None, max_length=100)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    # No length bounds: rejecting a login for a short password would reveal that the
    # policy changed, and the value is only ever compared, never stored.
    password: str = Field(max_length=MAX_PASSWORD_LENGTH)


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str = Field(min_length=1, max_length=512)


class UserResponse(BaseModel):
    """The public view of an account."""

    id: uuid.UUID
    email: EmailStr
    display_name: str | None
    created_at: datetime


class TokenResponse(BaseModel):
    """A token pair plus the account it belongs to.

    ``expires_at`` is absolute rather than a duration so a client with a skewed clock
    can compare against the server's own timestamps.
    """

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserResponse
