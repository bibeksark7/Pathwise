"""Access and refresh tokens.

Two different mechanisms for two different jobs:

* **Access tokens** are short-lived signed JWTs. They are never checked against the
  database, which is what makes authentication cheap — and is also why they are given
  a 15-minute life, since a stolen one cannot be revoked before it expires.
* **Refresh tokens** are long-lived opaque random strings. Only their SHA-256 digest
  is stored, so a database leak yields nothing usable, and each one is single-use: a
  refresh rotates the token and marks the old one replaced.

The reuse-detection rule below is the reason rotation is worth the complexity. If a
token that has already been rotated is presented again, either it was stolen and the
thief is using it, or it was stolen and the legitimate user is. There is no way to
tell which, so the entire token family is revoked and both parties must log in again.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Literal

import jwt

from pathwise.api.errors import AuthenticationError
from pathwise.config import Settings

#: 256 bits of entropy. Long enough that guessing is not a threat model.
REFRESH_TOKEN_BYTES: Final = 32

TokenType = Literal["access", "refresh"]


@dataclass(frozen=True, slots=True)
class AccessTokenClaims:
    """The verified contents of an access token."""

    user_id: uuid.UUID
    issued_at: datetime
    expires_at: datetime
    jti: str


@dataclass(frozen=True, slots=True)
class RefreshToken:
    """A freshly minted refresh token.

    ``value`` is returned to the client exactly once and never stored. ``digest`` is
    what goes in the database.
    """

    value: str
    digest: str
    expires_at: datetime


def create_access_token(
    user_id: uuid.UUID, settings: Settings, *, now: datetime | None = None
) -> tuple[str, AccessTokenClaims]:
    """Mint a signed, short-lived access token."""
    issued_at = now or datetime.now(UTC)
    expires_at = issued_at + timedelta(minutes=settings.access_token_ttl_minutes)
    jti = uuid.uuid4().hex

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": jti,
        "typ": "access",
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, AccessTokenClaims(user_id, issued_at, expires_at, jti)


def decode_access_token(token: str, settings: Settings) -> AccessTokenClaims:
    """Verify an access token and return its claims.

    The ``algorithms`` allowlist is not optional: without it, a forged token
    specifying ``alg: none`` — or an HMAC token verified against a public key — would
    be accepted. This is the classic JWT vulnerability, and it is a one-line mistake.

    Raises:
        AuthenticationError: if the token is expired, malformed, wrongly signed, or
            of the wrong type.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Access token has expired.") from None
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Access token is invalid.", reason=type(exc).__name__) from None

    if payload.get("typ") != "access":
        # A refresh token must never be usable as a bearer credential.
        raise AuthenticationError("Token is not an access token.")

    try:
        user_id = uuid.UUID(payload["sub"])
    except (KeyError, ValueError):
        raise AuthenticationError("Access token subject is not a valid user id.") from None

    return AccessTokenClaims(
        user_id=user_id,
        issued_at=datetime.fromtimestamp(payload["iat"], tz=UTC),
        expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC),
        jti=payload.get("jti", ""),
    )


def create_refresh_token(settings: Settings, *, now: datetime | None = None) -> RefreshToken:
    """Mint an opaque refresh token and the digest to store for it."""
    issued_at = now or datetime.now(UTC)
    value = secrets.token_urlsafe(REFRESH_TOKEN_BYTES)
    return RefreshToken(
        value=value,
        digest=hash_refresh_token(value),
        expires_at=issued_at + timedelta(days=settings.refresh_token_ttl_days),
    )


def hash_refresh_token(value: str) -> str:
    """Digest a refresh token for storage and lookup.

    Plain SHA-256 rather than Argon2, deliberately: the token is 256 bits of
    cryptographic randomness, so there is no low-entropy secret to protect against
    brute force, and refresh happens on a hot path where a memory-hard hash would be
    a self-inflicted denial of service.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class StoredRefreshToken:
    """The stored state of a refresh token, as the rotation policy sees it.

    Decoupled from the ORM row so the policy below is a pure function that can be
    tested exhaustively without a database.
    """

    id: uuid.UUID
    user_id: uuid.UUID
    expires_at: datetime
    revoked_at: datetime | None = None
    replaced_by: uuid.UUID | None = None


@dataclass(frozen=True, slots=True)
class RotationDecision:
    """What to do with a presented refresh token."""

    outcome: Literal["rotate", "reject", "revoke_family"]
    reason: str

    @property
    def is_allowed(self) -> bool:
        return self.outcome == "rotate"

    @property
    def requires_family_revocation(self) -> bool:
        """A replayed token means the family is compromised and must all be revoked."""
        return self.outcome == "revoke_family"


def evaluate_rotation(stored: StoredRefreshToken | None, *, now: datetime) -> RotationDecision:
    """Decide whether a presented refresh token may be exchanged. Pure function.

    The interesting case is ``replaced_by`` being set: the token was already
    exchanged, so this is a replay. Either an attacker stole it and is using it after
    the legitimate user, or the legitimate user is using a copy the attacker already
    spent. There is no way to distinguish the two, so the whole family is revoked and
    everyone re-authenticates. Rejecting only this one token would leave a thief
    holding a valid rotated token.
    """
    if stored is None:
        return RotationDecision("reject", "unknown_token")
    if stored.replaced_by is not None:
        return RotationDecision("revoke_family", "token_reuse_detected")
    if stored.revoked_at is not None:
        return RotationDecision("reject", "token_revoked")
    if stored.expires_at <= now:
        return RotationDecision("reject", "token_expired")
    return RotationDecision("rotate", "ok")
