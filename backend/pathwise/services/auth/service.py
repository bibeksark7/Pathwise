"""Authentication service.

Owns registration, login, refresh with rotation, and logout. Knows nothing about
HTTP: it raises the domain errors from ``pathwise.api.errors`` and lets the route
layer translate them. That keeps it callable from a worker, the CLI, or a test.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pathwise.api.errors import AuthenticationError, ConflictError, ValidationError
from pathwise.config import Settings
from pathwise.logging_config import get_logger
from pathwise.models.user import RefreshToken as RefreshTokenRow
from pathwise.models.user import User
from pathwise.services.auth import passwords, tokens

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TokenPair:
    """What a successful login or refresh hands back."""

    access_token: str
    refresh_token: str
    expires_at: datetime
    user: User


def normalise_email(email: str) -> str:
    """Lowercase and trim, so ``Alice@Example.com`` cannot register twice.

    Deliberately does not strip dots or ``+tags``: those are provider-specific
    conventions, and normalising them would merge addresses that some providers treat
    as genuinely distinct.
    """
    return email.strip().lower()


class AuthService:
    """Account lifecycle and session management."""

    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings

    # --- registration --------------------------------------------------------- #

    async def register(self, email: str, password: str, *, display_name: str | None = None) -> User:
        """Create an account.

        Raises:
            ConflictError: if the email is already registered.
            ValidationError: if the password fails policy.
        """
        normalised = normalise_email(email)
        if not normalised or "@" not in normalised:
            raise ValidationError("A valid email address is required.")

        existing = await self._session.scalar(select(User).where(User.email == normalised))
        if existing is not None:
            # Registration necessarily reveals that an address is taken; there is no
            # way to create a unique account without doing so. Login does not.
            raise ConflictError("That email address is already registered.")

        try:
            password_hash = passwords.hash_password(password)
        except passwords.PasswordPolicyError as exc:
            raise ValidationError(str(exc)) from exc

        user = User(
            email=normalised,
            password_hash=password_hash,
            display_name=(display_name or "").strip() or None,
        )
        self._session.add(user)
        await self._session.flush()

        log.info("user_registered", user_id=str(user.id))
        return user

    # --- login ---------------------------------------------------------------- #

    async def authenticate(self, email: str, password: str) -> User:
        """Verify credentials.

        Always reports the same error for an unknown account and a wrong password,
        and always performs a hash verification, so neither the message nor the
        response time distinguishes the two.

        Raises:
            AuthenticationError: on any credential failure.
        """
        user = await self._session.scalar(select(User).where(User.email == normalise_email(email)))

        if user is None:
            passwords.verify_dummy()
            raise AuthenticationError("Incorrect email or password.")

        if not passwords.verify_password(password, user.password_hash):
            raise AuthenticationError("Incorrect email or password.")

        if not user.is_active:
            raise AuthenticationError("This account is disabled.")

        # Transparently upgrade a hash made under weaker parameters, now that we
        # hold the plaintext and know it is correct.
        if passwords.needs_rehash(user.password_hash):
            user.password_hash = passwords.hash_password(password)
            log.info("password_hash_upgraded", user_id=str(user.id))

        user.last_login_at = datetime.now(UTC)
        return user

    async def login(self, email: str, password: str, *, user_agent: str | None = None) -> TokenPair:
        """Authenticate and issue a fresh token pair."""
        user = await self.authenticate(email, password)
        pair, _ = await self._issue_tokens(user, user_agent=user_agent)
        return pair

    # --- refresh -------------------------------------------------------------- #

    async def refresh(self, refresh_token: str, *, user_agent: str | None = None) -> TokenPair:
        """Exchange a refresh token for a new pair, rotating the old one.

        A replayed token revokes every token in that user's family — see
        ``tokens.evaluate_rotation`` for why rejecting only the replayed token is not
        enough.

        Raises:
            AuthenticationError: if the token is unknown, expired, revoked, or reused.
        """
        digest = tokens.hash_refresh_token(refresh_token)
        row = await self._session.scalar(
            select(RefreshTokenRow).where(RefreshTokenRow.token_hash == digest)
        )

        stored = (
            tokens.StoredRefreshToken(
                id=row.id,
                user_id=row.user_id,
                expires_at=row.expires_at,
                revoked_at=row.revoked_at,
                replaced_by=row.replaced_by,
            )
            if row is not None
            else None
        )

        now = datetime.now(UTC)
        decision = tokens.evaluate_rotation(stored, now=now)

        if decision.requires_family_revocation:
            assert row is not None  # only reachable with a stored row
            await self._revoke_all_for_user(row.user_id, now=now)

            # Committed here, deliberately, rather than left to the caller. This path
            # raises, and the request-scoped session rolls back on an exception — so
            # leaving the revocation staged would discard the entire security response
            # to a detected token theft. This is the one place the service owns its
            # own transaction boundary.
            await self._session.commit()

            log.warning(
                "refresh_token_reuse_detected",
                user_id=str(row.user_id),
                token_id=str(row.id),
            )
            raise AuthenticationError(
                "This session has been revoked. Please sign in again.",
                reason=decision.reason,
            )

        if not decision.is_allowed or row is None:
            raise AuthenticationError("Invalid refresh token.", reason=decision.reason)

        user = await self._session.get(User, row.user_id)
        if user is None or not user.is_active:
            raise AuthenticationError("Invalid refresh token.", reason="user_unavailable")

        pair, replacement_id = await self._issue_tokens(user, user_agent=user_agent)

        # Mark the presented token spent only after the replacement exists, so a
        # failure part-way through cannot leave the user with no valid token at all.
        row.revoked_at = now
        row.replaced_by = replacement_id
        return pair

    # --- logout --------------------------------------------------------------- #

    async def logout(self, refresh_token: str) -> None:
        """Revoke a single session. Silent when the token is already unusable.

        Logging out must always appear to succeed: reporting "no such token" would
        let a caller probe which tokens are live.
        """
        digest = tokens.hash_refresh_token(refresh_token)
        row = await self._session.scalar(
            select(RefreshTokenRow).where(RefreshTokenRow.token_hash == digest)
        )
        if row is not None and row.revoked_at is None:
            row.revoked_at = datetime.now(UTC)
            log.info("user_logged_out", user_id=str(row.user_id))

    async def logout_everywhere(self, user_id: uuid.UUID) -> None:
        """Revoke every session for a user."""
        await self._revoke_all_for_user(user_id, now=datetime.now(UTC))

    # --- internals ------------------------------------------------------------ #

    async def _issue_tokens(
        self, user: User, *, user_agent: str | None
    ) -> tuple[TokenPair, uuid.UUID]:
        """Mint a pair and persist the refresh token. Returns the pair and its row id."""
        access_token, claims = tokens.create_access_token(user.id, self._settings)
        refresh = tokens.create_refresh_token(self._settings)

        row = RefreshTokenRow(
            user_id=user.id,
            token_hash=refresh.digest,
            expires_at=refresh.expires_at,
            user_agent=(user_agent or "")[:255] or None,
        )
        self._session.add(row)
        await self._session.flush()

        pair = TokenPair(
            access_token=access_token,
            refresh_token=refresh.value,
            expires_at=claims.expires_at,
            user=user,
        )
        return pair, row.id

    async def _revoke_all_for_user(self, user_id: uuid.UUID, *, now: datetime) -> None:
        await self._session.execute(
            update(RefreshTokenRow)
            .where(RefreshTokenRow.user_id == user_id, RefreshTokenRow.revoked_at.is_(None))
            .values(revoked_at=now)
        )
