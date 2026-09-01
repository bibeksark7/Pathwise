"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from pathwise.api.errors import AuthenticationError
from pathwise.config import Settings, get_settings
from pathwise.database.session import get_db_session
from pathwise.models.user import User
from pathwise.services.auth import tokens
from pathwise.services.auth.service import AuthService

# auto_error=False so a missing header raises our own AuthenticationError (and so the
# RFC 9457 problem shape) rather than FastAPI's default 403 body.
_bearer = HTTPBearer(auto_error=False, description="Access token from /auth/login")

DbSession = Annotated[AsyncSession, Depends(get_db_session)]
AppSettings = Annotated[Settings, Depends(get_settings)]


async def get_auth_service(session: DbSession, settings: AppSettings) -> AuthService:
    return AuthService(session, settings)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


async def get_current_user(
    session: DbSession,
    settings: AppSettings,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> User:
    """Resolve the authenticated user from a bearer access token.

    The user row is loaded on every request rather than trusted from the token: an
    account disabled a minute ago must stop working immediately, and a token cannot
    be revoked before it expires.

    Raises:
        AuthenticationError: if the header is missing, the token is invalid, or the
            account no longer exists or is disabled.
    """
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Authentication required.")

    claims = tokens.decode_access_token(credentials.credentials, settings)

    user = await session.get(User, claims.user_id)
    if user is None:
        raise AuthenticationError("Account no longer exists.")
    if not user.is_active:
        raise AuthenticationError("This account is disabled.")

    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def client_user_agent(request: Request) -> str | None:
    """The caller's user agent, recorded against a session for the sessions list."""
    return request.headers.get("user-agent")


UserAgent = Annotated[str | None, Depends(client_user_agent)]
