"""Authentication routes.

Thin by design: each handler validates input via its schema, calls one service
method, commits, and shapes a response. All the policy — timing-safe login, rotation,
reuse detection — lives in ``pathwise.services.auth`` where it can be tested without
HTTP.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from pathwise.api.deps import AuthServiceDep, CurrentUser, DbSession, UserAgent
from pathwise.api.schemas.auth import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from pathwise.models.user import User
from pathwise.services.auth.service import TokenPair

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
    )


def _token_response(pair: TokenPair) -> TokenResponse:
    return TokenResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_at=pair.expires_at,
        user=_user_response(pair.user),
    )


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
)
async def register(
    payload: RegisterRequest, service: AuthServiceDep, session: DbSession
) -> UserResponse:
    user = await service.register(
        payload.email, payload.password, display_name=payload.display_name
    )
    await session.commit()
    return _user_response(user)


@router.post("/login", response_model=TokenResponse, summary="Sign in")
async def login(
    payload: LoginRequest,
    service: AuthServiceDep,
    session: DbSession,
    user_agent: UserAgent,
) -> TokenResponse:
    pair = await service.login(payload.email, payload.password, user_agent=user_agent)
    await session.commit()
    return _token_response(pair)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Exchange a refresh token for a new pair",
)
async def refresh(
    payload: RefreshRequest,
    service: AuthServiceDep,
    session: DbSession,
    user_agent: UserAgent,
) -> TokenResponse:
    """Rotate the refresh token.

    Reuse detection commits its own revocation inside the service, because this
    handler's commit is never reached on the error path.
    """
    pair = await service.refresh(payload.refresh_token, user_agent=user_agent)
    await session.commit()
    return _token_response(pair)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke the current session",
)
async def logout(payload: RefreshRequest, service: AuthServiceDep, session: DbSession) -> None:
    await service.logout(payload.refresh_token)
    await session.commit()


@router.post(
    "/logout-all",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke every session for the current account",
)
async def logout_all(user: CurrentUser, service: AuthServiceDep, session: DbSession) -> None:
    await service.logout_everywhere(user.id)
    await session.commit()


@router.get("/me", response_model=UserResponse, summary="The signed-in account")
async def me(user: CurrentUser) -> UserResponse:
    return _user_response(user)
