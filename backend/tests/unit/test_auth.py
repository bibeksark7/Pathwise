"""Authentication tests.

The security-critical logic — hashing, token signing and verification, and the
refresh-rotation policy — is deliberately free of database access, so it can be
tested exhaustively here rather than behind an integration gate. Everything below
runs offline in milliseconds.

The endpoint flows that do need a database live in `tests/integration/`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient

from pathwise.api.errors import AuthenticationError
from pathwise.config import Settings
from pathwise.services.auth import passwords, tokens
from pathwise.services.auth.service import normalise_email

VALID_PASSWORD = "correct-horse-battery-staple"
T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def auth_settings() -> Settings:
    return Settings(
        jwt_secret="a-test-signing-secret-of-at-least-32-bytes",
        access_token_ttl_minutes=15,
    )


# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #


def test_a_password_verifies_against_its_own_hash() -> None:
    assert passwords.verify_password(VALID_PASSWORD, passwords.hash_password(VALID_PASSWORD))


def test_a_wrong_password_does_not_verify() -> None:
    assert not passwords.verify_password("wrong", passwords.hash_password(VALID_PASSWORD))


def test_hashes_are_salted_so_identical_passwords_differ() -> None:
    """Without per-hash salt, a leaked table reveals which users share a password."""
    assert passwords.hash_password(VALID_PASSWORD) != passwords.hash_password(VALID_PASSWORD)


def test_the_hash_is_argon2id() -> None:
    assert passwords.hash_password(VALID_PASSWORD).startswith("$argon2id$")


def test_the_plaintext_never_appears_in_the_hash() -> None:
    assert VALID_PASSWORD not in passwords.hash_password(VALID_PASSWORD)


@pytest.mark.parametrize("password", ["", "short", "a" * 11])
def test_passwords_below_the_minimum_length_are_rejected(password: str) -> None:
    with pytest.raises(passwords.PasswordPolicyError, match="at least"):
        passwords.hash_password(password)


def test_an_unbounded_password_is_rejected() -> None:
    """Argon2's cost scales with input size, so an unbounded password is a DoS vector."""
    with pytest.raises(passwords.PasswordPolicyError, match="at most"):
        passwords.hash_password("a" * (passwords.MAX_PASSWORD_LENGTH + 1))


def test_verifying_a_malformed_hash_returns_false_rather_than_raising() -> None:
    """A corrupt stored hash is a failed login, not a 500."""
    assert not passwords.verify_password(VALID_PASSWORD, "not-a-hash")
    assert not passwords.verify_password(VALID_PASSWORD, "")


def test_a_current_hash_does_not_need_rehashing() -> None:
    assert not passwords.needs_rehash(passwords.hash_password(VALID_PASSWORD))


def test_an_unparseable_hash_is_reported_as_needing_rehash() -> None:
    assert passwords.needs_rehash("$argon2id$garbage")


def test_the_dummy_verification_is_callable_and_silent() -> None:
    """Called on the unknown-account path; it must never raise or the timing
    equalisation would turn into a 500 that also identifies the account."""
    passwords.verify_dummy()


# --------------------------------------------------------------------------- #
# Access tokens
# --------------------------------------------------------------------------- #


def test_a_minted_token_round_trips(auth_settings: Settings) -> None:
    user_id = uuid.uuid4()
    token, claims = tokens.create_access_token(user_id, auth_settings)
    decoded = tokens.decode_access_token(token, auth_settings)
    assert decoded.user_id == user_id == claims.user_id


def test_an_expired_access_token_is_rejected(auth_settings: Settings) -> None:
    token, _ = tokens.create_access_token(
        uuid.uuid4(), auth_settings, now=datetime.now(UTC) - timedelta(hours=2)
    )
    with pytest.raises(AuthenticationError, match="expired"):
        tokens.decode_access_token(token, auth_settings)


def test_a_token_signed_with_another_secret_is_rejected(auth_settings: Settings) -> None:
    token, _ = tokens.create_access_token(uuid.uuid4(), auth_settings)
    other = Settings(jwt_secret="a-completely-different-secret-over-32-bytes")
    with pytest.raises(AuthenticationError, match="invalid"):
        tokens.decode_access_token(token, other)


def test_an_unsigned_token_is_rejected(auth_settings: Settings) -> None:
    """The classic JWT vulnerability: a forged `alg: none` token must not be accepted.

    Guarded by passing an explicit `algorithms` allowlist to `jwt.decode`; omitting it
    is a one-line mistake that makes every token forgeable.
    """
    forged = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "typ": "access",
        },
        key="",
        algorithm="none",
    )
    with pytest.raises(AuthenticationError):
        tokens.decode_access_token(forged, auth_settings)


def test_a_tampered_payload_is_rejected(auth_settings: Settings) -> None:
    token, _ = tokens.create_access_token(uuid.uuid4(), auth_settings)
    header, payload, signature = token.split(".")
    with pytest.raises(AuthenticationError):
        tokens.decode_access_token(f"{header}.{payload}x.{signature}", auth_settings)


def test_a_token_of_the_wrong_type_is_rejected(auth_settings: Settings) -> None:
    """A refresh token must never be usable as a bearer credential."""
    not_an_access_token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "typ": "refresh",
        },
        auth_settings.jwt_secret,
        algorithm=auth_settings.jwt_algorithm,
    )
    with pytest.raises(AuthenticationError, match="not an access token"):
        tokens.decode_access_token(not_an_access_token, auth_settings)


def test_a_token_without_required_claims_is_rejected(auth_settings: Settings) -> None:
    incomplete = jwt.encode(
        {"sub": str(uuid.uuid4()), "typ": "access"},
        auth_settings.jwt_secret,
        algorithm=auth_settings.jwt_algorithm,
    )
    with pytest.raises(AuthenticationError):
        tokens.decode_access_token(incomplete, auth_settings)


def test_a_non_uuid_subject_is_rejected(auth_settings: Settings) -> None:
    malformed = jwt.encode(
        {
            "sub": "'; DROP TABLE users; --",
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            "typ": "access",
        },
        auth_settings.jwt_secret,
        algorithm=auth_settings.jwt_algorithm,
    )
    with pytest.raises(AuthenticationError, match="not a valid user id"):
        tokens.decode_access_token(malformed, auth_settings)


def test_garbage_is_rejected(auth_settings: Settings) -> None:
    for value in ("", "garbage", "a.b.c", "Bearer token"):
        with pytest.raises(AuthenticationError):
            tokens.decode_access_token(value, auth_settings)


def test_access_tokens_are_short_lived(auth_settings: Settings) -> None:
    """A stolen access token cannot be revoked, so its lifetime is the exposure."""
    _, claims = tokens.create_access_token(uuid.uuid4(), auth_settings, now=T0)
    assert claims.expires_at - claims.issued_at <= timedelta(minutes=30)


def test_each_token_has_a_distinct_identifier(auth_settings: Settings) -> None:
    _, first = tokens.create_access_token(uuid.uuid4(), auth_settings)
    _, second = tokens.create_access_token(uuid.uuid4(), auth_settings)
    assert first.jti != second.jti


# --------------------------------------------------------------------------- #
# Refresh tokens
# --------------------------------------------------------------------------- #


def test_refresh_tokens_are_unpredictable(auth_settings: Settings) -> None:
    values = {tokens.create_refresh_token(auth_settings).value for _ in range(50)}
    assert len(values) == 50


def test_only_the_digest_is_ever_stored(auth_settings: Settings) -> None:
    """A database leak must not yield usable refresh tokens."""
    token = tokens.create_refresh_token(auth_settings)
    assert token.digest != token.value
    assert token.value not in token.digest
    assert token.digest == tokens.hash_refresh_token(token.value)


def test_the_digest_is_deterministic_so_lookup_works() -> None:
    assert tokens.hash_refresh_token("abc") == tokens.hash_refresh_token("abc")
    assert tokens.hash_refresh_token("abc") != tokens.hash_refresh_token("abd")


def test_refresh_tokens_outlive_access_tokens(auth_settings: Settings) -> None:
    token = tokens.create_refresh_token(auth_settings, now=T0)
    assert token.expires_at - T0 >= timedelta(days=1)


# --------------------------------------------------------------------------- #
# Rotation policy — the reason rotation is worth its complexity
# --------------------------------------------------------------------------- #


def stored(**overrides: object) -> tokens.StoredRefreshToken:
    defaults: dict[str, object] = {
        "id": uuid.uuid4(),
        "user_id": uuid.uuid4(),
        "expires_at": T0 + timedelta(days=30),
    }
    return tokens.StoredRefreshToken(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_a_valid_token_may_be_rotated() -> None:
    decision = tokens.evaluate_rotation(stored(), now=T0)
    assert decision.is_allowed
    assert not decision.requires_family_revocation


def test_an_unknown_token_is_rejected() -> None:
    decision = tokens.evaluate_rotation(None, now=T0)
    assert not decision.is_allowed
    assert decision.reason == "unknown_token"


def test_an_expired_refresh_token_is_rejected() -> None:
    decision = tokens.evaluate_rotation(stored(expires_at=T0 - timedelta(seconds=1)), now=T0)
    assert not decision.is_allowed
    assert decision.reason == "token_expired"


def test_a_revoked_token_is_rejected() -> None:
    decision = tokens.evaluate_rotation(stored(revoked_at=T0), now=T0)
    assert not decision.is_allowed
    assert decision.reason == "token_revoked"


def test_replaying_a_rotated_token_revokes_the_whole_family() -> None:
    """The property that makes rotation worth having.

    A token that was already exchanged and is presented again means one of the two
    holders is an attacker, and there is no way to tell which. Rejecting only this
    token would leave the thief holding the valid rotated one, so every session for
    the account is revoked instead.
    """
    decision = tokens.evaluate_rotation(stored(replaced_by=uuid.uuid4()), now=T0)
    assert not decision.is_allowed
    assert decision.requires_family_revocation
    assert decision.reason == "token_reuse_detected"


def test_reuse_detection_takes_precedence_over_expiry() -> None:
    """A replayed token that has since expired is still evidence of theft, and the
    family must still be revoked — checking expiry first would silently drop it."""
    decision = tokens.evaluate_rotation(
        stored(replaced_by=uuid.uuid4(), expires_at=T0 - timedelta(days=1)), now=T0
    )
    assert decision.requires_family_revocation


def test_a_token_expiring_exactly_now_is_rejected() -> None:
    """Boundary: expiry is exclusive, so `expires_at == now` is already expired."""
    assert not tokens.evaluate_rotation(stored(expires_at=T0), now=T0).is_allowed


# --------------------------------------------------------------------------- #
# Email normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Alice@Example.com", "alice@example.com"),
        ("  bob@test.dev  ", "bob@test.dev"),
        ("CAPS@LOUD.ORG", "caps@loud.org"),
    ],
)
def test_emails_are_normalised(raw: str, expected: str) -> None:
    assert normalise_email(raw) == expected


def test_plus_tags_are_preserved() -> None:
    """Some providers treat `a+x@` and `a@` as distinct addresses; merging them would
    let one person's account be locked out by another's registration."""
    assert normalise_email("user+pathwise@example.com") == "user+pathwise@example.com"


# --------------------------------------------------------------------------- #
# Endpoint surface (no database required)
# --------------------------------------------------------------------------- #


def test_every_auth_endpoint_is_registered(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for path in (
        "/api/auth/register",
        "/api/auth/login",
        "/api/auth/refresh",
        "/api/auth/logout",
        "/api/auth/logout-all",
        "/api/auth/me",
    ):
        assert path in paths, path


def test_a_missing_token_is_rejected(client: TestClient) -> None:
    response = client.get("/api/auth/me")
    assert response.status_code == 401
    assert response.json()["type"].endswith("authentication_failed")


def test_an_invalid_token_is_rejected(client: TestClient) -> None:
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer nonsense"})
    assert response.status_code == 401


def test_registration_rejects_a_short_password_before_touching_the_database(
    client: TestClient,
) -> None:
    """Schema validation runs first, so this returns 422 with no database available."""
    response = client.post("/api/auth/register", json={"email": "a@b.com", "password": "short"})
    assert response.status_code == 422


def test_registration_rejects_a_malformed_email(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register", json={"email": "not-an-email", "password": VALID_PASSWORD}
    )
    assert response.status_code == 422


def test_unknown_fields_are_rejected(client: TestClient) -> None:
    """`extra="forbid"` — a typo'd field must fail rather than be silently ignored."""
    response = client.post(
        "/api/auth/register",
        json={"email": "a@b.com", "password": VALID_PASSWORD, "is_admin": True},
    )
    assert response.status_code == 422


def test_the_password_is_never_echoed_in_an_error(client: TestClient) -> None:
    """A validation error renders the offending input; the password must not be in it."""
    secret = "a-very-secret-password-value"
    response = client.post("/api/auth/register", json={"email": "not-an-email", "password": secret})
    assert secret not in response.text
