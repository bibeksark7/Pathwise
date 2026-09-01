"""Password hashing.

Argon2id, the current recommendation for password storage: memory-hard, so a GPU
gives an attacker far less advantage than it does against a fast hash like bcrypt or
(catastrophically) a bare SHA.

Two properties matter beyond picking the right algorithm:

* **Verification is constant-ish time and never leaks whether the user exists.**
  ``verify_dummy()`` exists so a login attempt for an unknown email burns the same
  work as one for a real account — without it, response time alone enumerates users.
* **Parameters can be raised without invalidating existing passwords.**
  ``needs_rehash`` reports when a stored hash was made with weaker settings, so it can
  be upgraded transparently during the next successful login.
"""

from __future__ import annotations

import contextlib
from typing import Final

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

#: OWASP's baseline for Argon2id (2024): 19 MiB, 2 iterations, 1 degree of
#: parallelism. Raising `memory_cost` is the most effective lever against GPU attacks.
_hasher: Final = PasswordHasher(
    time_cost=2,
    memory_cost=19 * 1024,
    parallelism=1,
    hash_len=32,
    salt_len=16,
)

MIN_PASSWORD_LENGTH: Final = 12
#: Argon2 hashes the password itself, so length is not a security limit — but an
#: unbounded input is a denial-of-service vector, since work scales with size.
MAX_PASSWORD_LENGTH: Final = 1024

#: Pre-computed so an unknown-email login does the same work as a real one.
_DUMMY_HASH: Final = _hasher.hash("a-password-that-is-never-a-real-one")


class PasswordPolicyError(ValueError):
    """The supplied password does not meet the minimum policy."""


def validate_password(password: str) -> None:
    """Check a password against policy before hashing it.

    Length only. Composition rules (a digit, a symbol, a capital) push people towards
    predictable substitutions and shorter passwords; length is the property that
    actually resists guessing.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")


def hash_password(password: str) -> str:
    """Hash a password after checking it against policy.

    Raises:
        PasswordPolicyError: if the password is too short or too long.
    """
    validate_password(password)
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Check a password against a stored hash. Never raises on a bad password."""
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False
    except Exception:
        # A corrupt or foreign hash format is a failed login, not a 500.
        return False


def verify_dummy() -> None:
    """Burn equivalent work when the account does not exist.

    Without this, an unknown email returns in microseconds while a known one takes
    the full Argon2 duration, and the difference enumerates accounts.
    """
    with contextlib.suppress(Exception):
        _hasher.verify(_DUMMY_HASH, "not-the-password")


def needs_rehash(password_hash: str) -> bool:
    """Whether a stored hash was produced with weaker parameters than current policy."""
    try:
        return _hasher.check_needs_rehash(password_hash)
    except Exception:
        # Unparseable means it certainly does not match current parameters.
        return True
