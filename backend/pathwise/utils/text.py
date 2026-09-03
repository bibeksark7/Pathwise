"""Small helpers for text that a learner reads.

Deterministic explanations are shown verbatim in the UI, so "1 step(s) from the goal"
is a visible defect rather than a cosmetic one. Pluralising in one place keeps that
consistent across the decision, adaptation, and planning engines.
"""

from __future__ import annotations

from typing import Final

__all__ = ["count_noun", "plural"]

#: Nouns whose plural is not formed by adding "s". Extended as needed rather than
#: pulling in an inflection library for a handful of words.
_IRREGULAR: Final[dict[str, str]] = {
    "is": "are",
    "has": "have",
    "this": "these",
}


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """Return the form of ``singular`` that agrees with ``count``."""
    if count == 1:
        return singular
    if plural_form is not None:
        return plural_form
    return _IRREGULAR.get(singular, f"{singular}s")


def count_noun(count: int, singular: str, plural_form: str | None = None) -> str:
    """Return ``count`` and the agreeing noun, e.g. ``"1 step"`` / ``"3 steps"``."""
    return f"{count} {plural(count, singular, plural_form)}"
