"""Scorers for the evaluation suites.

Every scorer here is **deterministic**. None of them asks a model to judge another
model's output, and that is deliberate: an LLM judge introduces its own variance, its
own cost, and its own failure modes into the one part of the system whose job is to
detect variance. A judge is worth adding for genuinely subjective qualities — is this
explanation clear? — but almost everything Pathwise needs to measure has a right
answer that can be checked directly.

What these actually measure is the deterministic engines against hand-labelled cases:
does blame attribution name the prerequisite a human would name, does the decision
engine pick the topic a tutor would pick, does the planner include what the goal
requires. Those are the claims the project rests on, and they are exactly the claims
a unit test on a toy fixture cannot confirm.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Score:
    """One scorer's verdict on one case."""

    name: str
    value: float
    passed: bool
    detail: str = ""

    @property
    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "passed": self.passed,
            "detail": self.detail,
        }


class Scorer(Protocol):
    """Compares an actual result against an expected one."""

    @property
    def name(self) -> str:
        """Read-only, so frozen dataclasses satisfy the protocol."""
        ...

    def __call__(self, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> Score: ...


# --------------------------------------------------------------------------- #
# Ranking accuracy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TopKMatch:
    """Whether the right answer appeared in the top *k* results.

    Top-1 is the honest headline for a recommendation the learner will actually act
    on; top-3 is worth tracking alongside it, because a change that pushes the right
    answer from first to second is a regression even though top-3 stayed flat.
    """

    key: str
    expected_key: str
    k: int = 1
    name: str = "top_k_match"

    def __call__(self, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> Score:
        produced = _as_sequence(actual.get(self.key))
        target = expected.get(self.expected_key)

        if target is None:
            # Not applicable to this case. Scoring it as a failure would bury the
            # real failures under cases that never made a claim.
            return Score(self.name, 1.0, True, "not applicable to this case")

        top = produced[: self.k]
        hit = target in top
        return Score(
            name=self.name,
            value=1.0 if hit else 0.0,
            passed=hit,
            detail=(
                f"expected '{target}', got {list(top)}"
                if not hit
                else f"'{target}' at position {top.index(target) + 1}"
            ),
        )


@dataclass(frozen=True, slots=True)
class SetRecall:
    """How much of an expected set was produced.

    Recall rather than exact match: a plan that contains everything required plus one
    extra concept is not wrong in the way a plan missing a prerequisite is wrong.
    """

    key: str
    expected_key: str
    threshold: float = 1.0
    name: str = "set_recall"

    def __call__(self, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> Score:
        produced = set(_as_sequence(actual.get(self.key)))
        required = set(_as_sequence(expected.get(self.expected_key)))

        if not required:
            return Score(self.name, 1.0, True, "nothing required")

        found = required & produced
        value = len(found) / len(required)
        missing = sorted(required - produced)

        return Score(
            name=self.name,
            value=value,
            passed=value >= self.threshold,
            detail=f"missing {missing[:5]}" if missing else "all present",
        )


@dataclass(frozen=True, slots=True)
class SetExcludes:
    """Nothing from a forbidden set was produced.

    The counterpart to recall, and the one that catches the failure mode that matters
    for blame attribution: accusing the nearest prerequisite regardless of whether the
    learner has demonstrated it. Coverage alone would score that perfect.
    """

    key: str
    forbidden_key: str
    name: str = "excludes"

    def __call__(self, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> Score:
        produced = set(_as_sequence(actual.get(self.key)))
        forbidden = set(_as_sequence(expected.get(self.forbidden_key)))

        violations = sorted(produced & forbidden)
        return Score(
            name=self.name,
            value=0.0 if violations else 1.0,
            passed=not violations,
            detail=f"wrongly included {violations}" if violations else "none present",
        )


@dataclass(frozen=True, slots=True)
class OrderedBefore:
    """One item precedes another in a sequence.

    Checks the property a roadmap lives or dies on: prerequisites before dependents.
    """

    key: str
    expected_key: str = "ordering"
    name: str = "ordering"

    def __call__(self, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> Score:
        sequence = list(_as_sequence(actual.get(self.key)))
        pairs = expected.get(self.expected_key) or []

        position = {item: index for index, item in enumerate(sequence)}
        violations: list[str] = []

        for pair in pairs:
            first, second = pair[0], pair[1]
            if first not in position or second not in position:
                violations.append(f"{first}/{second} not both present")
            elif position[first] >= position[second]:
                violations.append(f"{first} should precede {second}")

        value = 1.0 - (len(violations) / len(pairs)) if pairs else 1.0
        return Score(
            name=self.name,
            value=max(0.0, value),
            passed=not violations,
            detail="; ".join(violations[:3]) if violations else "ordering correct",
        )


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AllKnown:
    """Every produced identifier exists in a reference set.

    The structural anti-hallucination check. Applied to concept slugs and to resource
    URLs, it is what turns "the model should not invent things" into a measurement.
    """

    key: str
    known: frozenset[str]
    name: str = "grounded"

    def __call__(self, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> Score:
        produced = set(_as_sequence(actual.get(self.key)))
        invented = sorted(produced - self.known)

        return Score(
            name=self.name,
            value=0.0 if invented else 1.0,
            passed=not invented,
            detail=f"invented {invented[:5]}" if invented else "all grounded",
        )


@dataclass(frozen=True, slots=True)
class NoFabricatedNumbers:
    """Prose cites only figures it was given.

    Reuses the same guard the production path uses, so the evaluation and the runtime
    check cannot drift apart and start disagreeing about what counts as grounded.
    """

    text_key: str = "explanation"
    allowed_key: str = "citable_numbers"
    name: str = "numbers_grounded"

    def __call__(self, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> Score:
        from pathwise.ai.validators import grounded_in_trace

        text = str(actual.get(self.text_key, ""))
        allowed = [float(value) for value in _as_sequence(actual.get(self.allowed_key))]

        result = grounded_in_trace(allowed)(text)
        return Score(
            name=self.name,
            value=1.0 if result.is_valid else 0.0,
            passed=result.is_valid,
            detail=result.issues[0].problem if result.issues else "no fabricated figures",
        )


# --------------------------------------------------------------------------- #
# Structural validity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BooleanFlag:
    """A named boolean in the result must hold.

    Used for invariants a run either satisfies or does not — the plan is acyclic, the
    output parsed, nothing was skipped that should not have been.
    """

    key: str
    name: str = "flag"
    expected_value: bool = True

    def __call__(self, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> Score:
        value = bool(actual.get(self.key, False))
        passed = value is self.expected_value
        return Score(
            name=self.name,
            value=1.0 if passed else 0.0,
            passed=passed,
            detail=f"{self.key}={value}, wanted {self.expected_value}",
        )


@dataclass(frozen=True, slots=True)
class WithinRange:
    """A numeric result falls inside an expected band.

    Bands rather than exact values: an estimate of 14.8 weeks against a hand-labelled
    15 is correct, and a suite that demanded equality would fail on every harmless
    retuning while saying nothing about quality.
    """

    key: str
    minimum: float | None = None
    maximum: float | None = None
    name: str = "in_range"

    def __call__(self, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> Score:
        raw = actual.get(self.key)
        if raw is None:
            return Score(self.name, 0.0, False, f"'{self.key}' missing")

        value = float(raw)
        below = self.minimum is not None and value < self.minimum
        above = self.maximum is not None and value > self.maximum
        passed = not (below or above)

        return Score(
            name=self.name,
            value=1.0 if passed else 0.0,
            passed=passed,
            detail=f"{self.key}={value:g}, expected [{self.minimum}, {self.maximum}]",
        )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Every scorer's verdict on one case."""

    case_id: str
    scores: tuple[Score, ...]
    actual: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def passed(self) -> bool:
        """A case passes only if every scorer passed.

        Deliberately strict. Averaging would let a case that produces a cyclic
        roadmap still "mostly pass" because its other scores were fine.
        """
        return self.error is None and all(score.passed for score in self.scores)

    @property
    def failures(self) -> tuple[Score, ...]:
        return tuple(score for score in self.scores if not score.passed)


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """The outcome of running one suite."""

    suite: str
    cases: tuple[CaseResult, ...]

    @property
    def passed_count(self) -> int:
        return sum(1 for case in self.cases if case.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / len(self.cases) if self.cases else 0.0

    @property
    def all_passed(self) -> bool:
        return self.passed_count == len(self.cases)

    def aggregate_scores(self) -> dict[str, float]:
        """Mean value per scorer — the numbers a regression gate compares."""
        totals: dict[str, list[float]] = {}
        for case in self.cases:
            for score in case.scores:
                totals.setdefault(score.name, []).append(score.value)
        return {name: sum(values) / len(values) for name, values in totals.items()}

    def failures(self) -> tuple[CaseResult, ...]:
        return tuple(case for case in self.cases if not case.passed)


def _as_sequence(value: Any) -> Sequence[str]:
    """Coerce a scorer input to a sequence of strings.

    Tolerant on purpose: a case file may write a single value where a list is
    expected, and failing on that would report a data-entry slip as a quality
    regression.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)
