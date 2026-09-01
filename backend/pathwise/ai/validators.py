"""Output validation and repair.

Structured output guarantees a response *parses*. It says nothing about whether the
content is *true*. A schema-valid roadmap can cite a concept that does not exist; a
schema-valid recommendation can carry a URL the model invented; a schema-valid
explanation can assert a score that appears nowhere in the trace it was given.

So every structured call passes through here:

    parse (provider)  ->  domain validators  ->  repair once  ->  deterministic fallback

The repair round-trip is worth exactly one attempt. A model that produced an invalid
concept slug will usually fix it when shown the error; one that fails twice is not
going to succeed on the third try, and each attempt costs real money and latency. At
that point the caller takes a deterministic path instead — which is why every AI
feature in Pathwise has one.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from pathwise.api.errors import AIValidationError
from pathwise.logging_config import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)
#: Contravariant, because a validator *consumes* its value. A validator for a
#: base model is usable wherever one for a subclass is expected, not the reverse.
T_contra = TypeVar("T_contra", bound=BaseModel, contravariant=True)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One thing wrong with a model's output.

    ``repair_hint`` is what gets sent back to the model. It is phrased as an
    instruction rather than a complaint, because "use only these slugs: [...]" repairs
    far more reliably than "invalid slug".
    """

    field: str
    problem: str
    repair_hint: str

    def __str__(self) -> str:
        return f"{self.field}: {self.problem}"


@dataclass(slots=True)
class ValidationResult:
    """The outcome of validating one output."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def add(self, field_name: str, problem: str, repair_hint: str) -> None:
        self.issues.append(ValidationIssue(field_name, problem, repair_hint))

    def merge(self, other: ValidationResult) -> None:
        self.issues.extend(other.issues)

    def repair_instructions(self) -> str:
        """A correction message for the repair round-trip.

        All issues go in one message rather than one at a time: fixing them
        individually costs a call per issue and lets the model reintroduce an earlier
        problem while addressing a later one.
        """
        lines = ["Your previous response had the following problems. Fix all of them:"]
        lines.extend(
            f"{index}. {issue.field} — {issue.problem}. {issue.repair_hint}"
            for index, issue in enumerate(self.issues, start=1)
        )
        lines.append("\nReturn the corrected output. Change nothing else.")
        return "\n".join(lines)

    def summary(self) -> list[str]:
        """Issue strings, for the `llm_calls` audit row."""
        return [str(issue) for issue in self.issues]


class Validator(Protocol[T_contra]):
    """A domain rule an output must satisfy."""

    def __call__(self, value: T_contra) -> ValidationResult: ...


# --------------------------------------------------------------------------- #
# Reusable validators
# --------------------------------------------------------------------------- #


def known_concepts(
    known_slugs: Iterable[str], extract: Callable[[T], Iterable[str]]
) -> Callable[[T], ValidationResult]:
    """Every referenced concept slug must exist in the knowledge graph.

    The single most common structured-output failure: a model asked to build a
    roadmap from a catalogue invents a plausible neighbour — `linear-algebra-basics`
    where the graph has `vectors-and-spaces`. Schema-valid, and completely broken
    downstream, because nothing will join to it.
    """
    allowed = set(known_slugs)

    def validate(value: T) -> ValidationResult:
        result = ValidationResult()
        unknown = sorted({slug for slug in extract(value) if slug not in allowed})
        if unknown:
            # Offer near-misses; the model usually meant one of them.
            suggestions = {slug: _closest(slug, allowed) for slug in unknown[:5]}
            near_misses = "; ".join(
                f"'{bad}' is not a concept" + (f" — did you mean '{near}'?" if near else "")
                for bad, near in suggestions.items()
            )
            result.add(
                "concept_slugs",
                f"references {len(unknown)} concept(s) that do not exist: {unknown[:10]}",
                f"{near_misses}. {_allowed_values_hint(allowed, 'concept slugs')}",
            )
        return result

    return validate


#: How many permitted values to spell out in a repair hint. Enough for the model to
#: actually pick the right one, bounded so a large catalogue does not blow up the
#: repair prompt and cost more than the original call.
_MAX_HINT_VALUES = 40


def _allowed_values_hint(allowed: set[str], noun: str) -> str:
    """Name the permitted values.

    A hint that says only "use a valid slug" gives the model nothing to correct
    towards, and it will usually invent a second plausible-looking value. Listing the
    real options is what makes the single repair attempt worth spending.
    """
    if not allowed:
        return f"No {noun} are available for this request."
    listed = sorted(allowed)[:_MAX_HINT_VALUES]
    suffix = (
        f" (and {len(allowed) - len(listed)} more from the catalogue above)"
        if len(allowed) > len(listed)
        else ""
    )
    return f"Valid {noun} are: {', '.join(listed)}{suffix}."


def known_resources(
    catalogue_urls: Iterable[str], extract: Callable[[T], Iterable[str]]
) -> Callable[[T], ValidationResult]:
    """Every URL must come from the curated catalogue.

    This is the rule that enforces the spec's "do not simply ask an LLM to invent
    resource URLs". Models generate confident, plausible, dead links — a real-looking
    course URL on a real domain that has never existed. The only defence that works
    is refusing to accept any URL we did not already validate ourselves.
    """
    allowed = set(catalogue_urls)

    def validate(value: T) -> ValidationResult:
        result = ValidationResult()
        invented = sorted({url for url in extract(value) if url not in allowed})
        if invented:
            result.add(
                "resource_urls",
                f"cites {len(invented)} URL(s) not present in the catalogue: {invented[:5]}",
                "Never write a URL of your own. " + _allowed_values_hint(allowed, "resource URLs"),
            )
        return result

    return validate


def acyclic_edges(
    extract: Callable[[T], Sequence[tuple[str, str]]],
) -> Callable[[T], ValidationResult]:
    """Proposed prerequisite edges must not form a cycle.

    A cycle makes topological ordering undefined, so a cyclic roadmap cannot be
    laid out, sequenced, or traversed — it would fail much later and much less
    legibly than here.
    """

    def validate(value: T) -> ValidationResult:
        result = ValidationResult()
        edges = list(extract(value))

        adjacency: dict[str, list[str]] = {}
        for source, target in edges:
            adjacency.setdefault(source, []).append(target)

        cycle = _find_cycle(adjacency)
        if cycle:
            result.add(
                "prerequisite_edges",
                f"form a cycle: {' -> '.join(cycle)}",
                "Prerequisites must form a directed acyclic graph. Remove whichever "
                "edge in that cycle is least necessary.",
            )
        return result

    return validate


def grounded_in_trace(
    allowed_numbers: Iterable[float], *, tolerance: float = 0.005
) -> Callable[[str], ValidationResult]:
    """Prose may only cite numbers that appear in the data it was given.

    This is the hallucination guard on generated explanations, and the reason the
    decision engine can claim to be explainable rather than merely narrated. The
    engine produces a trace; the model turns it into prose; this checks the prose did
    not acquire figures along the way — an invented "you scored 72%" is indistinguishable
    from a real one to the reader, and corrosive to trust in every other number shown.

    Percentages are matched against their decimal equivalents, since a trace holding
    `0.48` is legitimately rendered as "48%".
    """
    permitted = set()
    for number in allowed_numbers:
        permitted.add(round(number, 3))
        permitted.add(round(number * 100, 1))  # the same value written as a percentage

    def validate(text: str) -> ValidationResult:
        result = ValidationResult()
        cited = _extract_numbers(text)
        ungrounded = [
            number
            for number in cited
            if not any(abs(number - allowed) <= tolerance for allowed in permitted)
        ]
        if ungrounded:
            result.add(
                "explanation",
                f"cites figures absent from the decision trace: {ungrounded[:5]}",
                "Quote only numbers that appear in the trace. If you need to describe "
                "a quantity you were not given, describe it in words instead.",
            )
        return result

    return validate


def non_empty(*field_names: str) -> Callable[[T], ValidationResult]:
    """Named fields must not be blank.

    A schema with `str` accepts `""`, and an empty rationale or explanation is a
    silently degraded product rather than an error.
    """

    def validate(value: T) -> ValidationResult:
        result = ValidationResult()
        for name in field_names:
            content = getattr(value, name, None)
            if content is None or (isinstance(content, str) and not content.strip()):
                result.add(
                    name,
                    "is empty",
                    f"Provide a substantive value for '{name}'.",
                )
        return result

    return validate


class ValidationPipeline(Generic[T]):
    """A set of validators applied together."""

    def __init__(self, *validators: Callable[[T], ValidationResult]) -> None:
        self._validators = validators

    def __call__(self, value: T) -> ValidationResult:
        """Run every validator and collect all issues.

        Deliberately does not stop at the first failure: the repair round-trip should
        see everything wrong at once, or fixing one problem costs a call and reveals
        the next.
        """
        combined = ValidationResult()
        for validator in self._validators:
            combined.merge(validator(value))
        return combined

    def enforce(self, value: T, *, context: str = "") -> T:
        """Return the value, or raise if it fails validation."""
        result = self(value)
        if not result.is_valid:
            log.warning(
                "ai_output_failed_validation",
                context=context,
                issues=result.summary(),
            )
            raise AIValidationError(
                "Model output failed domain validation.",
                context=context,
                issues=result.summary(),
            )
        return value


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

#: Standard three-colour DFS marking, used by `_find_cycle`.
_WHITE, _GREY, _BLACK = 0, 1, 2

_NUMBER_PATTERN = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%?")

#: Figures that are part of ordinary prose rather than claims about the learner.
#: "the first two topics" must not be flagged as a fabricated statistic.
_PROSE_NUMBERS = frozenset({0.0, 1.0, 2.0, 3.0})


def _extract_numbers(text: str) -> list[float]:
    """Numbers a reader would take as factual claims."""
    found: list[float] = []
    for match in _NUMBER_PATTERN.finditer(text):
        raw = match.group(1)
        try:
            value = float(raw)
        except ValueError:  # pragma: no cover — the pattern guarantees a float
            continue
        if value in _PROSE_NUMBERS:
            continue
        found.append(value)
    return found


def _find_cycle(adjacency: dict[str, list[str]]) -> list[str]:
    """Return one cycle as a node path, or an empty list. Iterative DFS.

    Iterative rather than recursive because a model can propose an arbitrarily long
    chain, and blowing the Python stack while validating untrusted output would turn
    a bad response into a crashed worker.
    """
    colour: dict[str, int] = {}
    nodes = set(adjacency) | {t for targets in adjacency.values() for t in targets}

    for start in sorted(nodes):
        if colour.get(start, _WHITE) != _WHITE:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        path: list[str] = []

        while stack:
            node, index = stack.pop()
            if index == 0:
                if colour.get(node, _WHITE) == _BLACK:
                    continue
                colour[node] = _GREY
                path.append(node)

            neighbours = adjacency.get(node, [])
            if index < len(neighbours):
                stack.append((node, index + 1))
                neighbour = neighbours[index]
                if colour.get(neighbour, _WHITE) == _GREY:
                    cut = path.index(neighbour) if neighbour in path else 0
                    return [*path[cut:], neighbour]
                if colour.get(neighbour, _WHITE) == _WHITE:
                    stack.append((neighbour, 0))
            else:
                colour[node] = _BLACK
                if path and path[-1] == node:
                    path.pop()

    return []


def _closest(candidate: str, options: Iterable[str]) -> str | None:
    """The nearest known slug, for a repair hint. Empty when nothing is close."""
    import difflib

    matches = difflib.get_close_matches(candidate, list(options), n=1, cutoff=0.7)
    return matches[0] if matches else None
