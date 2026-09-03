"""Choosing what a diagnostic should probe.

A learner starting an ML roadmap has 39 required concepts. Asking about all of them
is a 90-minute exam nobody finishes; asking about a handful at random tells you
almost nothing. This module picks the small set of concepts whose answers reveal the
most about the rest — deterministically, before any question is written.

Two ideas do the work.

**Success propagates upward.** Demonstrating a concept is evidence for its
prerequisites — you could not have done it otherwise. So one question about a
downstream concept can cover several beneath it, and picking probes becomes a
maximum-coverage problem rather than a sampling one.

**Coverage alone would ask only the hardest questions.** The single deepest concept
covers almost the whole graph, so pure greedy coverage produces a diagnostic that a
beginner fails entirely — which tells you they are a beginner and nothing else.
Probes are therefore stratified across difficulty bands first, and chosen greedily
within each. A learner who fails the hard band and passes the easy one has told you
where they actually are.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from pathwise.api.errors import ValidationError
from pathwise.services.knowledge.graph import KnowledgeGraph
from pathwise.services.knowledge.mastery import PROPAGATION_MAX_HOPS

#: Default question count. Long enough to stratify meaningfully, short enough that
#: people finish it — an abandoned diagnostic yields no evidence at all.
DEFAULT_QUESTION_COUNT = 10

#: Difficulty bands, and the share of questions each receives. Weighted towards the
#: middle: extreme questions are less informative, because almost everyone fails the
#: hardest and passes the easiest regardless of level.
DIFFICULTY_BANDS: tuple[tuple[str, tuple[int, ...], float], ...] = (
    ("foundational", (1, 2), 0.3),
    ("intermediate", (3,), 0.4),
    ("advanced", (4, 5), 0.3),
)

#: Rough time per question, for setting expectations before the learner starts.
MINUTES_PER_QUESTION = 2


@dataclass(frozen=True, slots=True)
class ProbeTarget:
    """One concept the diagnostic will ask about."""

    concept_id: UUID
    slug: str
    name: str
    difficulty: int
    band: str
    #: Concepts this probe yields evidence about: itself plus the prerequisites that
    #: success would credit. This is why a short diagnostic can cover a long path.
    covers: frozenset[UUID]
    #: Objective ids the question should target. Binding a question to an objective
    #: is what turns a score into evidence about a capability rather than a topic.
    objective_ids: tuple[str, ...] = ()

    @property
    def coverage_size(self) -> int:
        return len(self.covers)


@dataclass(frozen=True, slots=True)
class DiagnosticBlueprint:
    """The plan for a diagnostic, before any question exists."""

    targets: tuple[ProbeTarget, ...]
    #: Every concept the diagnostic will yield some evidence about.
    covered: frozenset[UUID]
    #: Everything that was in scope, kept so coverage can be reported honestly.
    candidates: frozenset[UUID]

    @property
    def question_count(self) -> int:
        return len(self.targets)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def estimated_minutes(self) -> int:
        return self.question_count * MINUTES_PER_QUESTION

    @property
    def coverage_ratio(self) -> float:
        """Share of in-scope concepts this diagnostic says something about."""
        if not self.candidates:
            return 0.0
        return len(self.covered) / len(self.candidates)

    @property
    def uncovered(self) -> frozenset[UUID]:
        """Concepts the diagnostic will not touch.

        Surfaced rather than hidden. These keep their "no evidence" state, and the
        planner correctly treats them as unknown — which is why a short diagnostic is
        safe: it never causes anything to be assumed, only measured.
        """
        return self.candidates - self.covered

    def band_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {band: 0 for band, _, _ in DIFFICULTY_BANDS}
        for target in self.targets:
            counts[target.band] = counts.get(target.band, 0) + 1
        return counts


def select_probes(
    graph: KnowledgeGraph,
    candidate_ids: Iterable[UUID],
    *,
    question_count: int = DEFAULT_QUESTION_COUNT,
    already_known: Iterable[UUID] = (),
) -> DiagnosticBlueprint:
    """Choose which concepts a diagnostic should ask about.

    Args:
        graph: The knowledge graph.
        candidate_ids: Concepts in scope — normally the goal's prerequisite closure.
        question_count: How many probes to select.
        already_known: Concepts with existing evidence. Excluded from selection —
            re-testing something already measured spends a question to learn nothing.

    Raises:
        ValidationError: if there is nothing in scope to probe.
    """
    if question_count < 1:
        raise ValidationError("A diagnostic needs at least one question.")

    known = set(already_known)
    candidates = [cid for cid in candidate_ids if cid in graph and cid not in known]
    if not candidates:
        raise ValidationError(
            "There are no concepts left to assess.",
            candidates=0,
            excluded_as_known=len(known),
        )

    coverage = {cid: _coverage_of(graph, cid) for cid in candidates}
    banded = _partition_by_band(graph, candidates)
    allocation = _allocate(question_count, banded)

    selected: list[ProbeTarget] = []
    covered: set[UUID] = set()

    # Bands are processed hardest-first. A hard probe covers more, so taking it
    # early means the easier bands spend their questions on genuinely uncovered
    # ground rather than re-confirming what an advanced probe already implied.
    for band, _, _ in reversed(DIFFICULTY_BANDS):
        for _ in range(allocation.get(band, 0)):
            best = _most_informative(banded[band], coverage, covered, selected)
            if best is None:
                break
            selected.append(_to_target(graph, best, band, coverage[best]))
            covered |= coverage[best]

    # Any questions the bands could not place (a band with too few concepts) are
    # redistributed rather than dropped, so the diagnostic is the requested length.
    while len(selected) < question_count:
        remaining = [cid for cid in candidates if cid not in {t.concept_id for t in selected}]
        best = _most_informative(remaining, coverage, covered, selected)
        if best is None:
            break
        band = _band_for(graph.node(best).difficulty)
        selected.append(_to_target(graph, best, band, coverage[best]))
        covered |= coverage[best]

    selected.sort(key=lambda t: (t.difficulty, t.slug))

    return DiagnosticBlueprint(
        targets=tuple(selected),
        # Coverage is intersected with the candidate set: a probe's prerequisites can
        # reach outside the goal's closure, and counting those would inflate the
        # reported coverage with concepts nobody asked to learn.
        covered=frozenset(covered & set(candidates)),
        candidates=frozenset(candidates),
    )


def _coverage_of(graph: KnowledgeGraph, concept_id: UUID) -> frozenset[UUID]:
    """What one probe tells us about.

    The concept itself, plus prerequisites within propagation range — those are
    exactly the concepts a correct answer would produce evidence for, so the coverage
    model matches what the mastery model will actually do with the result.
    """
    reachable = graph.prerequisite_closure(concept_id, max_depth=PROPAGATION_MAX_HOPS)
    return frozenset({concept_id, *reachable})


def _partition_by_band(graph: KnowledgeGraph, candidates: Sequence[UUID]) -> dict[str, list[UUID]]:
    banded: dict[str, list[UUID]] = {band: [] for band, _, _ in DIFFICULTY_BANDS}
    for concept_id in candidates:
        banded[_band_for(graph.node(concept_id).difficulty)].append(concept_id)
    return banded


def _band_for(difficulty: int) -> str:
    for band, levels, _ in DIFFICULTY_BANDS:
        if difficulty in levels:
            return band
    return DIFFICULTY_BANDS[-1][0]


def _allocate(question_count: int, banded: Mapping[str, list[UUID]]) -> dict[str, int]:
    """Split questions across bands, skipping bands with nothing in them.

    Shares are renormalised over non-empty bands: a goal whose closure contains no
    advanced concepts should not lose 30% of its diagnostic to an empty band.
    """
    live = [(band, share) for band, _, share in DIFFICULTY_BANDS if banded.get(band)]
    if not live:
        return {}

    total_share = sum(share for _, share in live)
    allocation = {band: int(question_count * share / total_share) for band, share in live}

    # Hand out the remainder to the largest bands, so rounding never loses questions.
    shortfall = question_count - sum(allocation.values())
    for band, _ in sorted(live, key=lambda item: -len(banded[item[0]])):
        if shortfall <= 0:
            break
        allocation[band] += 1
        shortfall -= 1

    return allocation


def _most_informative(
    pool: Sequence[UUID],
    coverage: Mapping[UUID, frozenset[UUID]],
    covered: set[UUID],
    selected: Sequence[ProbeTarget],
) -> UUID | None:
    """The candidate adding the most genuinely new coverage.

    Ties break on total coverage then on id, so selection is deterministic — two
    learners with the same goal get the same diagnostic, and a regenerated one is
    identical rather than subtly different.
    """
    chosen = {target.concept_id for target in selected}
    best: UUID | None = None
    best_key = (-1, -1, "")

    for concept_id in pool:
        if concept_id in chosen:
            continue
        gain = len(coverage[concept_id] - covered)
        key = (gain, len(coverage[concept_id]), str(concept_id))
        if key > best_key:
            best_key, best = key, concept_id

    return best


def _to_target(
    graph: KnowledgeGraph, concept_id: UUID, band: str, covers: frozenset[UUID]
) -> ProbeTarget:
    node = graph.node(concept_id)
    return ProbeTarget(
        concept_id=concept_id,
        slug=node.slug,
        name=node.name,
        difficulty=node.difficulty,
        band=band,
        covers=covers,
        objective_ids=node.objective_ids,
    )
