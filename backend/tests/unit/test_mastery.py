"""Mastery model tests.

Two things are being defended here.

The first is *behaviour a learner would notice*: a good assessment raises mastery, a
bad one lowers it, a project counts more than a self-report, knowledge fades, and a
single lucky quiz is not enough to skip material.

The second is *the property that makes the evidence log trustworthy*: Beta updates
commute, so replaying the same events in any order yields the same state. Without
that, the append-only log and the materialised `mastery_states` table would silently
drift apart and neither could be trusted.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pathwise.services.knowledge.mastery as mastery_module
from pathwise.models.enums import EvidenceSource, RelationType
from pathwise.services.knowledge.graph import GraphEdge, GraphNode, KnowledgeGraph
from pathwise.services.knowledge.mastery import (
    MASTERY_THRESHOLD,
    REVIEW_THRESHOLD,
    MasteryEstimate,
    Observation,
    aggregate_by_domain,
    apply_observation,
    concepts_due_for_review,
    confidence_map,
    effective_mastery_map,
    propagate_to_prerequisites,
    rebuild,
    rebuild_all,
    weakest_concepts,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def cid(slug: str) -> UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, slug)


def observe(
    slug: str = "target",
    score: float = 1.0,
    *,
    source: EvidenceSource = EvidenceSource.ASSESSMENT,
    at: datetime | None = None,
    weight: float = 1.0,
) -> Observation:
    return Observation(
        concept_id=cid(slug),
        source=source,
        score=score,
        occurred_at=at or T0,
        weight_multiplier=weight,
    )


# --------------------------------------------------------------------------- #
# The prior
# --------------------------------------------------------------------------- #


def test_prior_is_uninformative() -> None:
    estimate = MasteryEstimate()
    assert estimate.mastery == pytest.approx(0.5)
    assert estimate.confidence == pytest.approx(0.0)
    assert estimate.evidence_count == 0


def test_beta_parameters_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        MasteryEstimate(alpha=0.0, beta=1.0)


def test_observation_score_must_be_a_probability() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        observe(score=1.5)


# --------------------------------------------------------------------------- #
# Applying evidence
# --------------------------------------------------------------------------- #


def test_success_raises_mastery_and_failure_lowers_it() -> None:
    good, _ = apply_observation(MasteryEstimate(), observe(score=1.0))
    bad, _ = apply_observation(MasteryEstimate(), observe(score=0.0))
    assert good.mastery > 0.5 > bad.mastery


def test_evidence_always_raises_confidence() -> None:
    """Even a middling result tells us more than nothing."""
    before = MasteryEstimate()
    after, _ = apply_observation(before, observe(score=0.5))
    assert after.confidence > before.confidence


def test_a_project_moves_the_estimate_more_than_a_self_report() -> None:
    """Building something working is stronger evidence than saying you feel fine."""
    project, _ = apply_observation(
        MasteryEstimate(), observe(score=1.0, source=EvidenceSource.PROJECT)
    )
    self_report, _ = apply_observation(
        MasteryEstimate(), observe(score=1.0, source=EvidenceSource.SELF_REPORT)
    )
    assert project.mastery > self_report.mastery


def test_weight_multiplier_scales_the_update() -> None:
    """Three questions on a concept is stronger evidence than one."""
    heavy, _ = apply_observation(MasteryEstimate(), observe(score=1.0, weight=3.0))
    light, _ = apply_observation(MasteryEstimate(), observe(score=1.0, weight=1.0))
    assert heavy.mastery > light.mastery


def test_zero_weight_is_recorded_but_moves_nothing() -> None:
    """ "We saw this and it told us nothing" differs from "we saw nothing"."""
    before = MasteryEstimate()
    after, _ = apply_observation(before, observe(score=1.0, weight=0.0))
    assert after.mastery == pytest.approx(before.mastery)
    assert after.evidence_count == 1


def test_delta_reports_the_numbers_for_the_explanation() -> None:
    """The UI quotes these; it must not regenerate them."""
    _, delta = apply_observation(MasteryEstimate(), observe(score=0.48))
    assert delta.before == pytest.approx(0.5)
    assert delta.after < delta.before
    assert delta.change < 0
    assert delta.observation_score == pytest.approx(0.48)
    assert delta.source is EvidenceSource.ASSESSMENT


def test_repeated_success_converges_towards_certainty() -> None:
    estimate = MasteryEstimate()
    for _ in range(20):
        estimate, _ = apply_observation(estimate, observe(score=1.0))
    assert estimate.mastery > 0.9
    assert estimate.confidence > 0.8


# --------------------------------------------------------------------------- #
# Mastered / skippable
# --------------------------------------------------------------------------- #


def test_one_perfect_quiz_is_not_enough_to_skip_material() -> None:
    """The failure mode this guards against: stranding a learner on a lucky result."""
    estimate, _ = apply_observation(
        MasteryEstimate(), observe(score=1.0, source=EvidenceSource.QUIZ)
    )
    assert not estimate.is_skippable


def test_sustained_strong_performance_earns_a_skip() -> None:
    estimate = MasteryEstimate()
    for _ in range(8):
        estimate, _ = apply_observation(estimate, observe(score=1.0))
    assert estimate.is_skippable
    assert estimate.is_mastered


def test_high_mastery_with_low_confidence_is_not_skippable() -> None:
    """Level alone is never sufficient — the conjunction is the safety property."""
    thin = MasteryEstimate(alpha=2.6, beta=1.0, evidence_count=1, last_evidence_at=T0)
    assert thin.mastery > 0.7
    assert thin.confidence < 0.5
    assert not thin.is_skippable

    thick = MasteryEstimate(alpha=86.0, beta=14.0, evidence_count=20, last_evidence_at=T0)
    assert thick.confidence > thin.confidence
    assert thick.is_skippable


def test_confidence_ignores_whether_the_evidence_was_good_or_bad() -> None:
    """Confidence measures how much we know, not how well the learner did."""
    passed = rebuild([observe(score=1.0) for _ in range(5)])
    failed = rebuild([observe(score=0.0) for _ in range(5)])
    assert passed.confidence == pytest.approx(failed.confidence)
    assert passed.mastery > failed.mastery


def test_a_mixed_result_does_not_reduce_confidence() -> None:
    """A failure then a success moves the estimate towards 0.5 — a variance-based
    measure would report *less* confidence after *more* evidence."""
    after_failure = rebuild([observe(score=0.0)])
    after_both = rebuild([observe(score=0.0), observe(score=1.0)])
    assert after_both.confidence > after_failure.confidence


# --------------------------------------------------------------------------- #
# Forgetting
# --------------------------------------------------------------------------- #


def test_mastery_decays_with_time() -> None:
    estimate = rebuild([observe(score=1.0) for _ in range(6)])
    fresh = estimate.effective_mastery(T0)
    stale = estimate.effective_mastery(T0 + timedelta(days=120))
    assert stale < fresh


def test_decay_stops_at_a_retention_floor() -> None:
    """Well-learned material fades but is not forgotten entirely."""
    estimate = rebuild([observe(score=1.0) for _ in range(10)])
    after_years = estimate.effective_mastery(T0 + timedelta(days=3650))
    assert after_years > 0.5 * estimate.mastery


def test_stored_mastery_never_decays_only_the_effective_value_does() -> None:
    """State stays a pure function of the evidence log; the clock lives at read time."""
    estimate = rebuild([observe(score=1.0) for _ in range(6)])
    assert estimate.effective_mastery(T0 + timedelta(days=365)) < estimate.mastery


def test_stronger_mastery_decays_more_slowly() -> None:
    strong = rebuild([observe(score=1.0) for _ in range(12)])
    weak = rebuild([observe(score=0.6) for _ in range(2)])
    later = T0 + timedelta(days=60)
    strong_retention = strong.effective_mastery(later) / strong.mastery
    weak_retention = weak.effective_mastery(later) / weak.mastery
    assert strong_retention > weak_retention


def test_reviews_extend_retention() -> None:
    no_reviews = MasteryEstimate(alpha=9.0, beta=1.0, evidence_count=5, last_evidence_at=T0)
    reviewed = MasteryEstimate(
        alpha=9.0, beta=1.0, evidence_count=5, review_count=4, last_evidence_at=T0
    )
    later = T0 + timedelta(days=90)
    assert reviewed.effective_mastery(later) > no_reviews.effective_mastery(later)


def test_review_due_date_is_when_mastery_crosses_the_threshold() -> None:
    estimate = rebuild([observe(score=1.0) for _ in range(4)])
    due = estimate.review_due_at()
    assert due is not None
    assert estimate.effective_mastery(due) == pytest.approx(REVIEW_THRESHOLD, abs=1e-6)


def test_well_mastered_material_gets_a_very_long_review_interval() -> None:
    """Spacing effect: sustained mastery pushes the next review years out."""
    estimate = rebuild([observe(score=1.0) for _ in range(60)])
    due = estimate.review_due_at()
    assert due is not None
    assert due > T0 + timedelta(days=5 * 365)


def test_review_is_never_due_when_the_retention_floor_clears_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the branch that retires a concept from review entirely.

    Unreachable at the shipped constants (the floor is 0.6 x mastery and so cannot
    exceed the 0.70 threshold), but it must stay correct if retention is retuned.
    """
    monkeypatch.setattr(mastery_module, "RETENTION_FLOOR_RATIO", 0.95)
    estimate = rebuild([observe(score=1.0) for _ in range(20)])
    assert estimate.review_due_at() is None


def test_never_learned_material_has_no_review_date() -> None:
    assert MasteryEstimate().review_due_at() is None


def test_out_of_order_arrival_does_not_rewind_the_clock() -> None:
    """A late-arriving old event must not make fresh knowledge look stale."""
    estimate = rebuild([observe(score=1.0, at=T0 + timedelta(days=30)), observe(score=1.0, at=T0)])
    assert estimate.last_evidence_at == T0 + timedelta(days=30)


def test_concepts_due_for_review_are_ordered_by_decay() -> None:
    strong = rebuild([observe("strong", score=1.0) for _ in range(3)])
    weaker = rebuild([observe("weaker", score=0.8) for _ in range(3)])
    now = T0 + timedelta(days=200)
    due = concepts_due_for_review({cid("strong"): strong, cid("weaker"): weaker}, now)
    assert due == () or due[0] == cid("weaker")


def test_unmeasured_concepts_are_never_due_for_review() -> None:
    """You cannot be due to revise something you never learned."""
    assert concepts_due_for_review({cid("x"): MasteryEstimate()}, T0 + timedelta(days=999)) == ()


# --------------------------------------------------------------------------- #
# Propagation to prerequisites — how Pathwise reduces a roadmap
# --------------------------------------------------------------------------- #


@pytest.fixture
def chain() -> KnowledgeGraph:
    """calculus -> optimization -> gradient-descent"""
    slugs = ("calculus", "optimization", "gradient-descent")
    nodes = [GraphNode(id=cid(s), slug=s, name=s, domain="ml") for s in slugs]
    edges = [
        GraphEdge(cid("calculus"), cid("optimization"), RelationType.PREREQUISITE_OF, 1.0),
        GraphEdge(cid("optimization"), cid("gradient-descent"), RelationType.PREREQUISITE_OF, 1.0),
    ]
    return KnowledgeGraph(nodes, edges)


def test_success_credits_prerequisites(chain: KnowledgeGraph) -> None:
    """Succeeding at gradient descent is evidence you can do the optimisation under it."""
    derived = propagate_to_prerequisites(chain, observe("gradient-descent", score=0.95))
    assert {d.concept_id for d in derived} == {cid("optimization"), cid("calculus")}
    assert all(d.source is EvidenceSource.PROPAGATED for d in derived)


def test_failure_never_propagates(chain: KnowledgeGraph) -> None:
    """A failure says something is wrong but not where — guessing would corrupt
    exactly the prerequisite estimates blame attribution depends on."""
    assert propagate_to_prerequisites(chain, observe("gradient-descent", score=0.2)) == ()


def test_a_mediocre_score_does_not_propagate(chain: KnowledgeGraph) -> None:
    assert propagate_to_prerequisites(chain, observe("gradient-descent", score=0.6)) == ()


def test_propagation_attenuates_with_distance(chain: KnowledgeGraph) -> None:
    derived = {
        d.concept_id: d
        for d in propagate_to_prerequisites(chain, observe("gradient-descent", score=1.0))
    }
    near, far = derived[cid("optimization")], derived[cid("calculus")]
    assert near.effective_weight > far.effective_weight
    assert near.score > far.score  # the distant inference is pulled towards neutral


def test_propagated_evidence_is_weaker_than_direct_evidence(chain: KnowledgeGraph) -> None:
    """Inference about a prerequisite must never outweigh measuring it."""
    derived = propagate_to_prerequisites(chain, observe("gradient-descent", score=1.0))
    direct = observe("optimization", score=1.0, source=EvidenceSource.ASSESSMENT)
    assert derived[0].effective_weight < direct.effective_weight


def test_propagation_stops_at_the_hop_limit() -> None:
    slugs = ("a", "b", "c", "d")
    nodes = [GraphNode(id=cid(s), slug=s, name=s) for s in slugs]
    edges = [
        GraphEdge(cid(x), cid(y), RelationType.PREREQUISITE_OF, 1.0)
        for x, y in (("a", "b"), ("b", "c"), ("c", "d"))
    ]
    graph = KnowledgeGraph(nodes, edges)
    derived = propagate_to_prerequisites(graph, observe("d", score=1.0))
    assert cid("a") not in {d.concept_id for d in derived}


def test_propagation_ignores_concepts_outside_the_graph(chain: KnowledgeGraph) -> None:
    assert propagate_to_prerequisites(chain, observe("unknown", score=1.0)) == ()


# --------------------------------------------------------------------------- #
# Aggregations for the dashboard
# --------------------------------------------------------------------------- #


def test_rebuild_all_groups_by_concept() -> None:
    estimates = rebuild_all(
        [observe("a", score=1.0), observe("b", score=0.0), observe("a", score=1.0)]
    )
    assert estimates[cid("a")].evidence_count == 2
    assert estimates[cid("a")].mastery > estimates[cid("b")].mastery


def test_weakest_concepts_ignores_the_unstudied() -> None:
    """A concept not yet reached is not a weakness — conflating them ruins week one."""
    estimates = {
        cid("studied"): rebuild([observe("studied", score=0.1)]),
        cid("new"): MasteryEstimate(),
    }
    weakest = weakest_concepts(estimates, T0)
    assert [concept_id for concept_id, _ in weakest] == [cid("studied")]


def test_mastery_by_domain_weights_harder_concepts_more() -> None:
    nodes = [
        GraphNode(id=cid("easy"), slug="easy", name="Easy", difficulty=1, domain="math"),
        GraphNode(id=cid("hard"), slug="hard", name="Hard", difficulty=5, domain="math"),
    ]
    graph = KnowledgeGraph(nodes, [])
    strong_on_hard = {
        cid("easy"): MasteryEstimate(alpha=1.0, beta=9.0, evidence_count=3, last_evidence_at=T0),
        cid("hard"): MasteryEstimate(alpha=9.0, beta=1.0, evidence_count=3, last_evidence_at=T0),
    }
    strong_on_easy = {
        cid("easy"): MasteryEstimate(alpha=9.0, beta=1.0, evidence_count=3, last_evidence_at=T0),
        cid("hard"): MasteryEstimate(alpha=1.0, beta=9.0, evidence_count=3, last_evidence_at=T0),
    }
    assert (
        aggregate_by_domain(strong_on_hard, graph, T0)["math"]
        > aggregate_by_domain(strong_on_easy, graph, T0)["math"]
    )


def test_maps_expose_decayed_mastery_and_confidence_for_graph_queries() -> None:
    estimates = rebuild_all([observe("a", score=1.0) for _ in range(4)])
    later = T0 + timedelta(days=100)
    assert effective_mastery_map(estimates, later)[cid("a")] < estimates[cid("a")].mastery
    assert 0.0 <= confidence_map(estimates)[cid("a")] <= 1.0


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #

scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
sources = st.sampled_from(list(EvidenceSource))
observations = st.builds(
    lambda score, source, weight: Observation(
        concept_id=cid("target"),
        source=source,
        score=score,
        occurred_at=T0,
        weight_multiplier=weight,
    ),
    scores,
    sources,
    st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
)


@given(st.lists(observations, max_size=25))
def test_mastery_stays_a_probability(events: list[Observation]) -> None:
    estimate = rebuild(events)
    assert 0.0 <= estimate.mastery <= 1.0
    assert 0.0 <= estimate.confidence <= 1.0


@given(st.lists(observations, min_size=1, max_size=25))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_replay_order_does_not_change_the_result(events: list[Observation]) -> None:
    """The property that lets `mastery_states` be rebuilt from `evidence_events`.

    Without commutativity the materialised table and the append-only log would drift
    apart, and neither could be trusted as the source of truth.
    """
    forwards = rebuild(events)
    backwards = rebuild(list(reversed(events)))
    assert forwards.mastery == pytest.approx(backwards.mastery, abs=1e-9)
    assert forwards.confidence == pytest.approx(backwards.confidence, abs=1e-9)


@given(st.lists(observations, max_size=15), scores)
def test_a_better_result_never_lowers_mastery(history: list[Observation], score: float) -> None:
    """Monotonicity: scoring higher must never make the model think less of you."""
    base = rebuild(history)
    lower, _ = apply_observation(base, observe(score=score * 0.5))
    higher, _ = apply_observation(base, observe(score=score))
    assert higher.mastery >= lower.mastery - 1e-9


@given(st.lists(observations, min_size=1, max_size=20))
def test_confidence_never_decreases_with_more_evidence(events: list[Observation]) -> None:
    estimate = MasteryEstimate()
    previous = estimate.confidence
    for event in events:
        estimate, _ = apply_observation(estimate, event)
        assert estimate.confidence >= previous - 1e-9
        previous = estimate.confidence


@given(st.lists(observations, min_size=1, max_size=20), st.integers(min_value=0, max_value=3650))
def test_effective_mastery_never_exceeds_stored_mastery(
    events: list[Observation], days: int
) -> None:
    """Forgetting only ever removes; it must never manufacture knowledge."""
    estimate = rebuild(events)
    assert estimate.effective_mastery(T0 + timedelta(days=days)) <= estimate.mastery + 1e-9


@given(st.lists(observations, min_size=1, max_size=20))
def test_a_mastered_estimate_clears_the_threshold_it_claims(
    events: list[Observation],
) -> None:
    estimate = rebuild(events)
    assert estimate.is_mastered == (estimate.mastery >= MASTERY_THRESHOLD)
