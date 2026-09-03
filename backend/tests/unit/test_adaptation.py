"""Adaptation engine tests.

This is the behaviour the product exists for: a learner struggles, and the path
changes in response rather than marching them onward.

The distinction these tests defend is the one blame attribution was built to make.
Failing a concept because a *prerequisite* is weak and failing it because *the
concept itself* is hard call for opposite responses — go backwards, or stay and
practise. Getting it wrong either wastes hours on material already mastered, or
repeats a topic that will fail again for the same reason.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from pathwise.ai.validators import grounded_in_trace
from pathwise.models.enums import EvidenceSource, MutationType, NodeType, RelationType
from pathwise.services.adaptation.engine import (
    FAILURE_THRESHOLD,
    STRUGGLE_ESCALATION_COUNT,
    AdaptationResult,
    AdaptationTrigger,
    adapt_to_failure,
    adapt_to_mastery,
    adapt_to_review,
    explain,
    summarise,
)
from pathwise.services.knowledge.graph import GraphEdge, GraphNode, KnowledgeGraph
from pathwise.services.knowledge.mastery import MasteryEstimate, Observation, rebuild
from pathwise.services.knowledge.seed import (
    build_graph_from_corpus,
    concept_id_for,
    load_corpus,
)
from pathwise.services.roadmap.planner import plan_roadmap

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def cid(slug: str) -> UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, slug)


def node(slug: str, difficulty: int = 3) -> GraphNode:
    return GraphNode(
        id=cid(slug),
        slug=slug,
        name=slug.replace("-", " ").title(),
        difficulty=difficulty,
        estimated_minutes=120,
        domain="test",
        objective_ids=("lo-1", "lo-2"),
    )


def prereq(source: str, target: str) -> GraphEdge:
    return GraphEdge(cid(source), cid(target), RelationType.PREREQUISITE_OF, 1.0)


def estimate(slug: str, score: float, times: int = 6, at: datetime = NOW) -> MasteryEstimate:
    return rebuild(
        [Observation(cid(slug), EvidenceSource.ASSESSMENT, score, at) for _ in range(times)]
    )


def failure(slug: str = "target", score: float = 0.4, attempt: int = 1) -> AdaptationTrigger:
    return AdaptationTrigger(
        kind="assessment_failed",
        concept_id=cid(slug),
        concept_slug=slug,
        concept_name=slug.replace("-", " ").title(),
        score=score,
        attempt_number=attempt,
    )


@pytest.fixture
def chain() -> KnowledgeGraph:
    """foundation -> support -> target"""
    return KnowledgeGraph(
        [node("foundation", 1), node("support", 2), node("target", 3)],
        [prereq("foundation", "support"), prereq("support", "target")],
    )


@pytest.fixture
def plan(chain: KnowledgeGraph):
    return plan_roadmap(chain, [cid("target")], now=NOW)


@pytest.fixture(scope="module")
def seed_graph() -> KnowledgeGraph:
    return build_graph_from_corpus(load_corpus())


# --------------------------------------------------------------------------- #
# When to adapt at all
# --------------------------------------------------------------------------- #


def test_a_pass_changes_nothing(chain: KnowledgeGraph, plan) -> None:
    """Churning a roadmap on every imperfect result makes the path feel unstable."""
    result = adapt_to_failure(chain, plan, failure(score=0.85), {}, now=NOW)
    assert not result.changed
    assert explain(result) == "Your roadmap is unchanged."


def test_a_borderline_result_changes_nothing(chain: KnowledgeGraph, plan) -> None:
    result = adapt_to_failure(chain, plan, failure(score=FAILURE_THRESHOLD + 0.01), {}, now=NOW)
    assert not result.changed


def test_a_clear_failure_adapts(chain: KnowledgeGraph, plan) -> None:
    assert adapt_to_failure(chain, plan, failure(score=0.3), {}, now=NOW).changed


# --------------------------------------------------------------------------- #
# The core distinction
# --------------------------------------------------------------------------- #


def test_a_weak_prerequisite_inserts_remediation(chain: KnowledgeGraph, plan) -> None:
    """The problem is underneath, so go there."""
    mastery = {
        cid("foundation"): estimate("foundation", 1.0),
        cid("support"): estimate("support", 0.2),
    }
    result = adapt_to_failure(chain, plan, failure(), mastery, now=NOW)

    assert result.mutations[0].type is MutationType.INSERT_REMEDIATION
    assert result.mutations[0].concept_slug == "support"


def test_remediation_is_inserted_before_the_failed_concept(chain: KnowledgeGraph, plan) -> None:
    """Placing it after would have the learner reach the concept unprepared again —
    the ordering *is* the intervention."""
    mastery = {cid("support"): estimate("support", 0.2)}
    result = adapt_to_failure(chain, plan, failure(), mastery, now=NOW)
    assert result.mutations[0].before_concept_id == cid("target")


def test_solid_prerequisites_add_practice_on_the_concept_itself(
    chain: KnowledgeGraph, plan
) -> None:
    """Sending someone back to material they have demonstrated wastes their time and
    reads as the system not paying attention."""
    mastery = {
        cid("foundation"): estimate("foundation", 1.0),
        cid("support"): estimate("support", 1.0),
    }
    result = adapt_to_failure(chain, plan, failure(), mastery, now=NOW)

    assert result.mutations[0].type is MutationType.ADD_PRACTICE
    assert result.mutations[0].concept_slug == "target"


def test_weak_attribution_does_not_trigger_remediation(chain: KnowledgeGraph, plan) -> None:
    """Adding hours of material on a hunch is worse than adding none."""
    # Prerequisites met, so nothing is meaningfully to blame.
    mastery = {
        cid("foundation"): estimate("foundation", 0.95),
        cid("support"): estimate("support", 0.95),
    }
    result = adapt_to_failure(chain, plan, failure(), mastery, now=NOW)
    assert all(m.type is not MutationType.INSERT_REMEDIATION for m in result.mutations)


def test_at_most_two_remediations_are_proposed(chain: KnowledgeGraph, plan) -> None:
    """A revision that inserts five new topics is not an adaptation, it is a rewrite."""
    result = adapt_to_failure(chain, plan, failure(), {}, now=NOW)
    assert len(result.mutations) <= 2


# --------------------------------------------------------------------------- #
# Escalation
# --------------------------------------------------------------------------- #


def test_repeated_failure_escalates_to_breaking_the_concept_down(
    chain: KnowledgeGraph, plan
) -> None:
    """More practice has already been tried and did not work. Repeating it a third
    time is the definition of not adapting."""
    mastery = {cid("support"): estimate("support", 0.2)}
    result = adapt_to_failure(
        chain, plan, failure(attempt=STRUGGLE_ESCALATION_COUNT), mastery, now=NOW
    )
    assert result.mutations[0].type is MutationType.SPLIT_NODE


def test_a_first_failure_does_not_escalate(chain: KnowledgeGraph, plan) -> None:
    mastery = {cid("support"): estimate("support", 0.2)}
    result = adapt_to_failure(chain, plan, failure(attempt=1), mastery, now=NOW)
    assert result.mutations[0].type is not MutationType.SPLIT_NODE


def test_escalation_records_the_attempt_count(chain: KnowledgeGraph, plan) -> None:
    result = adapt_to_failure(chain, plan, failure(attempt=4), {}, now=NOW)
    assert result.mutations[0].evidence["attempts"] == 4


# --------------------------------------------------------------------------- #
# Adapting in the shortening direction
# --------------------------------------------------------------------------- #


def test_demonstrated_material_is_removed(chain: KnowledgeGraph) -> None:
    """A system that only ever adds material punishes a learner for being good at
    something, and the path grows monotonically no matter how well they do."""
    mastery = {
        cid("target"): estimate("target", 1.0),
        cid("foundation"): estimate("foundation", 1.0, times=8),
    }
    plan = plan_roadmap(chain, [cid("target")], now=NOW)
    trigger = AdaptationTrigger(
        kind="mastery_gained",
        concept_id=cid("target"),
        concept_slug="target",
        concept_name="Target",
    )
    result = adapt_to_mastery(chain, plan, trigger, mastery, now=NOW)

    assert any(m.type is MutationType.SKIP for m in result.mutations)
    assert result.added_minutes < 0  # the path got shorter


def test_thin_evidence_does_not_remove_material(chain: KnowledgeGraph) -> None:
    mastery = {
        cid("target"): estimate("target", 1.0),
        cid("foundation"): estimate("foundation", 1.0, times=1),
    }
    plan = plan_roadmap(chain, [cid("target")], now=NOW)
    trigger = AdaptationTrigger("mastery_gained", cid("target"), "target", "Target")
    assert adapt_to_mastery(chain, plan, trigger, mastery, now=NOW).mutations == ()


def test_decayed_knowledge_is_scheduled_for_review(chain: KnowledgeGraph) -> None:
    long_ago = NOW - timedelta(days=500)
    mastery = {cid("foundation"): estimate("foundation", 1.0, times=4, at=long_ago)}
    result = adapt_to_review(chain, [cid("foundation")], mastery, now=NOW)

    assert result.mutations[0].type is MutationType.ADD_REVIEW
    assert result.mutations[0].node_type is NodeType.REVIEW


def test_a_review_records_how_far_it_decayed(chain: KnowledgeGraph) -> None:
    long_ago = NOW - timedelta(days=500)
    mastery = {cid("foundation"): estimate("foundation", 1.0, times=4, at=long_ago)}
    evidence = adapt_to_review(chain, [cid("foundation")], mastery, now=NOW).mutations[0].evidence
    assert evidence["current_mastery"] < evidence["peak_mastery"]


# --------------------------------------------------------------------------- #
# Evidence and auditability
# --------------------------------------------------------------------------- #


def test_every_mutation_carries_the_numbers_that_produced_it(chain: KnowledgeGraph, plan) -> None:
    """A revision must be justified, not merely applied."""
    mastery = {cid("support"): estimate("support", 0.2)}
    result = adapt_to_failure(chain, plan, failure(), mastery, now=NOW)
    evidence = result.mutations[0].evidence

    assert "blame_score" in evidence
    assert "prerequisite_mastery" in evidence
    assert evidence["hops_from_failure"] >= 1


def test_the_revision_payload_matches_the_stored_shape(chain: KnowledgeGraph, plan) -> None:
    """These three keys are what `roadmap_revisions` stores, and what answers "why
    did my roadmap change?" months later."""
    mastery = {cid("support"): estimate("support", 0.2)}
    payload = adapt_to_failure(chain, plan, failure(), mastery, now=NOW).as_revision_payload()

    assert set(payload) == {"trigger", "mutations", "blame"}
    assert payload["trigger"]["score"] == 0.4  # type: ignore[index]
    assert payload["mutations"]


def test_the_summary_is_notification_ready(chain: KnowledgeGraph, plan) -> None:
    summary = summarise(adapt_to_failure(chain, plan, failure(), {}, now=NOW))
    assert summary["changed"] is True
    assert summary["explanation"]
    assert summary["mutation_count"] >= 1


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #


def test_the_explanation_names_the_concept_not_the_slug(chain: KnowledgeGraph, plan) -> None:
    """ "gradient-descent" is an identifier; "Gradient Descent" is what a person reads."""
    mastery = {cid("support"): estimate("support", 0.2)}
    text = explain(adapt_to_failure(chain, plan, failure(), mastery, now=NOW))
    assert "Target" in text
    assert "target-" not in text  # no kebab-case leaking into learner-facing prose


def test_the_explanation_only_cites_figures_from_its_own_evidence(
    chain: KnowledgeGraph, plan
) -> None:
    """Both halves of the explainability claim, checked against each other. If the
    deterministic prose failed its own guard, the guard would be miscalibrated and
    would reject good generated output too."""
    mastery = {cid("support"): estimate("support", 0.2)}
    result = adapt_to_failure(chain, plan, failure(), mastery, now=NOW)
    assert grounded_in_trace(result.citable_numbers())(explain(result)).is_valid


def test_an_invented_score_is_rejected(chain: KnowledgeGraph, plan) -> None:
    result = adapt_to_failure(chain, plan, failure(score=0.48), {}, now=NOW)
    validate = grounded_in_trace(result.citable_numbers())
    assert not validate("You scored 91% and so this changed.").is_valid


def test_nothing_to_explain_when_nothing_changed() -> None:
    result = AdaptationResult(trigger=failure(score=0.9), mutations=())
    assert "unchanged" in explain(result)


# --------------------------------------------------------------------------- #
# The spec's worked example, on the real graph
# --------------------------------------------------------------------------- #


def test_the_spec_scenario_end_to_end(seed_graph: KnowledgeGraph) -> None:
    """ "You scored 48% on the gradient descent assessment. Your calculus performance
    is strong, so I've added a short optimization section before continuing."
    """
    strong = (
        "functions-and-graphs",
        "limits-and-continuity",
        "derivatives",
        "partial-derivatives",
        "chain-rule",
        "gradients-and-jacobians",
        "vectors-and-spaces",
    )
    mastery = {
        concept_id_for(slug): rebuild(
            [
                Observation(concept_id_for(slug), EvidenceSource.ASSESSMENT, 1.0, NOW)
                for _ in range(6)
            ]
        )
        for slug in strong
    }
    weak = concept_id_for("optimization-fundamentals")
    mastery[weak] = rebuild(
        [Observation(weak, EvidenceSource.ASSESSMENT, 0.25, NOW) for _ in range(3)]
    )

    plan = plan_roadmap(seed_graph, [concept_id_for("backpropagation")], mastery, now=NOW)
    trigger = AdaptationTrigger(
        kind="assessment_failed",
        concept_id=concept_id_for("gradient-descent"),
        concept_slug="gradient-descent",
        concept_name="Gradient Descent",
        score=0.48,
    )
    result = adapt_to_failure(seed_graph, plan, trigger, mastery, now=NOW)

    # It found the actual gap, not the strong calculus above it.
    assert result.mutations[0].concept_slug == "optimization-fundamentals"
    assert result.mutations[0].type is MutationType.INSERT_REMEDIATION
    # And placed it before the concept that failed.
    assert result.mutations[0].before_concept_id == concept_id_for("gradient-descent")

    text = explain(result)
    assert "48%" in text
    assert "Optimisation Fundamentals" in text
    assert grounded_in_trace(result.citable_numbers())(text).is_valid


def test_strong_calculus_is_not_blamed(seed_graph: KnowledgeGraph) -> None:
    """The failure mode this guards: blaming the nearest prerequisite regardless of
    whether the learner has actually demonstrated it."""
    strong = ("derivatives", "chain-rule", "gradients-and-jacobians", "partial-derivatives")
    mastery = {
        concept_id_for(slug): rebuild(
            [
                Observation(concept_id_for(slug), EvidenceSource.ASSESSMENT, 1.0, NOW)
                for _ in range(6)
            ]
        )
        for slug in strong
    }
    weak = concept_id_for("optimization-fundamentals")
    mastery[weak] = rebuild(
        [Observation(weak, EvidenceSource.ASSESSMENT, 0.2, NOW) for _ in range(3)]
    )

    plan = plan_roadmap(seed_graph, [concept_id_for("backpropagation")], mastery, now=NOW)
    trigger = AdaptationTrigger(
        kind="assessment_failed",
        concept_id=concept_id_for("gradient-descent"),
        concept_slug="gradient-descent",
        concept_name="Gradient Descent",
        score=0.45,
    )
    blamed = {
        m.concept_slug
        for m in adapt_to_failure(seed_graph, plan, trigger, mastery, now=NOW).mutations
    }
    assert "derivatives" not in blamed
    assert "chain-rule" not in blamed
