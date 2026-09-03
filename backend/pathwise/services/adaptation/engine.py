"""The adaptation engine: restructuring a roadmap when evidence arrives.

The behaviour the whole product is for. A learner fails an assessment; the system
works out *why*, and changes the path in response, rather than marching them onward.

Every mutation here is produced by a rule, from evidence, deterministically. The rules
are stated once, in one place, and each carries the evidence that fired it — so a
revision is not merely applied but *justified*, and the justification survives in the
database for the learner to read later.

The core distinction, and the reason blame attribution exists:

* Failed a concept, and a **prerequisite is weak** → the problem is underneath.
  Insert remediation *before* the failed concept. Repeating the concept itself would
  fail again for the same reason.
* Failed a concept, and **prerequisites are solid** → the problem is here. Add
  practice on the concept itself. Sending them back to material they have already
  demonstrated is wasted time and reads as the system not paying attention.

Everything is proposed as data. Applying mutations to a stored roadmap is a separate,
persistent step; this module computes what should change and why.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final
from uuid import UUID

from pathwise.models.enums import MutationType, NodeType
from pathwise.services.knowledge.graph import BlameCandidate, KnowledgeGraph
from pathwise.services.knowledge.mastery import (
    MASTERY_THRESHOLD,
    MasteryEstimate,
)
from pathwise.services.roadmap.planner import RoadmapPlan
from pathwise.utils.text import count_noun, plural

#: A score at or below this counts as a failure worth adapting to. Above it the
#: learner is struggling but coping, and churning their roadmap on every imperfect
#: result would make the path feel unstable.
FAILURE_THRESHOLD: Final = 0.60

#: A blame candidate must score at least this to justify inserting remediation. Below
#: it the attribution is too weak to act on, and adding material on a hunch wastes
#: hours of someone's life.
MIN_BLAME_TO_ACT: Final = 0.25

#: Consecutive failures on one concept before the engine stops adding practice and
#: escalates to breaking the concept down.
STRUGGLE_ESCALATION_COUNT: Final = 3

#: Remediation is a focused intervention, not a full re-teach.
REMEDIATION_MINUTES: Final = 45
PRACTICE_MINUTES: Final = 30


@dataclass(frozen=True, slots=True)
class AdaptationTrigger:
    """The evidence that prompted an adaptation.

    Stored verbatim on the revision, so "why did my roadmap change?" is answered from
    the record rather than reconstructed.
    """

    kind: str
    concept_id: UUID
    concept_slug: str
    #: Display name. Learner-facing prose must never show a slug — "gradient-descent"
    #: is an identifier, "Gradient Descent" is what a person reads.
    concept_name: str = ""
    score: float | None = None
    attempt_number: int = 1
    detail: str = ""

    @property
    def display_name(self) -> str:
        """The name to show, falling back to the slug if none was supplied."""
        return self.concept_name or self.concept_slug

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "concept": self.concept_slug,
            "name": self.display_name,
            "score": self.score,
            "attempt": self.attempt_number,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RoadmapMutation:
    """One proposed structural change.

    ``evidence`` is what makes this auditable: the numbers the rule actually read.
    An explanation generated from a mutation is validated against these, so it cannot
    assert a figure the rule never saw.
    """

    type: MutationType
    concept_id: UUID
    concept_slug: str
    concept_name: str
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    #: Insert immediately before this concept. ``None`` means append.
    before_concept_id: UUID | None = None
    node_type: NodeType = NodeType.TOPIC
    estimated_minutes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": str(self.type),
            "concept": self.concept_slug,
            "name": self.concept_name,
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "before": str(self.before_concept_id) if self.before_concept_id else None,
            "estimated_minutes": self.estimated_minutes,
        }


@dataclass(frozen=True, slots=True)
class AdaptationResult:
    """Everything one adaptation produced."""

    trigger: AdaptationTrigger
    mutations: tuple[RoadmapMutation, ...]
    #: Blame ranking behind the decision, when one was computed. Kept so the
    #: explanation can name the prerequisite and cite its actual deficit.
    blame: tuple[BlameCandidate, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.mutations)

    @property
    def added_minutes(self) -> int:
        return sum(m.estimated_minutes for m in self.mutations)

    def citable_numbers(self) -> tuple[float, ...]:
        """Figures an explanation of this revision may quote."""
        numbers: list[float] = []
        if self.trigger.score is not None:
            numbers.append(round(self.trigger.score, 3))
            numbers.append(round(self.trigger.score * 100, 1))
        for mutation in self.mutations:
            numbers.append(float(mutation.estimated_minutes))
            numbers.append(round(mutation.estimated_minutes / 60, 1))
            for value in mutation.evidence.values():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    numbers.append(round(float(value), 3))
                    numbers.append(round(float(value) * 100, 1))
        return tuple(numbers)

    def as_revision_payload(self) -> dict[str, Any]:
        """The shape stored on a ``roadmap_revisions`` row."""
        return {
            "trigger": self.trigger.as_dict(),
            "mutations": [m.as_dict() for m in self.mutations],
            "blame": [
                {
                    "concept": str(b.concept_id),
                    "score": round(b.score, 3),
                    "deficit": round(b.deficit, 3),
                    "hops": b.hops,
                }
                for b in self.blame
            ],
        }


def adapt_to_failure(
    graph: KnowledgeGraph,
    plan: RoadmapPlan,
    trigger: AdaptationTrigger,
    mastery: Mapping[UUID, MasteryEstimate],
    *,
    now: datetime,
) -> AdaptationResult:
    """Decide how a roadmap should change after a poor result.

    This is the spec's worked example, implemented: a learner scores 48% on gradient
    descent, blame attribution finds the weak prerequisite, and an optimisation
    section is inserted *before* gradient descent rather than after it.
    """
    if trigger.score is not None and trigger.score > FAILURE_THRESHOLD:
        return AdaptationResult(trigger=trigger, mutations=())

    effective = {cid: est.effective_mastery(now) for cid, est in mastery.items()}
    confidence = {cid: est.confidence for cid, est in mastery.items()}

    blame = graph.blame_candidates(trigger.concept_id, effective, confidence=confidence, limit=3)
    in_plan = set(plan.concept_ids)

    # Repeated failure means the earlier interventions did not work. More of the same
    # will not either, so break the concept down instead of adding another exercise.
    if trigger.attempt_number >= STRUGGLE_ESCALATION_COUNT:
        return AdaptationResult(
            trigger=trigger,
            mutations=(_split(graph, trigger),),
            blame=blame,
        )

    actionable = [b for b in blame if b.score >= MIN_BLAME_TO_ACT]

    if actionable:
        # The problem is underneath. Repeating the failed concept would fail again
        # for exactly the same reason.
        return AdaptationResult(
            trigger=trigger,
            mutations=tuple(
                _remediation(graph, candidate, trigger, already_present=cid in in_plan)
                for candidate in actionable[:2]
                if (cid := candidate.concept_id) is not None
            ),
            blame=blame,
        )

    # Prerequisites are solid, so the difficulty is with this concept itself. Sending
    # the learner back to material they have already demonstrated would waste their
    # time and read as the system not paying attention.
    return AdaptationResult(
        trigger=trigger,
        mutations=(_practice(graph, trigger),),
        blame=blame,
    )


def adapt_to_mastery(
    graph: KnowledgeGraph,
    plan: RoadmapPlan,
    trigger: AdaptationTrigger,
    mastery: Mapping[UUID, MasteryEstimate],
    *,
    now: datetime,
) -> AdaptationResult:
    """Shorten a roadmap when a learner demonstrates more than expected.

    Adaptation runs both directions. A system that only ever adds material punishes
    the learner for being good at something, and the path grows monotonically no
    matter how well they do.
    """
    estimate = mastery.get(trigger.concept_id)
    if estimate is None:
        return AdaptationResult(trigger=trigger, mutations=())

    mutations: list[RoadmapMutation] = []

    for node in plan.nodes:
        if node.concept_id == trigger.concept_id:
            continue
        candidate = mastery.get(node.concept_id)
        if candidate is None:
            continue

        # Propagation may have lifted prerequisites of the demonstrated concept above
        # the bar; those are now removable.
        if candidate.is_skippable_at(now):
            mutations.append(
                RoadmapMutation(
                    type=MutationType.SKIP,
                    concept_id=node.concept_id,
                    concept_slug=node.slug,
                    concept_name=node.name,
                    reason=(
                        f"You have demonstrated {node.name} well enough that it no "
                        "longer needs its own step."
                    ),
                    evidence={
                        "mastery": round(candidate.effective_mastery(now), 3),
                        "confidence": round(candidate.confidence, 3),
                        "evidence_count": candidate.evidence_count,
                    },
                    estimated_minutes=-node.estimated_minutes,
                )
            )

    return AdaptationResult(trigger=trigger, mutations=tuple(mutations))


def adapt_to_review(
    graph: KnowledgeGraph,
    due_concept_ids: Sequence[UUID],
    mastery: Mapping[UUID, MasteryEstimate],
    *,
    now: datetime,
) -> AdaptationResult:
    """Schedule reviews for knowledge that has decayed."""
    mutations: list[RoadmapMutation] = []

    for concept_id in due_concept_ids:
        if concept_id not in graph:
            continue
        estimate = mastery.get(concept_id)
        if estimate is None:
            continue

        node = graph.node(concept_id)
        mutations.append(
            RoadmapMutation(
                type=MutationType.ADD_REVIEW,
                concept_id=concept_id,
                concept_slug=node.slug,
                concept_name=node.name,
                reason=(
                    f"{node.name} has not been practised recently and is due for a short review."
                ),
                evidence={
                    "current_mastery": round(estimate.effective_mastery(now), 3),
                    "peak_mastery": round(estimate.mastery, 3),
                },
                node_type=NodeType.REVIEW,
                estimated_minutes=PRACTICE_MINUTES,
            )
        )

    trigger = AdaptationTrigger(
        kind="review_due",
        concept_id=due_concept_ids[0] if due_concept_ids else UUID(int=0),
        concept_slug=graph.node(due_concept_ids[0]).slug if due_concept_ids else "",
        concept_name=graph.node(due_concept_ids[0]).name if due_concept_ids else "",
        detail=f"{len(mutations)} concept(s) due for review",
    )
    return AdaptationResult(trigger=trigger, mutations=tuple(mutations))


# --------------------------------------------------------------------------- #
# Mutation builders
# --------------------------------------------------------------------------- #


def _remediation(
    graph: KnowledgeGraph,
    candidate: BlameCandidate,
    trigger: AdaptationTrigger,
    *,
    already_present: bool,
) -> RoadmapMutation:
    """Insert focused work on a weak prerequisite, before the concept that failed."""
    node = graph.node(candidate.concept_id)
    return RoadmapMutation(
        type=MutationType.INSERT_REMEDIATION,
        concept_id=candidate.concept_id,
        concept_slug=node.slug,
        concept_name=node.name,
        reason=(
            f"{node.name} underpins {trigger.concept_slug}, and your work there "
            "suggests it is the gap."
            if not already_present
            else f"{node.name} is already in your path but needs attention before "
            f"{trigger.concept_slug}."
        ),
        evidence={
            "blame_score": round(candidate.score, 3),
            "prerequisite_mastery": round(candidate.mastery, 3),
            "deficit": round(candidate.deficit, 3),
            "hops_from_failure": candidate.hops,
        },
        # The insertion point is the whole point: remediation goes *before* the
        # concept that failed, so the learner arrives at it prepared.
        before_concept_id=trigger.concept_id,
        node_type=NodeType.PRACTICE,
        estimated_minutes=REMEDIATION_MINUTES,
    )


def _practice(graph: KnowledgeGraph, trigger: AdaptationTrigger) -> RoadmapMutation:
    """Add practice on the concept itself, when its prerequisites are sound."""
    node = graph.node(trigger.concept_id)
    return RoadmapMutation(
        type=MutationType.ADD_PRACTICE,
        concept_id=trigger.concept_id,
        concept_slug=node.slug,
        concept_name=node.name,
        reason=(
            f"Your prerequisites for {node.name} are solid, so the difficulty is with "
            "this topic itself rather than anything beneath it."
        ),
        evidence={"score": round(trigger.score, 3) if trigger.score is not None else 0.0},
        before_concept_id=None,
        node_type=NodeType.PRACTICE,
        estimated_minutes=PRACTICE_MINUTES,
    )


def _split(graph: KnowledgeGraph, trigger: AdaptationTrigger) -> RoadmapMutation:
    """Break a concept down after repeated failure.

    At this point more practice on the same material has already been tried and has
    not worked. Repeating it a third time is the definition of not adapting.
    """
    node = graph.node(trigger.concept_id)
    return RoadmapMutation(
        type=MutationType.SPLIT_NODE,
        concept_id=trigger.concept_id,
        concept_slug=node.slug,
        concept_name=node.name,
        reason=(
            f"This is attempt {trigger.attempt_number} at {node.name}. Rather than "
            "repeating it, it is worth working through its objectives one at a time."
        ),
        evidence={
            "attempts": trigger.attempt_number,
            "score": round(trigger.score, 3) if trigger.score is not None else 0.0,
            "objectives": len(node.objective_ids),
        },
        node_type=NodeType.PRACTICE,
        estimated_minutes=REMEDIATION_MINUTES,
    )


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #


def explain(result: AdaptationResult) -> str:
    """A deterministic explanation of a revision.

    Reproduces the shape of the spec's example, built entirely from the trigger and
    the mutations' recorded evidence. Used as the fallback when generation fails, and
    as the reference the generated version is checked against.
    """
    if not result.changed:
        return "Your roadmap is unchanged."

    trigger = result.trigger
    lead = ""
    if trigger.score is not None:
        lead = f"You scored {trigger.score * 100:.0f}% on {trigger.display_name}. "

    first = result.mutations[0]

    if first.type is MutationType.INSERT_REMEDIATION:
        return (
            f"{lead}{first.concept_name} is what it builds on, and that looks like "
            f"where the gap is. A {first.estimated_minutes}-minute section on it now "
            f"comes first, so you reach {trigger.display_name} prepared."
        )

    if first.type is MutationType.ADD_PRACTICE:
        return (
            f"{lead}Everything it builds on is solid, so rather than sending you "
            f"backwards there is {first.estimated_minutes} minutes of extra practice "
            f"on {first.concept_name} itself."
        )

    if first.type is MutationType.SPLIT_NODE:
        attempts = first.evidence.get("attempts", trigger.attempt_number)
        return (
            f"{lead}That is attempt {attempts}. {first.concept_name} has been broken "
            "into its individual objectives so you can work through them one at a time."
        )

    if first.type is MutationType.SKIP:
        removed = len(result.mutations)
        return (
            f"You have demonstrated enough that {count_noun(removed, 'step')} "
            f"{plural(removed, 'has')} been removed from your path."
        )

    if first.type is MutationType.ADD_REVIEW:
        return (
            f"{len(result.mutations)} topic(s) have not been practised in a while and "
            "have short reviews scheduled."
        )

    return f"{lead}Your roadmap has been updated."


def summarise(result: AdaptationResult) -> dict[str, Any]:
    """The payload behind a "your roadmap changed" notification."""
    return {
        "changed": result.changed,
        "trigger": result.trigger.as_dict(),
        "mutation_count": len(result.mutations),
        "added_minutes": result.added_minutes,
        "explanation": explain(result),
        "mutations": [m.as_dict() for m in result.mutations],
    }


def mastery_is_sufficient(estimate: MasteryEstimate | None, now: datetime) -> bool:
    """Whether a concept currently counts as known, for adaptation purposes."""
    return estimate is not None and estimate.effective_mastery(now) >= MASTERY_THRESHOLD
