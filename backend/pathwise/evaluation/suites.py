"""Evaluation suites.

Each suite runs a real engine over hand-labelled cases and scores what comes back.
The cases are the interesting artifact: they encode what a competent tutor *should*
answer for a given learner, which is the only way to tell whether an algorithm change
made the product better or merely different.

The suites deliberately exercise the deterministic engines rather than the model.
Those engines make every decision a learner experiences — what to study, what was
skipped, what to blame — so they are what quality actually depends on. A prompt
regression shows up as a validation-failure rate in `llm_calls`; an *engine*
regression shows up nowhere unless it is measured here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pathwise.api.errors import ValidationError
from pathwise.evaluation.scorers import (
    AllKnown,
    BooleanFlag,
    CaseResult,
    OrderedBefore,
    Score,
    Scorer,
    SetExcludes,
    SetRecall,
    SuiteResult,
    TopKMatch,
    WithinRange,
)
from pathwise.models.enums import EvidenceSource
from pathwise.services.adaptation.engine import AdaptationTrigger, adapt_to_failure
from pathwise.services.decision.engine import LearnerContext, recommend_next
from pathwise.services.knowledge.graph import KnowledgeGraph
from pathwise.services.knowledge.mastery import MasteryEstimate, Observation, rebuild
from pathwise.services.knowledge.seed import (
    build_graph_from_corpus,
    concept_id_for,
    load_corpus,
)
from pathwise.services.roadmap.planner import plan_roadmap

DATASETS_DIR = Path(__file__).resolve().parents[2] / "evals" / "datasets"

#: Fixed reference time. Evaluation must not drift because it ran on a different day —
#: mastery decay is time-dependent, so a moving clock would silently change results.
EVAL_NOW = datetime(2026, 9, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One labelled scenario."""

    case_id: str
    inputs: Mapping[str, Any]
    expected: Mapping[str, Any]
    #: Why this case exists. Read when it fails, which is the moment it matters.
    note: str = ""


@dataclass(frozen=True, slots=True)
class Suite:
    """A dataset plus the function that runs it and the scorers that judge it."""

    name: str
    description: str
    runner: Callable[[EvalCase, KnowledgeGraph], Mapping[str, Any]]
    scorers: Sequence[Scorer]

    def run(self, cases: Iterable[EvalCase], graph: KnowledgeGraph) -> SuiteResult:
        results: list[CaseResult] = []

        for case in cases:
            try:
                actual = self.runner(case, graph)
            except Exception as exc:
                # A crash is a failure of the case, not of the run. Recording it as a
                # scored failure keeps the suite comparable across versions instead of
                # aborting halfway and reporting nothing.
                results.append(
                    CaseResult(
                        case_id=case.case_id,
                        scores=(Score("error", 0.0, False, f"{type(exc).__name__}: {exc}"),),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            results.append(
                CaseResult(
                    case_id=case.case_id,
                    scores=tuple(scorer(actual, case.expected) for scorer in self.scorers),
                    actual=actual,
                )
            )

        return SuiteResult(suite=self.name, cases=tuple(results))


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _mastery_from(spec: Mapping[str, Any]) -> dict[Any, MasteryEstimate]:
    """Build mastery estimates from a case's `{slug: {score, times}}` block."""
    estimates: dict[Any, MasteryEstimate] = {}
    for slug, entry in (spec or {}).items():
        score = float(entry["score"]) if isinstance(entry, dict) else float(entry)
        times = int(entry.get("times", 6)) if isinstance(entry, dict) else 6
        concept_id = concept_id_for(slug)
        estimates[concept_id] = rebuild(
            [
                Observation(concept_id, EvidenceSource.ASSESSMENT, score, EVAL_NOW)
                for _ in range(times)
            ]
        )
    return estimates


def _slug(graph: KnowledgeGraph, concept_id: Any) -> str:
    return graph.node(concept_id).slug


# --------------------------------------------------------------------------- #
# Suite runners
# --------------------------------------------------------------------------- #


def run_planner_case(case: EvalCase, graph: KnowledgeGraph) -> Mapping[str, Any]:
    """Plan a roadmap for a labelled learner."""
    goals = [concept_id_for(slug) for slug in case.inputs["goals"]]
    mastery = _mastery_from(case.inputs.get("mastery", {}))

    plan = plan_roadmap(
        graph,
        goals,
        mastery,
        hours_per_week=float(case.inputs.get("hours_per_week", 8.0)),
        now=EVAL_NOW,
    )
    return {
        "steps": list(plan.slugs),
        "skipped": [s.slug for s in plan.skipped],
        "compressed": [c.slug for c in plan.compressed],
        "total_hours": plan.pacing.total_minutes / 60,
        "estimated_weeks": plan.pacing.estimated_weeks,
        "is_ordered": _ordering_holds(graph, plan),
    }


def _ordering_holds(graph: KnowledgeGraph, plan: Any) -> bool:
    position = {node.concept_id: node.order_index for node in plan.nodes}
    return all(position[source] < position[target] for source, target, _ in plan.edges)


def run_decision_case(case: EvalCase, graph: KnowledgeGraph) -> Mapping[str, Any]:
    """Ask the decision engine what a labelled learner should do next."""
    goals = tuple(concept_id_for(slug) for slug in case.inputs["goals"])
    mastery = _mastery_from(case.inputs.get("mastery", {}))
    plan = plan_roadmap(graph, list(goals), mastery, now=EVAL_NOW)

    trace = recommend_next(
        graph,
        plan,
        LearnerContext(
            mastery=mastery,
            goal_concept_ids=goals,
            hours_per_week=float(case.inputs.get("hours_per_week", 8.0)),
            remediation_targets=frozenset(
                concept_id_for(slug) for slug in case.inputs.get("remediation_targets", [])
            ),
            last_domain=case.inputs.get("last_domain"),
        ),
        now=EVAL_NOW,
        limit=5,
    )

    ranked = [c.slug for c in (trace.recommended, *trace.alternatives) if c]
    return {
        "ranked": ranked,
        "deciding_factor": trace.deciding_factor.name if trace.deciding_factor else None,
        "has_recommendation": trace.has_recommendation,
    }


def run_blame_case(case: EvalCase, graph: KnowledgeGraph) -> Mapping[str, Any]:
    """Ask blame attribution why a labelled learner failed a concept."""
    mastery = _mastery_from(case.inputs.get("mastery", {}))
    effective = {cid: est.effective_mastery(EVAL_NOW) for cid, est in mastery.items()}
    confidence = {cid: est.confidence for cid, est in mastery.items()}

    candidates = graph.blame_candidates(
        concept_id_for(case.inputs["failed_concept"]),
        effective,
        confidence=confidence,
        max_depth=int(case.inputs.get("max_depth", 3)),
        limit=5,
    )
    return {"blamed": [_slug(graph, c.concept_id) for c in candidates]}


def run_adaptation_case(case: EvalCase, graph: KnowledgeGraph) -> Mapping[str, Any]:
    """Adapt a roadmap to a labelled failure."""
    goals = [concept_id_for(slug) for slug in case.inputs["goals"]]
    mastery = _mastery_from(case.inputs.get("mastery", {}))
    plan = plan_roadmap(graph, goals, mastery, now=EVAL_NOW)

    failed = case.inputs["failed_concept"]
    result = adapt_to_failure(
        graph,
        plan,
        AdaptationTrigger(
            kind="assessment_failed",
            concept_id=concept_id_for(failed),
            concept_slug=failed,
            concept_name=graph.by_slug(failed).name,
            score=float(case.inputs["score"]),
            attempt_number=int(case.inputs.get("attempt", 1)),
        ),
        mastery,
        now=EVAL_NOW,
    )

    from pathwise.services.adaptation.engine import explain

    return {
        "mutation_types": [str(m.type) for m in result.mutations],
        "targets": [m.concept_slug for m in result.mutations],
        "changed": result.changed,
        "explanation": explain(result),
        "citable_numbers": list(result.citable_numbers()),
    }


# --------------------------------------------------------------------------- #
# Suite registry
# --------------------------------------------------------------------------- #


def build_suites(graph: KnowledgeGraph) -> dict[str, Suite]:
    """Every suite, bound to a graph.

    Bound rather than global because the grounding scorers need the real slug set —
    a check that "no concept was invented" is meaningless without knowing which
    concepts exist.
    """
    known_slugs = frozenset(graph.node(cid).slug for cid in graph.node_ids)

    return {
        "planner": Suite(
            name="planner",
            description="Roadmaps contain what the goal requires, correctly ordered.",
            runner=run_planner_case,
            scorers=[
                SetRecall("steps", "must_include", name="required_coverage"),
                SetExcludes("steps", "must_exclude", name="no_irrelevant_steps"),
                SetRecall("skipped", "must_skip", name="skip_accuracy"),
                OrderedBefore("steps"),
                BooleanFlag("is_ordered", name="prerequisite_ordering"),
                AllKnown("steps", known_slugs, name="concepts_grounded"),
                WithinRange("estimated_weeks", minimum=0.1, name="pacing_sane"),
            ],
        ),
        "decision": Suite(
            name="decision",
            description="The next step matches what a tutor would choose.",
            runner=run_decision_case,
            scorers=[
                TopKMatch("ranked", "expected_next", k=1, name="top_1"),
                TopKMatch("ranked", "expected_next", k=3, name="top_3"),
                SetExcludes("ranked", "must_not_suggest", name="no_bad_suggestions"),
                BooleanFlag("has_recommendation", name="produced_a_recommendation"),
            ],
        ),
        "blame": Suite(
            name="blame",
            description="Struggle is attributed to the prerequisite actually missing.",
            runner=run_blame_case,
            scorers=[
                TopKMatch("blamed", "expected_cause", k=1, name="top_1"),
                TopKMatch("blamed", "expected_cause", k=3, name="top_3"),
                # The failure mode that matters: blaming the nearest prerequisite
                # regardless of whether the learner has demonstrated it.
                SetExcludes("blamed", "must_not_blame", name="no_false_accusation"),
            ],
        ),
        "adaptation": Suite(
            name="adaptation",
            description="Roadmap changes match the intervention a tutor would make.",
            runner=run_adaptation_case,
            scorers=[
                SetRecall("mutation_types", "expected_mutations", name="mutation_match"),
                SetRecall("targets", "expected_targets", name="target_match"),
                SetExcludes("targets", "must_not_target", name="no_wrong_target"),
                AllKnown("targets", known_slugs, name="concepts_grounded"),
            ],
        ),
    }


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


def load_cases(suite_name: str, directory: Path | None = None) -> tuple[EvalCase, ...]:
    """Load one suite's cases from JSONL.

    JSONL rather than a single JSON document so a case can be added in a one-line
    diff, and a malformed one names its own line number.
    """
    directory = directory or DATASETS_DIR
    path = directory / f"{suite_name}.jsonl"

    if not path.is_file():
        raise ValidationError(f"No dataset for suite '{suite_name}'.", expected_path=str(path))

    cases: list[EvalCase] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("//"):
            continue
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"{path.name} line {number} is not valid JSON: {exc}",
                file=path.name,
                line=number,
            ) from exc

        cases.append(
            EvalCase(
                case_id=raw.get("id", f"{suite_name}-{number}"),
                inputs=raw.get("inputs", {}),
                expected=raw.get("expected", {}),
                note=raw.get("note", ""),
            )
        )

    if not cases:
        raise ValidationError(f"Dataset '{suite_name}' contains no cases.", file=path.name)

    return tuple(cases)


def load_graph() -> KnowledgeGraph:
    """The knowledge graph the suites evaluate against."""
    return build_graph_from_corpus(load_corpus())
