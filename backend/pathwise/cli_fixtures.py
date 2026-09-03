"""Generate frontend fixtures from the real engines.

    python -m pathwise.cli_fixtures > ../frontend/src/lib/fixtures.ts

The frontend is built against fixtures until the API routes land. Writing those by
hand would mean designing against prettier data than the system produces — which is
how a UI ends up unable to display the real thing, discovering at integration time
that a rationale is longer than the card or that a plan has 39 steps rather than 6.

So the fixtures are *generated* by running the actual planner, decision engine and
recommender over the actual seed graph. Regenerate after any change to those engines
and the frontend keeps telling the truth.
"""

from __future__ import annotations

import json
import pathlib
from datetime import UTC, datetime
from typing import Any

from pathwise.models.enums import EvidenceSource
from pathwise.services.adaptation.engine import AdaptationTrigger, adapt_to_failure, explain
from pathwise.services.decision.engine import LearnerContext, recommend_next
from pathwise.services.knowledge.mastery import MasteryEstimate, Observation, rebuild
from pathwise.services.knowledge.seed import (
    build_graph_from_corpus,
    concept_id_for,
    load_corpus,
)
from pathwise.services.recommendation.recommender import (
    RecommendationContext,
    recommend,
)
from pathwise.services.recommendation.recommender import (
    fallback_explanation as resource_reason,
)
from pathwise.services.resource.catalogue import load_catalogue
from pathwise.services.roadmap.planner import plan_roadmap

NOW = datetime(2026, 9, 1, tzinfo=UTC)
GOAL = "backpropagation"

#: The scenario the whole spec is written around: knows Python and basic calculus,
#: eight hours a week, aiming at the maths behind training a network.
DEMONSTRATED = {
    "programming-basics": (1.0, 8),
    "python-syntax-and-types": (1.0, 8),
    "python-control-flow": (1.0, 8),
    "python-functions": (1.0, 8),
    "python-data-structures": (1.0, 7),
    "functions-and-graphs": (1.0, 7),
    "limits-and-continuity": (1.0, 6),
    "derivatives": (0.92, 6),
    "vectors-and-spaces": (1.0, 6),
    "matrix-operations": (0.88, 5),
    # Measured once, well — enough to compress the step, not to delete it.
    "numpy-fundamentals": (1.0, 1),
    # The gap. This is what blame attribution should find.
    "partial-derivatives": (0.35, 3),
}


def _mastery() -> dict[Any, MasteryEstimate]:
    return {
        concept_id_for(slug): rebuild(
            [
                Observation(concept_id_for(slug), EvidenceSource.ASSESSMENT, score, NOW)
                for _ in range(times)
            ]
        )
        for slug, (score, times) in DEMONSTRATED.items()
    }


def _estimate_json(estimate: MasteryEstimate | None) -> dict[str, Any] | None:
    if estimate is None:
        return None
    return {
        "alpha": round(estimate.alpha, 4),
        "beta": round(estimate.beta, 4),
        "mastery": round(estimate.effective_mastery(NOW), 4),
        "confidence": round(estimate.confidence, 4),
        "evidenceCount": estimate.evidence_count,
    }


def build() -> dict[str, Any]:
    graph = build_graph_from_corpus(load_corpus())
    catalogue = load_catalogue()
    mastery = _mastery()
    goal_id = concept_id_for(GOAL)

    plan = plan_roadmap(graph, [goal_id], mastery, hours_per_week=8.0, now=NOW)
    compressed_by_id = {c.concept_id: c for c in plan.compressed}

    trace = recommend_next(
        graph,
        plan,
        LearnerContext(
            mastery=mastery,
            goal_concept_ids=(goal_id,),
            hours_per_week=8.0,
            last_domain="mathematics",
        ),
        now=NOW,
        limit=3,
    )

    nodes = [
        {
            "conceptId": str(node.concept_id),
            "slug": node.slug,
            "name": node.name,
            "domain": node.domain,
            "difficulty": node.difficulty,
            "estimatedMinutes": node.estimated_minutes,
            "status": (
                "recommended"
                if trace.recommended and node.concept_id == trace.recommended.concept_id
                else str(node.status)
            ),
            "nodeType": str(node.node_type),
            "orderIndex": node.order_index,
            "dependsOn": [str(dep) for dep in node.depends_on],
            "rationale": _rationale(graph, node, compressed_by_id),
            "mastery": _estimate_json(mastery.get(node.concept_id)),
        }
        for node in plan.nodes
    ]

    next_step = None
    if trace.recommended:
        candidate = trace.recommended
        next_step = {
            "slug": candidate.slug,
            "name": candidate.name,
            "kind": candidate.kind,
            "estimatedMinutes": candidate.estimated_minutes,
            "difficulty": candidate.difficulty,
            "score": round(candidate.score, 4),
            "decidingFactor": trace.deciding_factor.name if trace.deciding_factor else "",
            "explanation": _next_step_prose(trace),
            "factors": [
                {
                    "name": factor.name,
                    "value": round(factor.value, 4),
                    "weight": round(factor.weight, 4),
                    "contribution": round(factor.contribution, 4),
                    "detail": factor.detail,
                }
                for factor in sorted(candidate.factors, key=lambda f: -f.contribution)
            ],
            "alternatives": [
                {"slug": alt.slug, "name": alt.name, "score": round(alt.score, 4)}
                for alt in trace.alternatives
            ],
        }

    # A revision, so the "why did my roadmap change" view has something real in it.
    adaptation = adapt_to_failure(
        graph,
        plan,
        AdaptationTrigger(
            kind="assessment_failed",
            concept_id=concept_id_for("gradients-and-jacobians"),
            concept_slug="gradients-and-jacobians",
            concept_name=graph.by_slug("gradients-and-jacobians").name,
            score=0.48,
        ),
        mastery,
        now=NOW,
    )

    roadmap = {
        "id": "rm_demo",
        "title": f"Path to {graph.by_slug(GOAL).name}",
        "summary": _summary(plan, graph),
        "goalText": "I want to understand how neural networks actually learn.",
        "nodes": nodes,
        "edges": [
            {"source": str(s), "target": str(t), "strength": round(w, 2)} for s, t, w in plan.edges
        ],
        "skipped": [
            {
                "slug": s.slug,
                "name": s.name,
                "mastery": round(s.mastery, 3),
                "evidenceCount": s.evidence_count,
            }
            for s in plan.skipped
        ],
        "compressed": [
            {
                "slug": c.slug,
                "name": c.name,
                "mastery": round(c.mastery, 3),
                "originalMinutes": c.original_minutes,
                "reviewMinutes": c.review_minutes,
            }
            for c in plan.compressed
        ],
        "pacing": {
            "totalMinutes": plan.pacing.total_minutes,
            "hoursPerWeek": plan.pacing.hours_per_week,
            "estimatedWeeks": plan.pacing.estimated_weeks,
            "meetsDeadline": plan.pacing.meets_deadline,
            "requiredHoursPerWeek": plan.pacing.required_hours_per_week,
        },
        "warnings": list(plan.warnings),
        "revisionCount": 1,
    }

    dashboard = {
        "next": next_step,
        "roadmapTitle": roadmap["title"],
        "stepsTotal": len(plan.nodes),
        "stepsCompleted": 0,
        "hoursStudied": round(sum(s.evidence_count for s in plan.skipped) * 0.8, 1),
        "hoursRemaining": round(plan.pacing.total_minutes / 60, 1),
        "estimatedWeeks": plan.pacing.estimated_weeks,
        "byDomain": _by_domain(graph, mastery),
        "weakest": [
            {
                "slug": graph.node(cid).slug,
                "name": graph.node(cid).name,
                "mastery": round(est.effective_mastery(NOW), 3),
            }
            for cid, est in sorted(mastery.items(), key=lambda kv: kv[1].effective_mastery(NOW))[:3]
        ],
        "dueForReview": [],
        "recentRevisions": [
            {
                "revisionNo": 1,
                "createdAt": "2026-08-28T14:12:00Z",
                "explanation": explain(adaptation),
                "trigger": adaptation.trigger.as_dict(),
                "mutations": [
                    {
                        "type": str(m.type),
                        "name": m.concept_name,
                        "estimatedMinutes": m.estimated_minutes,
                    }
                    for m in adaptation.mutations
                ],
            }
        ]
        if adaptation.changed
        else [],
    }

    resources: dict[str, list[dict[str, Any]]] = {}
    for node in plan.nodes[:12]:
        result = recommend(
            catalogue.resources,
            node.slug,
            RecommendationContext(
                concept_mastery=(
                    mastery[node.concept_id].effective_mastery(NOW)
                    if node.concept_id in mastery
                    else 0.0
                ),
                minutes_available=(8 * 60),
                today=NOW.date(),
            ),
            limit=3,
        )
        if result.ranked:
            resources[node.slug] = [
                {
                    "title": item.resource.title,
                    "url": item.resource.url,
                    "type": str(item.resource.resource_type),
                    "publisher": item.resource.publisher,
                    "durationMinutes": item.resource.duration_minutes,
                    "difficulty": item.resource.difficulty,
                    "why": resource_reason(item),
                }
                for item in result.ranked
            ]

    return {"roadmap": roadmap, "dashboard": dashboard, "resources": resources}


def _rationale(graph: Any, node: Any, compressed: dict[Any, Any]) -> str:
    """Deterministic rationale, matching the annotator's fallback shape."""
    if node.concept_id in compressed:
        entry = compressed[node.concept_id]
        return (
            f"You have already shown some command of this, so it is a "
            f"{entry.review_minutes}-minute review rather than the full "
            f"{entry.original_minutes} minutes."
        )

    unlocks = [
        graph.node(dependent).name for dependent in graph.direct_dependents(node.concept_id)
    ][:2]
    if unlocks:
        return f"Needed for {' and '.join(unlocks)}."
    return f"About {node.estimated_minutes / 60:.1f} hours at difficulty {node.difficulty}/5."


def _next_step_prose(trace: Any) -> str:
    """The deterministic explanation, which the generated one is validated against."""
    from pathwise.services.decision.engine import fallback_explanation

    return fallback_explanation(trace)


def _summary(plan: Any, graph: Any) -> str:
    parts = [
        f"{len(plan.nodes)} steps from {plan.nodes[0].name} to {plan.nodes[-1].name}, "
        f"about {plan.pacing.total_minutes / 60:.0f} hours."
    ]
    if plan.skipped:
        parts.append(
            f"{len(plan.skipped)} prerequisites were removed because you have already "
            "demonstrated them."
        )
    if plan.compressed:
        saved = sum(c.minutes_saved for c in plan.compressed) / 60
        parts.append(
            f"{len(plan.compressed)} more were shortened to reviews, saving about "
            f"{saved:.0f} hours."
        )
    return " ".join(parts)


def _by_domain(graph: Any, mastery: dict[Any, Any]) -> list[dict[str, Any]]:
    from pathwise.services.knowledge.mastery import aggregate_by_domain

    aggregated = aggregate_by_domain(mastery, graph, NOW)
    counts: dict[str, int] = {}
    for concept_id in mastery:
        if concept_id in graph:
            domain = graph.node(concept_id).domain
            counts[domain] = counts.get(domain, 0) + 1

    return [
        {"domain": domain, "mastery": round(value, 3), "conceptsMeasured": counts.get(domain, 0)}
        for domain, value in sorted(aggregated.items())
    ]


def main() -> None:
    """Write the fixture module.

    Writes the file directly rather than printing for shell redirection: on Windows
    stdout is redirected in the console codepage, which mangles every non-ASCII
    character into bytes Vite cannot parse as UTF-8.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate frontend fixtures.")
    parser.add_argument(
        "--out",
        default="../frontend/src/lib/fixtures.ts",
        help="Where to write the generated module.",
    )
    args = parser.parse_args()

    data = build()
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    emit("/**")
    emit(" * Generated by `python -m pathwise.cli_fixtures`. Do not edit by hand.")
    emit(" *")
    emit(" * Real output from the deterministic engines over the real seed graph, not")
    emit(" * invented sample data - so the UI is designed against what the system")
    emit(" * actually produces, including the awkward parts.")
    emit(" */")
    emit()
    emit('import type { Dashboard, Resource, Roadmap } from "@/lib/types";')
    emit()
    emit(f"export const ROADMAP_FIXTURE: Roadmap = {json.dumps(data['roadmap'], indent=2)};")
    emit()
    emit(f"export const DASHBOARD_FIXTURE: Dashboard = {json.dumps(data['dashboard'], indent=2)};")
    emit()
    emit(
        "export const RESOURCES_FIXTURE: Record<string, Resource[]> = "
        f"{json.dumps(data['resources'], indent=2)};"
    )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
