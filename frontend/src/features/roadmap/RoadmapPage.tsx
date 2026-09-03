import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "@/lib/api";
import { useAsync } from "@/hooks/useAsync";
import { formatDuration, formatPercent } from "@/lib/format";
import { RoadmapCanvas } from "@/features/roadmap/RoadmapCanvas";
import { NodeDetail } from "@/features/roadmap/NodeDetail";
import type { ConceptNode, Roadmap } from "@/lib/types";

export function RoadmapPage() {
  const [params, setParams] = useSearchParams();
  const state = useAsync(() => api.getRoadmap());
  const selectedSlug = params.get("concept");

  // The open panel lives in the URL, so a specific step is a link someone can send.
  const select = useCallback(
    (slug: string) => {
      const next = new URLSearchParams(params);
      if (slug) next.set("concept", slug);
      else next.delete("concept");
      setParams(next, { replace: true });
    },
    [params, setParams],
  );

  const selected = useMemo<ConceptNode | null>(() => {
    if (state.status !== "ready" || !selectedSlug) return null;
    return state.data.nodes.find((node) => node.slug === selectedSlug) ?? null;
  }, [state, selectedSlug]);

  const prerequisites = useMemo<ConceptNode[]>(() => {
    if (state.status !== "ready" || !selected) return [];
    const byId = new Map(state.data.nodes.map((node) => [node.conceptId, node]));
    return selected.dependsOn
      .map((id) => byId.get(id))
      .filter((node): node is ConceptNode => node !== undefined);
  }, [state, selected]);

  if (state.status === "loading") {
    return (
      <div className="grid h-full place-items-center" aria-busy="true">
        <p className="eyebrow">Laying out your path</p>
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="grid h-full place-items-center px-6">
        <div className="max-w-md text-center">
          <p className="eyebrow text-warn">Could not load your path</p>
          <p className="mt-2 text-sm leading-relaxed text-muted">{state.error.message}</p>
        </div>
      </div>
    );
  }

  const roadmap = state.data;

  return (
    <div className="flex h-full min-h-0 flex-col lg:flex-row">
      <aside className="hidden w-72 shrink-0 overflow-y-auto border-r border-line bg-surface px-5 py-6 lg:block">
        <RailContent roadmap={roadmap} />
      </aside>

      {/* Below the rail breakpoint the same content collapses into a disclosure, so
          what was removed from the path stays reachable rather than being dropped
          along with the column that used to hold it. */}
      <details className="shrink-0 border-b border-line bg-surface lg:hidden">
        <summary className="cursor-pointer list-none px-5 py-3 font-mono text-micro uppercase tracking-wider text-muted">
          {roadmap.title} — {roadmap.nodes.length} steps, {roadmap.skipped.length} skipped
        </summary>
        <div className="max-h-[50vh] overflow-y-auto border-t border-line px-5 py-4">
          <RailContent roadmap={roadmap} />
        </div>
      </details>

      <div className="relative min-h-0 min-w-0 flex-1">
        <RoadmapCanvas
          nodes={roadmap.nodes}
          edges={roadmap.edges}
          selectedSlug={selectedSlug}
          onSelect={select}
        />

        {selected && (
          <NodeDetail
            concept={selected}
            prerequisites={prerequisites}
            onClose={() => select("")}
          />
        )}
      </div>
    </div>
  );
}

/**
 * The left rail carries what the canvas structurally cannot: the work that is *not*
 * on it.
 *
 * Nine concepts were removed and two shortened because the learner demonstrated them.
 * That is the adaptive behaviour doing its job, and a graph of only the remaining
 * steps would hide the most interesting thing the system did.
 */
function RailContent({ roadmap }: { roadmap: Roadmap }) {
  return (
    <>
      <p className="eyebrow">Goal</p>
      <p className="mt-2 text-sm leading-relaxed text-ink">{roadmap.goalText}</p>

      <p className="mt-6 text-sm leading-relaxed text-muted">{roadmap.summary}</p>

      <dl className="mt-6 space-y-2 border-t border-line pt-4">
        <Row label="Steps" value={String(roadmap.nodes.length)} />
        <Row label="Total time" value={formatDuration(roadmap.pacing.totalMinutes)} />
        <Row label="Per week" value={`${roadmap.pacing.hoursPerWeek}h`} />
        <Row label="Finishes in" value={`${roadmap.pacing.estimatedWeeks.toFixed(1)} weeks`} />
        {roadmap.pacing.requiredHoursPerWeek !== null && (
          <Row
            label="Deadline needs"
            value={`${roadmap.pacing.requiredHoursPerWeek.toFixed(1)}h/wk`}
            alert={roadmap.pacing.meetsDeadline === false}
          />
        )}
      </dl>

      {roadmap.warnings.length > 0 && (
        <ul className="mt-4 space-y-1.5 border-t border-line pt-4">
          {roadmap.warnings.map((warning) => (
            <li key={warning} className="text-sm leading-relaxed text-warn">
              {warning}
            </li>
          ))}
        </ul>
      )}

      {roadmap.skipped.length > 0 && (
        <section className="mt-6 border-t border-line pt-4">
          <h2 className="eyebrow">Skipped — already demonstrated</h2>
          <ul className="mt-2 space-y-1">
            {roadmap.skipped.map((concept) => (
              <li
                key={concept.slug}
                className="flex items-baseline justify-between gap-3 text-sm text-muted"
              >
                <span className="truncate">{concept.name}</span>
                <span className="tabular shrink-0 font-mono text-micro text-faint">
                  {formatPercent(concept.mastery)}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {roadmap.compressed.length > 0 && (
        <section className="mt-6 border-t border-line pt-4">
          <h2 className="eyebrow">Shortened to a review</h2>
          <ul className="mt-2 space-y-1">
            {roadmap.compressed.map((concept) => (
              <li
                key={concept.slug}
                className="flex items-baseline justify-between gap-3 text-sm text-muted"
              >
                <span className="truncate">{concept.name}</span>
                <span className="tabular shrink-0 font-mono text-micro text-faint">
                  {formatDuration(concept.originalMinutes)} →{" "}
                  {formatDuration(concept.reviewMinutes)}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}

function Row({ label, value, alert = false }: { label: string; value: string; alert?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="font-mono text-micro text-faint">{label}</dt>
      <dd className={`tabular font-mono text-micro ${alert ? "text-warn" : "text-ink"}`}>
        {value}
      </dd>
    </div>
  );
}
