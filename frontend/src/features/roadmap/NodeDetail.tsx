import { useEffect, useRef } from "react";
import { api } from "@/lib/api";
import { useAsync } from "@/hooks/useAsync";
import { MasteryCurve, describeConfidence } from "@/components/MasteryCurve";
import { domainLabel, formatDuration, formatPercent } from "@/lib/format";
import type { ConceptNode } from "@/lib/types";

/**
 * The panel behind a node.
 *
 * Two things earn the space: what the system believes about this concept and why,
 * and what to actually read or watch. The mastery block leads because it is the
 * claim everything else rests on — including whether this step exists at all.
 */

interface Props {
  concept: ConceptNode;
  prerequisites: ConceptNode[];
  onClose: () => void;
}

export function NodeDetail({ concept, prerequisites, onClose }: Props) {
  const panel = useRef<HTMLDivElement>(null);
  const resources = useAsync(() => api.getResources(concept.slug), [concept.slug]);

  // Escape closes, and focus moves into the panel on open so a keyboard user is not
  // stranded back on the canvas behind it.
  useEffect(() => {
    panel.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <aside
      ref={panel}
      tabIndex={-1}
      role="dialog"
      aria-modal="false"
      aria-label={concept.name}
      className="absolute inset-y-0 right-0 z-10 w-full max-w-md overflow-y-auto border-l border-line bg-surface shadow-drawer sm:w-[26rem]"
    >
      <div className="sticky top-0 border-b border-line bg-surface/95 px-5 py-4 backdrop-blur">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="eyebrow">{domainLabel(concept.domain)}</p>
            <h2 className="mt-1 font-display text-lg font-semibold leading-tight text-ink">
              {concept.name}
            </h2>
            <p className="mt-0.5 truncate font-mono text-micro text-faint">{concept.slug}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="shrink-0 rounded-node border border-line px-2 py-1 font-mono text-micro text-muted hover:border-line-bright hover:text-ink"
          >
            esc
          </button>
        </div>

        <p className="tabular mt-3 font-mono text-micro text-muted">
          {formatDuration(concept.estimatedMinutes)}
          <span className="mx-2 text-faint">·</span>difficulty {concept.difficulty}/5
          <span className="mx-2 text-faint">·</span>
          {concept.nodeType}
        </p>
      </div>

      <div className="space-y-6 px-5 py-5">
        {concept.rationale && (
          <section>
            <h3 className="eyebrow">Why it is in your path</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink">{concept.rationale}</p>
          </section>
        )}

        <section>
          <h3 className="eyebrow">What we know</h3>
          {concept.mastery ? (
            <div className="mt-2">
              <div className="flex items-end gap-4">
                <MasteryCurve estimate={concept.mastery} width={140} height={44} />
                <div>
                  <p className="tabular font-display text-2xl font-semibold text-ink">
                    {formatPercent(concept.mastery.mastery)}
                  </p>
                  <p className="font-mono text-micro text-faint">
                    {describeConfidence(concept.mastery.confidence)}
                  </p>
                </div>
              </div>
              <p className="mt-3 text-sm leading-relaxed text-muted">
                From {concept.mastery.evidenceCount}{" "}
                {concept.mastery.evidenceCount === 1 ? "result" : "results"}. The curve is the
                posterior: where it is narrow, the estimate is settled; where it is wide, more
                evidence would still move it.
              </p>
            </div>
          ) : (
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Nothing measured yet. That is not the same as scoring zero — until there is
              evidence, this concept is neither skippable nor known to be weak.
            </p>
          )}
        </section>

        <section>
          <h3 className="eyebrow">Prerequisites</h3>
          {prerequisites.length === 0 ? (
            <p className="mt-2 text-sm text-muted">None. You can start this whenever.</p>
          ) : (
            <ul className="mt-2 space-y-2">
              {prerequisites.map((prerequisite) => (
                <li
                  key={prerequisite.conceptId}
                  className="flex items-center justify-between gap-3 border-l border-line pl-3"
                >
                  <span className="truncate text-sm text-ink">{prerequisite.name}</span>
                  <MasteryCurve estimate={prerequisite.mastery} />
                </li>
              ))}
            </ul>
          )}
        </section>

        <section>
          <h3 className="eyebrow">Where to learn it</h3>
          {resources.status === "loading" && (
            <p className="mt-2 font-mono text-micro text-faint">Ranking resources…</p>
          )}
          {resources.status === "error" && (
            <p className="mt-2 text-sm text-warn">{resources.error.message}</p>
          )}
          {resources.status === "ready" &&
            (resources.data.length === 0 ? (
              <p className="mt-2 text-sm leading-relaxed text-muted">
                No catalogued resource covers this concept yet. Nothing is generated to fill
                the gap — an invented link is worse than an honest hole.
              </p>
            ) : (
              <ul className="mt-2 space-y-3">
                {resources.data.map((resource) => (
                  <li key={resource.url} className="border-l border-line pl-3">
                    <a
                      href={resource.url}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="text-sm font-medium text-ink underline decoration-line-bright underline-offset-4 hover:text-signal hover:decoration-signal"
                    >
                      {resource.title}
                    </a>
                    <p className="tabular mt-1 font-mono text-micro text-faint">
                      {resource.publisher}
                      <span className="mx-2">·</span>
                      {resource.type}
                      {resource.durationMinutes !== null && (
                        <>
                          <span className="mx-2">·</span>
                          {formatDuration(resource.durationMinutes)}
                        </>
                      )}
                    </p>
                    <p className="mt-1 text-sm leading-relaxed text-muted">{resource.why}</p>
                  </li>
                ))}
              </ul>
            ))}
        </section>
      </div>
    </aside>
  );
}
