import type { ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "@/lib/api";
import { useAsync } from "@/hooks/useAsync";
import { domainLabel, formatDate, formatDuration, formatPercent } from "@/lib/format";
import { NextStepCard } from "@/features/dashboard/NextStepCard";
import type { RevisionEntry } from "@/lib/types";

export function DashboardPage() {
  const navigate = useNavigate();
  const state = useAsync(() => api.getDashboard());

  if (state.status === "loading") return <Skeleton />;
  if (state.status === "error") return <LoadFailed message={state.error.message} />;

  const dashboard = state.data;
  const openConcept = (slug: string) => navigate(`/path?concept=${encodeURIComponent(slug)}`);

  return (
    <div className="mx-auto max-w-5xl px-6 py-10 sm:px-8">
      <header>
        <p className="eyebrow">{dashboard.roadmapTitle}</p>
        <h1 className="mt-2 font-display text-xl font-semibold text-ink">Where you are</h1>
      </header>

      <dl className="mt-6 grid grid-cols-2 gap-px overflow-hidden rounded-node border border-line bg-line sm:grid-cols-4">
        <Stat label="Steps done" value={`${dashboard.stepsCompleted}/${dashboard.stepsTotal}`} />
        <Stat label="Hours studied" value={dashboard.hoursStudied.toFixed(1)} />
        <Stat label="Hours left" value={dashboard.hoursRemaining.toFixed(1)} />
        <Stat label="Weeks left" value={dashboard.estimatedWeeks.toFixed(1)} />
      </dl>

      {dashboard.next ? (
        <div className="mt-8">
          <NextStepCard next={dashboard.next} onOpen={openConcept} />
        </div>
      ) : (
        <p className="mt-8 rounded-node border border-line bg-surface px-6 py-8 text-sm text-muted">
          Nothing is unlocked right now. Finish an in-progress step, or take a diagnostic so
          the engine has evidence to work from.
        </p>
      )}

      <div className="mt-10 grid gap-6 lg:grid-cols-2">
        <Panel title="Mastery by subject">
          <ul className="space-y-3">
            {dashboard.byDomain.map((domain) => (
              <li key={domain.domain}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-sm text-ink">{domainLabel(domain.domain)}</span>
                  <span className="tabular font-mono text-micro text-muted">
                    {formatPercent(domain.mastery)}
                    <span className="ml-2 text-faint">{domain.conceptsMeasured} measured</span>
                  </span>
                </div>
                <div className="mt-1.5 h-1.5 w-full bg-line/60">
                  <div
                    className="h-full bg-line-bright"
                    style={{ width: `${(domain.mastery * 100).toFixed(1)}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
        </Panel>

        <Panel title="Weakest measured concepts">
          {dashboard.weakest.length === 0 ? (
            <Empty>Nothing measured yet. A diagnostic fills this in.</Empty>
          ) : (
            <ul className="space-y-2">
              {dashboard.weakest.map((concept) => (
                <li key={concept.slug} className="flex items-baseline justify-between gap-4">
                  <button
                    type="button"
                    onClick={() => openConcept(concept.slug)}
                    className="truncate text-left text-sm text-ink hover:text-signal"
                  >
                    {concept.name}
                  </button>
                  <span className="tabular font-mono text-micro text-muted">
                    {formatPercent(concept.mastery)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Due for review">
          {dashboard.dueForReview.length === 0 ? (
            <Empty>
              Nothing is due. Review dates come from the forgetting curve, so this fills
              itself in as time passes.
            </Empty>
          ) : (
            <ul className="space-y-2">
              {dashboard.dueForReview.map((concept) => (
                <li key={concept.slug} className="flex items-baseline justify-between gap-4">
                  <button
                    type="button"
                    onClick={() => openConcept(concept.slug)}
                    className="truncate text-left text-sm text-ink hover:text-signal"
                  >
                    {concept.name}
                  </button>
                  <span className="tabular font-mono text-micro text-warn">
                    {concept.daysOverdue}d overdue
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Why your path changed">
          {dashboard.recentRevisions.length === 0 ? (
            <Empty>
              The path has not been revised yet. It changes when evidence says it should.
            </Empty>
          ) : (
            <ol className="space-y-5">
              {dashboard.recentRevisions.map((revision) => (
                <Revision key={revision.revisionNo} revision={revision} />
              ))}
            </ol>
          )}
        </Panel>
      </div>
    </div>
  );
}

/**
 * One roadmap revision, shown as evidence then change.
 *
 * This is the honest form of "the system changed your plan": the score that triggered
 * it, the reasoning, and the exact mutations applied. All three come from the stored
 * revision row, so what is displayed is what happened, not a retelling of it.
 */
function Revision({ revision }: { revision: RevisionEntry }) {
  return (
    <li>
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-mono text-micro text-faint">
          rev {revision.revisionNo} · {formatDate(revision.createdAt)}
        </span>
        {revision.trigger.score !== null && (
          <span className="tabular font-mono text-micro text-warn">
            {formatPercent(revision.trigger.score)} on {revision.trigger.name}
          </span>
        )}
      </div>

      <p className="mt-2 text-sm leading-relaxed text-ink">{revision.explanation}</p>

      <ul className="mt-2 space-y-1">
        {revision.mutations.map((mutation, index) => (
          <li
            key={`${mutation.type}-${mutation.name}-${index}`}
            className="font-mono text-micro text-muted"
          >
            <span className="text-faint">{mutation.type.replace(/_/g, " ")}</span>{" "}
            {mutation.name}{" "}
            <span className="text-faint">+{formatDuration(mutation.estimatedMinutes)}</span>
          </li>
        ))}
      </ul>
    </li>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface px-4 py-3">
      <dt className="font-mono text-micro uppercase tracking-wider text-faint">{label}</dt>
      <dd className="tabular mt-1 font-display text-xl font-semibold text-ink">{value}</dd>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-node border border-line bg-surface px-5 py-4">
      <h2 className="eyebrow">{title}</h2>
      <div className="mt-3">{children}</div>
    </section>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <p className="text-sm leading-relaxed text-faint">{children}</p>;
}

function Skeleton() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-10 sm:px-8" aria-busy="true">
      <p className="eyebrow">Loading</p>
      <div className="mt-6 h-24 rounded-node border border-line bg-surface" />
      <div className="mt-8 h-72 rounded-node border border-line bg-surface" />
    </div>
  );
}

function LoadFailed({ message }: { message: string }) {
  return (
    <div className="mx-auto max-w-5xl px-6 py-10 sm:px-8">
      <p className="eyebrow text-warn">Could not load your dashboard</p>
      <p className="mt-2 max-w-prose text-sm leading-relaxed text-muted">{message}</p>
      <button
        type="button"
        onClick={() => window.location.reload()}
        className="mt-4 rounded-node border border-line-bright px-3 py-1.5 text-sm text-ink hover:border-signal hover:text-signal"
      >
        Try again
      </button>
    </div>
  );
}
