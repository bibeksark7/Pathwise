import { factorLabel, formatDuration } from "@/lib/format";
import type { NextStep } from "@/lib/types";

/**
 * "What should I do next?" — the one question the product exists to answer.
 *
 * The answer came from a weighted sum, so the card shows the weighted sum. Every
 * factor, its weight, and its contribution are rendered verbatim from the
 * `DecisionTrace`; nothing here is recomputed, because the prose above was validated
 * against exactly these numbers and a second implementation could disagree with it.
 *
 * Showing the arithmetic is the point. A recommendation you cannot interrogate is
 * indistinguishable from a guess, and the deciding factor — the one that beat the
 * runner-up, not merely the largest term — is what makes the answer specific to you.
 */

interface Props {
  next: NextStep;
  onOpen: (slug: string) => void;
}

export function NextStepCard({ next, onOpen }: Props) {
  const largest = Math.max(...next.factors.map((factor) => factor.contribution), 0.0001);
  const deciding = next.factors.find((factor) => factor.name === next.decidingFactor);

  return (
    <section
      aria-labelledby="next-heading"
      className="rounded-node border border-signal/50 bg-raised shadow-signal"
    >
      <div className="border-b border-line px-6 py-5 sm:px-8 sm:py-6">
        <p className="eyebrow text-signal" id="next-heading">
          Do this next
        </p>

        <div className="mt-3 flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <h2 className="font-display text-2xl font-semibold leading-tight text-ink sm:text-3xl">
            {next.name}
          </h2>
          <span className="tabular font-mono text-micro text-muted">
            {formatDuration(next.estimatedMinutes)} · d{next.difficulty}/5 ·{" "}
            {next.kind === "review" ? "review" : "new material"}
          </span>
        </div>

        <p className="mt-3 max-w-prose text-[0.9375rem] leading-relaxed text-muted">
          {next.explanation}
        </p>

        <button
          type="button"
          onClick={() => onOpen(next.slug)}
          className="mt-5 rounded-node bg-signal px-4 py-2 font-display text-sm font-semibold text-ground transition-colors duration-150 ease-instrument hover:bg-signal/90"
        >
          Start {next.name}
        </button>
      </div>

      <div className="px-6 py-5 sm:px-8">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <p className="eyebrow">Why this, and not something else</p>
          <p className="tabular font-mono text-micro text-faint">
            score {next.score.toFixed(3)}
          </p>
        </div>

        {deciding && (
          <p className="mt-2 max-w-prose text-sm leading-relaxed text-ink">
            <span className="text-signal">{factorLabel(deciding.name)}</span> decided it:{" "}
            {deciding.detail}.
          </p>
        )}

        {/* Capped: a factor bar stretched across a full-width card puts its number
            so far from its label that the pair stops reading as one row. */}
        <ul className="mt-4 max-w-2xl space-y-1.5">
          {next.factors.map((factor) => {
            const isDeciding = factor.name === next.decidingFactor;
            const width = (factor.contribution / largest) * 100;

            return (
              <li key={factor.name} className="grid grid-cols-[9.5rem_1fr_3.25rem] items-center gap-3">
                <span
                  className={`truncate font-mono text-micro ${
                    isDeciding ? "text-signal" : "text-muted"
                  }`}
                  title={factor.detail}
                >
                  {factorLabel(factor.name)}
                </span>

                {/* The track is the full weight available to this factor; the fill is
                    what it actually contributed. An empty track is information: the
                    factor was considered and scored zero. */}
                <span className="h-1.5 w-full bg-line/60" aria-hidden="true">
                  <span
                    className={`block h-full ${isDeciding ? "bg-signal" : "bg-line-bright"}`}
                    style={{ width: `${width.toFixed(1)}%` }}
                  />
                </span>

                <span className="tabular text-right font-mono text-micro text-faint">
                  {factor.contribution.toFixed(3)}
                </span>
              </li>
            );
          })}
        </ul>

        {next.alternatives.length > 0 && (
          <div className="mt-5 max-w-2xl border-t border-line pt-4">
            <p className="eyebrow">Runners-up</p>
            <ul className="mt-2 space-y-1">
              {next.alternatives.map((alternative) => (
                <li
                  key={alternative.slug}
                  className="flex items-baseline justify-between gap-4 text-sm text-muted"
                >
                  <button
                    type="button"
                    onClick={() => onOpen(alternative.slug)}
                    className="truncate text-left hover:text-ink"
                  >
                    {alternative.name}
                  </button>
                  <span className="tabular font-mono text-micro text-faint">
                    {alternative.score.toFixed(3)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}
