/**
 * Formatting the numbers the engines produce.
 *
 * These live in one place because the same quantity must read the same way
 * everywhere: a duration is never "150 minutes" in one view and "2.5h" in another.
 */

/** Durations are shown in the unit a person would plan with, not the unit stored. */
export function formatDuration(minutes: number): string {
  if (minutes < 60) return `${Math.round(minutes)}m`;
  const hours = minutes / 60;
  if (hours < 10) return `${hours.toFixed(1)}h`;
  return `${Math.round(hours)}h`;
}

export function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/**
 * Factor names come from the backend as identifiers. They are shown as prose, but
 * the mapping is exhaustive and explicit — a new factor should surface as a visible
 * gap rather than being silently prettified into something plausible.
 */
const FACTOR_LABELS: Record<string, string> = {
  goal_relevance: "Goal relevance",
  readiness: "Readiness",
  review_debt: "Review debt",
  difficulty_fit: "Difficulty fit",
  remediation: "Remediation",
  momentum: "Momentum",
};

export function factorLabel(name: string): string {
  return FACTOR_LABELS[name] ?? name.replace(/_/g, " ");
}

const DOMAIN_LABELS: Record<string, string> = {
  mathematics: "Mathematics",
  programming: "Programming",
  "machine-learning": "Machine learning",
};

export function domainLabel(domain: string): string {
  return DOMAIN_LABELS[domain] ?? domain;
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}
