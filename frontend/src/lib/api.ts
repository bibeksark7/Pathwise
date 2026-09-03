/**
 * The API layer.
 *
 * Every call goes through here, and today every call is answered from fixtures. The
 * point is that the *signatures* are the real ones — when the backend routes land,
 * only the bodies of these functions change, and no component is touched.
 *
 * The fixture data is not invented. It is the output the deterministic engines
 * actually produce for a learner aiming at backpropagation who knows Python and
 * basic calculus: the same 25-step path, the same skipped concepts, the same
 * `chain-rule` recommendation with `goal_relevance` as the deciding factor. Building
 * a UI against prettier data than the system produces is how you end up with a
 * design that cannot display the real thing.
 */

import type { Dashboard, Resource, Roadmap } from "@/lib/types";
import { DASHBOARD_FIXTURE, RESOURCES_FIXTURE, ROADMAP_FIXTURE } from "@/lib/fixtures";

const USE_FIXTURES = import.meta.env.VITE_API_BASE_URL === undefined;
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

/** Enough latency that loading states are visible during development. */
const FIXTURE_DELAY_MS = 220;

async function delay<T>(value: T): Promise<T> {
  await new Promise((resolve) => setTimeout(resolve, FIXTURE_DELAY_MS));
  return value;
}

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { Accept: "application/json" },
    credentials: "include",
  });

  if (!response.ok) {
    // The backend returns RFC 9457 problem documents, whose `detail` is written to
    // be shown to a person. Surface it rather than a bare status code.
    const problem = await response.json().catch(() => null);
    throw new ApiError(problem?.detail ?? `Request failed (${response.status})`, response.status);
  }

  return response.json() as Promise<T>;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export const api = {
  async getRoadmap(): Promise<Roadmap> {
    return USE_FIXTURES ? delay(ROADMAP_FIXTURE) : request<Roadmap>("/api/roadmaps/current");
  },

  async getDashboard(): Promise<Dashboard> {
    return USE_FIXTURES ? delay(DASHBOARD_FIXTURE) : request<Dashboard>("/api/dashboard");
  },

  async getResources(slug: string): Promise<Resource[]> {
    if (USE_FIXTURES) return delay(RESOURCES_FIXTURE[slug] ?? []);
    return request<Resource[]>(`/api/resources?concept=${encodeURIComponent(slug)}`);
  },
};
