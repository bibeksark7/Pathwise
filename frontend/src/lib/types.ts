/**
 * Shapes the API returns.
 *
 * Deliberately mirrors the backend's own vocabulary — `NodeStatus`, `DecisionTrace`,
 * `FactorScore` — rather than inventing frontend names for the same ideas. When the
 * two disagree about what a thing is called, the bug is always in the translation.
 */

export type NodeStatus =
  | "not_started"
  | "in_progress"
  | "completed"
  | "needs_review"
  | "locked"
  | "recommended";

export type NodeType = "topic" | "practice" | "assessment" | "project" | "review";

export interface ConceptNode {
  conceptId: string;
  slug: string;
  name: string;
  domain: string;
  difficulty: number;
  estimatedMinutes: number;
  status: NodeStatus;
  nodeType: NodeType;
  orderIndex: number;
  dependsOn: string[];
  /** Why this step is in the path. Generated, or a deterministic fallback. */
  rationale?: string;
  /** Absent when nothing has been measured — which is not the same as zero. */
  mastery?: MasteryEstimate | null;
}

/**
 * Mastery is a Beta posterior, so level and certainty are separate quantities.
 * The UI has to show both or it misrepresents the model: 0.8 from one quiz and 0.8
 * from twelve assessments are different states, and only the second justifies
 * skipping material.
 */
export interface MasteryEstimate {
  alpha: number;
  beta: number;
  mastery: number;
  confidence: number;
  evidenceCount: number;
}

export interface RoadmapEdge {
  source: string;
  target: string;
  strength: number;
}

export interface SkippedConcept {
  slug: string;
  name: string;
  mastery: number;
  evidenceCount: number;
}

export interface CompressedConcept {
  slug: string;
  name: string;
  mastery: number;
  originalMinutes: number;
  reviewMinutes: number;
}

export interface Pacing {
  totalMinutes: number;
  hoursPerWeek: number;
  estimatedWeeks: number;
  meetsDeadline: boolean | null;
  requiredHoursPerWeek: number | null;
}

export interface Roadmap {
  id: string;
  title: string;
  summary: string;
  goalText: string;
  nodes: ConceptNode[];
  edges: RoadmapEdge[];
  skipped: SkippedConcept[];
  compressed: CompressedConcept[];
  pacing: Pacing;
  warnings: string[];
  revisionCount: number;
}

/** One weighted term in a decision. The UI renders these verbatim — it does not
 *  recompute or reword them, because the prose was validated against them. */
export interface FactorScore {
  name: string;
  value: number;
  weight: number;
  contribution: number;
  detail: string;
}

export interface NextStep {
  slug: string;
  name: string;
  kind: "new" | "review";
  estimatedMinutes: number;
  difficulty: number;
  score: number;
  /** The factor that chose this over the runner-up — what the explanation leads with. */
  decidingFactor: string;
  explanation: string;
  factors: FactorScore[];
  alternatives: { slug: string; name: string; score: number }[];
}

export interface Resource {
  title: string;
  url: string;
  type: string;
  publisher: string;
  durationMinutes: number | null;
  difficulty: number;
  /** Why this was ranked here. Always present — a recommendation without a reason
   *  is just a link. */
  why: string;
}

export interface RevisionEntry {
  revisionNo: number;
  createdAt: string;
  explanation: string;
  trigger: {
    kind: string;
    concept: string;
    name: string;
    score: number | null;
    attempt?: number;
    detail?: string;
  };
  mutations: { type: string; name: string; estimatedMinutes: number }[];
}

export interface DomainMastery {
  domain: string;
  mastery: number;
  conceptsMeasured: number;
}

export interface Dashboard {
  next: NextStep | null;
  roadmapTitle: string;
  stepsTotal: number;
  stepsCompleted: number;
  hoursStudied: number;
  hoursRemaining: number;
  estimatedWeeks: number;
  byDomain: DomainMastery[];
  weakest: { slug: string; name: string; mastery: number }[];
  dueForReview: { slug: string; name: string; daysOverdue: number }[];
  recentRevisions: RevisionEntry[];
}
