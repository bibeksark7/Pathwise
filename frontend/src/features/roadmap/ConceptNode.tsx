import { Handle, Position, type NodeProps } from "@xyflow/react";
import { MasteryCurve } from "@/components/MasteryCurve";
import type { ConceptNode as Concept } from "@/lib/types";

/**
 * One step on the canvas.
 *
 * Six states have to be distinguishable at a glance, and the temptation is six
 * colours. That fails: with everything shouting, the one thing that matters — what
 * to do next — becomes invisible. So state is carried by *treatment* (a hairline, a
 * left bar, a dash, reduced opacity) and the single saturated colour in the palette
 * is spent exclusively on `recommended`.
 */

export interface ConceptNodeData extends Record<string, unknown> {
  concept: Concept;
  onOpen: (slug: string) => void;
  isSelected: boolean;
}

const STATE_STYLES: Record<Concept["status"], string> = {
  recommended: "border-signal bg-raised shadow-signal",
  in_progress: "border-line-bright bg-raised",
  completed: "border-ok/35 bg-ok/[0.06]",
  needs_review: "border-warn/50 border-dashed bg-warn/[0.05]",
  not_started: "border-line bg-surface hover:border-line-bright",
  // Reduced to a whisper: present, clearly unavailable, not competing for attention.
  locked: "border-line bg-surface/60 opacity-[0.62]",
};

const STATE_LABELS: Record<Concept["status"], string> = {
  recommended: "Next",
  in_progress: "In progress",
  completed: "Done",
  needs_review: "Review due",
  not_started: "Ready",
  locked: "Locked",
};

export function ConceptNodeCard({ data }: NodeProps) {
  const { concept, onOpen, isSelected } = data as unknown as ConceptNodeData;
  const isLocked = concept.status === "locked";
  const isNext = concept.status === "recommended";
  const hours = concept.estimatedMinutes / 60;

  return (
    <div className="relative">
      <Handle type="target" position={Position.Top} />

      <button
        type="button"
        onClick={() => onOpen(concept.slug)}
        aria-label={`${concept.name}. ${STATE_LABELS[concept.status]}. ${hours.toFixed(1)} hours, difficulty ${concept.difficulty} of 5.`}
        aria-current={isNext ? "step" : undefined}
        className={[
          "w-[232px] rounded-node border px-3 py-2.5 text-left",
          "transition-[border-color,box-shadow,transform] duration-200 ease-instrument",
          isLocked ? "cursor-default" : "cursor-pointer hover:-translate-y-px",
          isSelected ? "ring-1 ring-line-bright" : "",
          STATE_STYLES[concept.status],
        ].join(" ")}
      >
        <div className="flex items-baseline justify-between gap-2">
          <span
            className={`font-mono text-micro uppercase tracking-wider ${
              isNext ? "text-signal" : "text-faint"
            }`}
          >
            {STATE_LABELS[concept.status]}
          </span>
          {concept.nodeType === "review" && (
            <span className="font-mono text-micro text-muted">review</span>
          )}
        </div>

        <h3
          className={`mt-1 font-display text-[0.9375rem] font-semibold leading-tight ${
            isLocked ? "text-muted" : "text-ink"
          }`}
        >
          {concept.name}
        </h3>

        {/* The slug is an identifier, not prose — set as one. */}
        <p className="mt-0.5 truncate font-mono text-micro text-faint">{concept.slug}</p>

        <div className="mt-2.5 flex items-end justify-between gap-3">
          <div className="tabular font-mono text-micro text-muted">
            {hours < 1 ? `${concept.estimatedMinutes}m` : `${hours.toFixed(1)}h`}
            <span className="mx-1.5 text-faint">·</span>
            <span aria-label={`difficulty ${concept.difficulty} of 5`}>
              d{concept.difficulty}
            </span>
          </div>
          <MasteryCurve estimate={concept.mastery} highlighted={isNext} />
        </div>
      </button>

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
