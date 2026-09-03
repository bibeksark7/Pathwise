import { useMemo } from "react";
import type { MasteryEstimate } from "@/lib/types";

/**
 * The posterior density for one concept's mastery.
 *
 * This is the signature element, and it exists because a progress bar would lie.
 * Mastery here is a Beta(α, β) posterior, so *level* and *certainty* are separate
 * quantities — 0.8 measured once and 0.8 measured twelve times are different states,
 * and only the second justifies skipping material. A bar can show one number. A
 * density shows both: where the mass sits, and how tightly.
 *
 * Nobody has to read it as a distribution for it to work. A wide flat smear reads as
 * "we're not sure"; a narrow spike reads as "we know". That is exactly the
 * distinction the product turns on, and it is legible at 44 pixels wide.
 */

interface Props {
  estimate?: MasteryEstimate | null;
  width?: number;
  height?: number;
  /** Tints the curve when this node is the recommended next step. */
  highlighted?: boolean;
  className?: string;
}

const SAMPLES = 48;

/**
 * Unnormalised Beta density, scaled to its own maximum.
 *
 * The true PDF's normalising constant needs a Beta function; the shape is all that
 * is being drawn, so it is omitted and the curve is scaled to fit its own peak.
 * Computed in log space — α and β reach the hundreds after enough evidence, and
 * `x**(α-1)` underflows to zero long before that.
 */
function betaShape(alpha: number, beta: number): { x: number; y: number }[] {
  const a = Math.max(alpha, 1e-6);
  const b = Math.max(beta, 1e-6);
  const points: { x: number; y: number }[] = [];
  let peak = -Infinity;

  for (let i = 0; i <= SAMPLES; i += 1) {
    // Nudged off the exact endpoints: log(0) is -Infinity and would blank the curve.
    const x = Math.min(Math.max(i / SAMPLES, 1e-4), 1 - 1e-4);
    const logY = (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x);
    points.push({ x, y: logY });
    if (logY > peak) peak = logY;
  }

  return points.map(({ x, y }) => ({ x, y: Math.exp(y - peak) }));
}

export function MasteryCurve({
  estimate,
  width = 44,
  height = 16,
  highlighted = false,
  className = "",
}: Props) {
  const path = useMemo(() => {
    if (!estimate) return null;
    const points = betaShape(estimate.alpha, estimate.beta);
    const segments = points.map(({ x, y }) => {
      const px = x * width;
      const py = height - y * (height - 1) - 0.5;
      return `${px.toFixed(2)},${py.toFixed(2)}`;
    });
    return `M0,${height} L${segments.join(" L")} L${width},${height} Z`;
  }, [estimate, width, height]);

  // No evidence is not the same as zero mastery, and must not render as an empty
  // bar — that would read as "you scored nothing" rather than "not measured yet".
  if (!estimate || !path) {
    return (
      <div
        className={`flex items-center ${className}`}
        style={{ width, height }}
        title="Not measured yet"
        aria-label="Not measured yet"
      >
        <div className="h-px w-full bg-line-bright" />
      </div>
    );
  }

  const stroke = highlighted ? "#FFB020" : "#8A97A6";
  const fill = highlighted ? "rgba(255,176,32,0.22)" : "rgba(138,151,166,0.16)";

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={className}
      role="img"
      aria-label={`Mastery ${(estimate.mastery * 100).toFixed(0)} percent, ${describeConfidence(
        estimate.confidence,
      )} from ${estimate.evidenceCount} result${estimate.evidenceCount === 1 ? "" : "s"}`}
    >
      {/* Baseline: the 0–1 axis the density sits on. */}
      <line x1={0} y1={height - 0.5} x2={width} y2={height - 0.5} stroke="#2E3742" strokeWidth={1} />
      <path d={path} fill={fill} stroke={stroke} strokeWidth={1} strokeLinejoin="round" />
      {/* The posterior mean, so a precise reading is available beside the shape. */}
      <line
        x1={estimate.mastery * width}
        y1={1}
        x2={estimate.mastery * width}
        y2={height}
        stroke={stroke}
        strokeWidth={1}
        strokeDasharray="1.5 1.5"
        opacity={0.7}
      />
    </svg>
  );
}

export function describeConfidence(confidence: number): string {
  if (confidence >= 0.75) return "well established";
  if (confidence >= 0.5) return "reasonably confident";
  if (confidence >= 0.25) return "early signal";
  return "barely measured";
}
