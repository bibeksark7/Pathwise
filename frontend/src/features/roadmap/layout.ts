import type { ConceptNode, RoadmapEdge } from "@/lib/types";

/**
 * Position nodes by prerequisite depth.
 *
 * A roadmap is a DAG, and the reading a learner needs from it is "what comes before
 * what". So depth — the longest path from any root — becomes the vertical axis, and
 * everything at the same depth sits on one row. Anything that can be started now is
 * literally on the top line.
 *
 * Longest path rather than shortest, deliberately: a concept is only genuinely
 * reachable once *every* prerequisite is done, so its row should reflect the slowest
 * of them. Shortest path would float a node above work it actually depends on and
 * make edges run upwards, which is exactly the confusion the layout exists to avoid.
 *
 * Dagre would do this too, but it is a dependency and a black box for what amounts to
 * a longest-path pass over a graph that is already topologically ordered.
 */

const COLUMN_WIDTH = 268;
const ROW_HEIGHT = 168;

export interface PositionedNode {
  id: string;
  position: { x: number; y: number };
  depth: number;
}

export function layoutByDepth(
  nodes: ConceptNode[],
  edges: RoadmapEdge[],
): Map<string, PositionedNode> {
  const incoming = new Map<string, string[]>();
  for (const node of nodes) incoming.set(node.conceptId, []);
  for (const edge of edges) {
    incoming.get(edge.target)?.push(edge.source);
  }

  // The backend already emits nodes in topological order, so one forward pass
  // suffices: every prerequisite has its depth before its dependent is reached.
  const depth = new Map<string, number>();
  for (const node of nodes) {
    const parents = incoming.get(node.conceptId) ?? [];
    const deepestParent = parents.reduce(
      (deepest, parent) => Math.max(deepest, (depth.get(parent) ?? 0) + 1),
      0,
    );
    depth.set(node.conceptId, deepestParent);
  }

  const rows = new Map<number, ConceptNode[]>();
  for (const node of nodes) {
    const level = depth.get(node.conceptId) ?? 0;
    const row = rows.get(level) ?? [];
    row.push(node);
    rows.set(level, row);
  }

  const positioned = new Map<string, PositionedNode>();
  for (const [level, row] of rows) {
    // Stable ordering within a row, and centred so the graph reads as a spine
    // rather than drifting left.
    const ordered = [...row].sort((a, b) => a.orderIndex - b.orderIndex);
    const offset = ((ordered.length - 1) * COLUMN_WIDTH) / 2;

    ordered.forEach((node, index) => {
      positioned.set(node.conceptId, {
        id: node.conceptId,
        position: { x: index * COLUMN_WIDTH - offset, y: level * ROW_HEIGHT },
        depth: level,
      });
    });
  }

  return positioned;
}

/** How many prerequisite layers deep the path runs. */
export function depthOf(positioned: Map<string, PositionedNode>): number {
  let deepest = 0;
  for (const node of positioned.values()) deepest = Math.max(deepest, node.depth);
  return deepest + 1;
}
