import { describe, expect, it } from "vitest";
import { depthOf, layoutByDepth } from "@/features/roadmap/layout";
import type { ConceptNode, RoadmapEdge } from "@/lib/types";

/**
 * The layout is the one piece of graph reasoning the frontend does for itself, so it
 * gets the same treatment as the backend's algorithms: the invariant is that a node
 * never sits at or above a prerequisite, because an edge running upwards would tell
 * the learner the opposite of the truth.
 */

function node(slug: string, orderIndex: number, dependsOn: string[] = []): ConceptNode {
  return {
    conceptId: slug,
    slug,
    name: slug,
    domain: "mathematics",
    difficulty: 3,
    estimatedMinutes: 60,
    status: "not_started",
    nodeType: "topic",
    orderIndex,
    dependsOn,
  };
}

function edge(source: string, target: string): RoadmapEdge {
  return { source, target, strength: 1 };
}

describe("layoutByDepth", () => {
  it("puts everything with no prerequisites on the first row", () => {
    const nodes = [node("a", 0), node("b", 1), node("c", 2, ["a"])];
    const positioned = layoutByDepth(nodes, [edge("a", "c")]);

    expect(positioned.get("a")?.depth).toBe(0);
    expect(positioned.get("b")?.depth).toBe(0);
    expect(positioned.get("c")?.depth).toBe(1);
  });

  it("places a node below its deepest prerequisite, not its shallowest", () => {
    // d depends on a (depth 0) and on c (depth 2). Taking the shortest path would put
    // d at depth 1, above work it actually requires.
    const nodes = [node("a", 0), node("b", 1, ["a"]), node("c", 2, ["b"]), node("d", 3, ["a", "c"])];
    const edges = [edge("a", "b"), edge("b", "c"), edge("a", "d"), edge("c", "d")];
    const positioned = layoutByDepth(nodes, edges);

    expect(positioned.get("d")?.depth).toBe(3);
  });

  it("never places a node at or above one of its prerequisites", () => {
    const nodes = [node("a", 0), node("b", 1, ["a"]), node("c", 2, ["a", "b"]), node("d", 3, ["b"])];
    const edges = [edge("a", "b"), edge("a", "c"), edge("b", "c"), edge("b", "d")];
    const positioned = layoutByDepth(nodes, edges);

    for (const { source, target } of edges) {
      expect(positioned.get(target)!.position.y).toBeGreaterThan(
        positioned.get(source)!.position.y,
      );
    }
  });

  it("centres each row so the path reads as a spine", () => {
    const nodes = [node("a", 0), node("b", 1), node("c", 2)];
    const positioned = layoutByDepth(nodes, []);
    const xs = [...positioned.values()].map((entry) => entry.position.x);

    expect(xs.reduce((sum, x) => sum + x, 0)).toBeCloseTo(0);
  });

  it("orders a row by the planner's sequence, not by discovery", () => {
    const nodes = [node("late", 5), node("early", 1)];
    const positioned = layoutByDepth(nodes, []);

    expect(positioned.get("early")!.position.x).toBeLessThan(positioned.get("late")!.position.x);
  });

  it("reports the number of prerequisite layers", () => {
    const nodes = [node("a", 0), node("b", 1, ["a"]), node("c", 2, ["b"])];
    const positioned = layoutByDepth(nodes, [edge("a", "b"), edge("b", "c")]);

    expect(depthOf(positioned)).toBe(3);
  });

  it("handles an empty roadmap", () => {
    expect(layoutByDepth([], []).size).toBe(0);
  });
});
