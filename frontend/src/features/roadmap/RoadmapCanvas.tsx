import { useCallback, useMemo, useRef } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  ReactFlow,
  type Edge,
  type Node,
  type ReactFlowInstance,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { ConceptNodeCard, type ConceptNodeData } from "@/features/roadmap/ConceptNode";
import { layoutByDepth } from "@/features/roadmap/layout";
import type { ConceptNode, RoadmapEdge } from "@/lib/types";

/**
 * The prerequisite graph, laid out by depth.
 *
 * The edges are the product. A list of topics is a syllabus; a list of topics with
 * the dependencies drawn is an argument for the order, and it is what makes a removed
 * or inserted step legible as a consequence rather than an arbitrary change.
 */

interface Props {
  nodes: ConceptNode[];
  edges: RoadmapEdge[];
  selectedSlug: string | null;
  onSelect: (slug: string) => void;
}

const nodeTypes = { concept: ConceptNodeCard };

/** Kept in step with the card in `ConceptNode.tsx`, for centring maths only. */
const NODE_WIDTH = 232;

/** Fraction of the pane height above the focused node on first paint. */
const FOCUS_FROM_TOP = 0.5 - 0.22;

export function RoadmapCanvas({ nodes, edges, selectedSlug, onSelect }: Props) {
  const pane = useRef<HTMLDivElement>(null);
  const recommended = useMemo(
    () => nodes.find((node) => node.status === "recommended"),
    [nodes],
  );

  const flowNodes = useMemo<Node[]>(() => {
    const positions = layoutByDepth(nodes, edges);

    return nodes.map((concept) => ({
      id: concept.conceptId,
      type: "concept",
      position: positions.get(concept.conceptId)?.position ?? { x: 0, y: 0 },
      draggable: false,
      data: {
        concept,
        onOpen: onSelect,
        isSelected: concept.slug === selectedSlug,
      } satisfies ConceptNodeData,
    }));
  }, [nodes, edges, selectedSlug, onSelect]);

  const flowEdges = useMemo<Edge[]>(
    () =>
      edges.map((edge) => ({
        id: `${edge.source}->${edge.target}`,
        source: edge.source,
        target: edge.target,
        type: "smoothstep",
        // Only edges feeding the recommended node are drawn in the signal colour: they
        // are the reason it is available now, which is the one relationship worth
        // pointing at on a graph this dense.
        animated: recommended !== undefined && edge.target === recommended.conceptId,
        // A weak prerequisite is genuinely a weaker claim, and the line says so.
        style: edge.strength < 0.85 ? { strokeDasharray: "4 3" } : undefined,
      })),
    [edges, recommended],
  );

  /**
   * Open centred on the recommended node at full size, rather than fitting the whole
   * path.
   *
   * A thirteen-step graph fitted to the viewport is thirteen unreadable rectangles.
   * The reason to arrive here is to see what to do next and what it rests on, so that
   * is what the opening frame shows; the fit-view control is right there for the
   * overview. If a specific concept was linked to, that one wins.
   */
  const focus = useMemo(() => {
    const positions = layoutByDepth(nodes, edges);
    const target =
      nodes.find((node) => node.slug === selectedSlug) ??
      recommended ??
      nodes[0];
    if (!target) return null;
    const position = positions.get(target.conceptId);
    return position ? { x: position.position.x + NODE_WIDTH / 2, y: position.position.y } : null;
    // Only the first frame is placed; afterwards the viewport belongs to the user.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onInit = useCallback(
    (instance: ReactFlowInstance) => {
      if (!focus) {
        instance.fitView({ padding: 0.2, maxZoom: 1 });
        return;
      }

      // On a phone a full-size card fills the width, so pull back a little.
      const narrow = window.innerWidth < 640;
      const zoom = narrow ? 0.75 : 1;

      // Sit the focus node near the top of the pane rather than dead centre, so the
      // space below it shows what it unlocks. `setCenter` places a point at the
      // middle, hence the offset back down by the difference.
      const height = pane.current?.clientHeight ?? 0;
      instance.setCenter(focus.x, focus.y + (height * FOCUS_FROM_TOP) / zoom, {
        zoom,
        duration: 0,
      });
    },
    [focus],
  );

  return (
    <div ref={pane} className="h-full w-full">
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        nodeTypes={nodeTypes}
        onInit={onInit}
        minZoom={0.2}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
        nodesConnectable={false}
        className="bg-ground"
        onPaneClick={() => onSelect("")}
      >
        {/* Two grids at two scales, drawn by React Flow so they pan and zoom with the
            nodes — a static background would slide against the content and break the
            illusion that this is one drawing. */}
        <Background id="fine" variant={BackgroundVariant.Lines} gap={24} lineWidth={1} color="#1D242C" />
        <Background id="coarse" variant={BackgroundVariant.Lines} gap={120} lineWidth={1} color="#232B35" />
        <Controls showInteractive={false} position="bottom-right" />
        <Legend />
      </ReactFlow>
    </div>
  );
}

/**
 * Six states carried by treatment rather than by six colours need a key, once.
 * It sits in a corner at low contrast: findable when you want it, ignorable when you
 * do not.
 */
function Legend() {
  const entries: [string, string][] = [
    ["Next", "border-signal bg-raised"],
    ["Ready", "border-line bg-surface"],
    ["In progress", "border-line-bright bg-raised"],
    ["Done", "border-ok/35 bg-ok/[0.06]"],
    ["Review due", "border-warn/50 border-dashed bg-warn/[0.05]"],
    ["Locked", "border-line bg-surface/60 opacity-[0.62]"],
  ];

  return (
    <div className="pointer-events-none absolute bottom-3 left-3 z-10 hidden flex-wrap gap-x-4 gap-y-1.5 sm:flex rounded-node border border-line bg-surface/85 px-3 py-2 backdrop-blur">
      {entries.map(([label, swatch]) => (
        <span key={label} className="flex items-center gap-1.5 font-mono text-micro text-muted">
          <span className={`h-2.5 w-4 rounded-[1px] border ${swatch}`} aria-hidden="true" />
          {label}
        </span>
      ))}
    </div>
  );
}
