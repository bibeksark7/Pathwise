/**
 * Design tokens for Pathwise.
 *
 * The palette is a drafting table under a lamp: a blue-shifted graphite ground,
 * hairline structure, chalk-white ink — and exactly one saturated colour.
 *
 * `signal` (sodium amber) is reserved for a single meaning: **what to do next**. It
 * appears on the recommended node, the primary action, and nowhere else. Every other
 * state is distinguished by treatment — a hairline, a left bar, a dash, reduced
 * opacity — rather than by competing for attention with its own hue. Six states in
 * six bright colours would make the one that matters invisible.
 */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ground: "#14181D",
        surface: "#1C222A",
        raised: "#232B35",
        line: "#2E3742",
        "line-bright": "#3E4A58",
        ink: "#E8EDF2",
        muted: "#8A97A6",
        faint: "#5A6674",

        /** The only saturated colour. Means "do this next" and nothing else. */
        signal: "#FFB020",
        "signal-dim": "#7A5514",

        /** Quiet status hues, used at low opacity for fills and hairlines only. */
        ok: "#4FD1A5",
        warn: "#E8705A",
      },
      fontFamily: {
        display: ['"Space Grotesk"', "system-ui", "sans-serif"],
        sans: ['"IBM Plex Sans"', "system-ui", "sans-serif"],
        /** Every number and every concept slug. Slugs are identifiers, and setting
         *  them as measurements rather than prose is the honest treatment. */
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
      },
      fontSize: {
        micro: ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.06em" }],
        eyebrow: ["0.75rem", { lineHeight: "1rem", letterSpacing: "0.14em" }],
      },
      borderRadius: { node: "3px" },
      boxShadow: {
        /** A lamp on the drafting table, not a drop shadow. */
        signal: "0 0 0 1px #FFB020, 0 0 24px -6px rgba(255,176,32,0.45)",
        drawer: "-24px 0 48px -24px rgba(0,0,0,0.7)",
      },
      transitionTimingFunction: { instrument: "cubic-bezier(0.2, 0, 0, 1)" },
    },
  },
  plugins: [],
};
