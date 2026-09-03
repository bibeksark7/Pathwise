import { describe, expect, it } from "vitest";
import { factorLabel, formatDuration, formatPercent } from "@/lib/format";

describe("formatDuration", () => {
  it("keeps short sessions in minutes", () => {
    expect(formatDuration(45)).toBe("45m");
  });

  it("switches to hours at an hour", () => {
    expect(formatDuration(60)).toBe("1.0h");
    expect(formatDuration(150)).toBe("2.5h");
  });

  it("drops the decimal once it stops carrying information", () => {
    // "40.0h" implies a precision the estimate does not have.
    expect(formatDuration(2400)).toBe("40h");
  });
});

describe("formatPercent", () => {
  it("rounds to a whole percent", () => {
    expect(formatPercent(0.3875)).toBe("39%");
    expect(formatPercent(1)).toBe("100%");
  });
});

describe("factorLabel", () => {
  it("names the known decision factors", () => {
    expect(factorLabel("goal_relevance")).toBe("Goal relevance");
  });

  it("degrades readably if the backend adds a factor", () => {
    expect(factorLabel("cost_of_delay")).toBe("cost of delay");
  });
});
