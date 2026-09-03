"""The evaluation runner.

    python -m pathwise.evaluation.run --suite all

Runs every suite, prints a table, and **exits non-zero on a regression**. That last
part is what makes it a gate rather than a report: a number that nobody is obliged to
look at does not prevent anything.

Comparison is against a stored baseline (`evals/baseline.json`), not against a fixed
target. Absolute thresholds either sit so low they never fire or so high they block
every commit; a baseline asks the only question that matters on a pull request — is
this worse than what we had?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathwise.evaluation.scorers import SuiteResult
from pathwise.evaluation.suites import DATASETS_DIR, build_suites, load_cases, load_graph
from pathwise.logging_config import configure_logging

BASELINE_PATH = DATASETS_DIR.parent / "baseline.json"

#: Scores wobble in the last decimal place from floating-point ordering. Anything
#: below this is noise, and failing CI on noise trains people to ignore the gate.
REGRESSION_TOLERANCE = 0.001


@dataclass(frozen=True, slots=True)
class Regression:
    """One metric that got worse."""

    suite: str
    metric: str
    baseline: float
    current: float

    @property
    def delta(self) -> float:
        return self.current - self.baseline

    def __str__(self) -> str:
        return (
            f"{self.suite}/{self.metric}: {self.baseline:.3f} -> {self.current:.3f} "
            f"({self.delta:+.3f})"
        )


def run_suites(names: list[str]) -> dict[str, SuiteResult]:
    """Run the named suites against the real knowledge graph."""
    graph = load_graph()
    available = build_suites(graph)

    if "all" in names:
        names = list(available)

    results: dict[str, SuiteResult] = {}
    for name in names:
        suite = available.get(name)
        if suite is None:
            raise SystemExit(f"Unknown suite '{name}'. Available: {', '.join(sorted(available))}")
        results[name] = suite.run(load_cases(name), graph)

    return results


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, dict[str, float]]:
    """The accepted reference scores, or empty when none has been recorded."""
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8")).get("suites", {})
        return dict(loaded) if isinstance(loaded, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def write_baseline(results: dict[str, SuiteResult], path: Path = BASELINE_PATH) -> None:
    """Record current scores as the new reference."""
    payload: dict[str, Any] = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "suites": {
            name: {
                **result.aggregate_scores(),
                "pass_rate": result.pass_rate,
            }
            for name, result in results.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_regressions(
    results: dict[str, SuiteResult], baseline: dict[str, dict[str, float]]
) -> list[Regression]:
    """Metrics that fell below their recorded baseline.

    Only ever compares metrics the baseline already contains. A newly added scorer is
    not a regression — there is nothing to have regressed from — and treating it as
    one would make adding a check painful, which is the opposite of the intent.
    """
    regressions: list[Regression] = []

    for name, result in results.items():
        reference = baseline.get(name, {})
        current = {**result.aggregate_scores(), "pass_rate": result.pass_rate}

        for metric, previous in reference.items():
            now = current.get(metric)
            if now is None:
                continue
            if now < previous - REGRESSION_TOLERANCE:
                regressions.append(Regression(name, metric, previous, now))

    return regressions


def format_results(results: dict[str, SuiteResult], *, verbose: bool = False) -> str:
    """A readable report."""
    lines: list[str] = []

    for name, result in results.items():
        status = "PASS" if result.all_passed else "FAIL"
        lines.append(
            f"\n{status}  {name}  "
            f"({result.passed_count}/{len(result.cases)} cases, "
            f"pass rate {result.pass_rate:.0%})"
        )

        for metric, value in sorted(result.aggregate_scores().items()):
            lines.append(f"        {metric:26} {value:.3f}")

        for case in result.failures():
            lines.append(f"    x {case.case_id}")
            for score in case.failures:
                lines.append(f"        {score.name}: {score.detail}")
            if verbose:
                lines.append(f"        actual: {dict(case.actual)}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Pathwise AI evaluation suites.")
    parser.add_argument(
        "--suite",
        nargs="+",
        default=["all"],
        help="Suite names, or 'all' (the default).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Record the current scores as the new reference. Review the diff before "
        "committing it — this is how an accepted improvement is locked in, and also "
        "how a regression would be silently accepted.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit 0 when no datasets exist yet, instead of failing.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show actual output on failure.")
    args = parser.parse_args(argv)

    configure_logging(level="WARNING", json_output=False)

    try:
        results = run_suites(list(args.suite))
    except Exception as exc:
        if args.allow_empty and "No dataset" in str(exc):
            print("No evaluation datasets found; nothing to run.")
            return 0
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 2

    print(format_results(results, verbose=args.verbose))

    total_cases = sum(len(result.cases) for result in results.values())
    total_passed = sum(result.passed_count for result in results.values())
    print(f"\n{total_passed}/{total_cases} cases passed across {len(results)} suite(s).")

    if args.update_baseline:
        write_baseline(results)
        print(f"Baseline updated: {BASELINE_PATH}")
        return 0

    baseline = load_baseline()
    if not baseline:
        print(
            "\nNo baseline recorded yet. Run with --update-baseline once the scores "
            "above are ones you are willing to defend."
        )
        # Without a reference, case failures are still failures.
        return 0 if total_passed == total_cases else 1

    regressions = find_regressions(results, baseline)
    if regressions:
        print("\nREGRESSIONS against baseline:", file=sys.stderr)
        for regression in regressions:
            print(f"  {regression}", file=sys.stderr)
        return 1

    if total_passed < total_cases:
        print("\nSome cases failed, though no metric regressed.", file=sys.stderr)
        return 1

    print("No regressions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
