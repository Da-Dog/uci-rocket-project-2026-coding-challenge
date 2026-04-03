from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent


@dataclass
class SuiteSummary:
    label: str
    total: int
    passed: int
    failures: int
    errors: int
    skipped: int

    @property
    def successful(self) -> bool:
        return self.failures == 0 and self.errors == 0


def discover(relative_path: str) -> unittest.TestSuite:
    loader = unittest.TestLoader()
    return loader.discover(
        start_dir=str(ROOT / relative_path),
        pattern="test_*.py",
        top_level_dir=str(ROOT),
    )


def summarize(label: str, result: unittest.TestResult) -> SuiteSummary:
    failures = len(result.failures) + len(getattr(result, "unexpectedSuccesses", []))
    errors = len(result.errors)
    skipped = len(getattr(result, "skipped", [])) + len(getattr(result, "expectedFailures", []))
    passed = max(0, result.testsRun - failures - errors - skipped)
    return SuiteSummary(
        label=label,
        total=result.testsRun,
        passed=passed,
        failures=failures,
        errors=errors,
        skipped=skipped,
    )


def run_suite(label: str, relative_path: str, *, verbosity: int) -> SuiteSummary:
    print(f"\n== {label} ==")
    suite = discover(relative_path)
    stream = sys.stdout if verbosity > 1 else io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=verbosity)
    result = runner.run(suite)
    summary = summarize(label, result)
    print(
        f"{summary.label}: {summary.passed}/{summary.total} passed "
        f"({summary.failures} failures, {summary.errors} errors, {summary.skipped} skipped)"
    )
    if verbosity == 1 and not summary.successful:
        print("Run with --verbose to see full failure details.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the shared buffer score harness.")
    parser.add_argument("--module", default="solution", help="Module name to score. Defaults to solution.")
    parser.add_argument(
        "--include-applicant-tests",
        dest="include_applicant_tests",
        action="store_true",
        help="Also run tests under tests/applicant.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any official test fails.",
    )
    parser.add_argument("--verbose", action="store_true", help="Use verbose unittest output.")
    args = parser.parse_args()

    os.environ["SHARED_BUFFER_MODULE"] = args.module
    verbosity = 2 if args.verbose else 1

    print(f"Scoring module: {args.module}")
    official = run_suite("Official Tests", "tests/official", verbosity=verbosity)

    applicant = None
    if args.include_applicant_tests:
        applicant = run_suite("Applicant Tests", "tests/applicant", verbosity=verbosity)

    if applicant is None:
        print(f"\nCurrent official score: {official.passed}/{official.total}")
    else:
        print(
            f"\nCurrent score: {official.passed}/{official.total} official, "
            f"{applicant.passed}/{applicant.total} applicant"
        )

    if args.strict and not official.successful:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
