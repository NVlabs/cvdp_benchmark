#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sanity check script for CVDP benchmark dataset variants.

Validates that:
  - Golden mode (apply golden patch): every dataset variant passes 100% of tests
  - No-patch mode (-d flag): every dataset variant passes 0% of tests

Usage:
  python tools/sanity_check.py [options]

Examples:
  # Quick smoke test with one variant:
  python tools/sanity_check.py --filter example_nonagentic_code_generation_no_commercial -t 4

  # Full sanity check (all variants):
  python tools/sanity_check.py -t 4

  # Only run golden checks, skip no-patch:
  python tools/sanity_check.py --only-golden -t 4

  # Resume a partial run (skip already-completed work dirs):
  python tools/sanity_check.py -s -t 4
"""

import argparse
import glob
import json
import os
import subprocess
import sys


DEFAULT_DATASET_DIR = os.path.expanduser("~/cvdp_benchmark_dataset")
DATASET_VERSION = "v1.1.0"
DATASET_GLOB = f"cvdp_{DATASET_VERSION}_*_with_solutions.jsonl"


def discover_datasets(dataset_dir: str, filter_str: str | None = None) -> list[str]:
    """Return sorted list of matching dataset files."""
    pattern = os.path.join(dataset_dir, DATASET_GLOB)
    files = sorted(glob.glob(pattern))
    if filter_str:
        files = [f for f in files if filter_str in os.path.basename(f)]
    return files


def variant_name(dataset_file: str) -> str:
    """Extract a short variant name from a dataset filename.

    cvdp_v1.1.0_nonagentic_code_generation_no_commercial_with_solutions.jsonl
    -> nonagentic_code_generation_no_commercial
    """
    base = os.path.basename(dataset_file)
    prefix = f"cvdp_{DATASET_VERSION}_"
    name = base
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith("_with_solutions.jsonl"):
        name = name[: -len("_with_solutions.jsonl")]
    return name


def work_dir_name(variant: str, mode: str) -> str:
    """Return the work directory name for a given variant and mode."""
    return f"work_sanity_{variant}_{mode}"


def run_benchmark(dataset_file: str, work_dir: str, threads: int, no_patch: bool) -> bool:
    """
    Invoke run_samples.py with k=1, n=1 for the given dataset.

    Returns True if the subprocess succeeded, False otherwise.
    """
    cmd = [
        sys.executable, "run_samples.py",
        "-k", "1",
        "-n", "1",
        "-t", str(threads),
        "-f", dataset_file,
        "-p", work_dir,
    ]
    if no_patch:
        cmd.append("-d")

    print(f"  Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: run_samples.py exited with code {e.returncode}")
        return False


def get_pass_stats(work_dir: str) -> tuple[int, int, list[dict], list[dict]]:
    """
    Parse composite_report.json in work_dir.

    Returns (passed, total, failing_tests, passing_tests).
    failing_tests / passing_tests are lists of dicts from test_details.
    """
    report_path = os.path.join(work_dir, "composite_report.json")
    if not os.path.exists(report_path):
        print(f"  WARNING: No composite_report.json found at {report_path}")
        return 0, 0, [], []

    with open(report_path) as f:
        report = json.load(f)

    samples = report.get("samples", [])
    if not samples:
        return 0, 0, [], []

    # Use the first (and only) sample for n=1 runs
    sample = samples[0]

    passed = 0
    total = 0
    skip_keys = {"metadata", "test_details", "sample_index"}
    for key, cat_data in sample.items():
        if key in skip_keys:
            continue
        for difficulty in ("easy", "medium", "hard"):
            if difficulty in cat_data:
                passed += cat_data[difficulty].get("Passed Problems", 0)
                total += cat_data[difficulty].get("Total Problems", 0)

    test_details = sample.get("test_details", {})
    failing_tests = test_details.get("failing_tests", []) or []
    passing_tests = test_details.get("passing_tests", []) or []

    # Filter passing_tests to only include problems where ALL tests passed.
    # passing_tests is test-level (any sub-test that returned 0), but Passed Problems
    # is problem-level (all sub-tests must pass).  A problem that has both a passing
    # sub-test and a failing sub-test must not appear in the "should have failed" list.
    failing_ids = {t.get("test_id") for t in failing_tests}
    passing_tests = [t for t in passing_tests if t.get("test_id") not in failing_ids]

    return passed, total, failing_tests, passing_tests


def build_rerun_cmd(dataset_file: str, work_dir: str, threads: int, no_patch: bool, test_id: str) -> str:
    """Build the re-run command for a single failing/unexpected-passing test."""
    parts = [
        "./run_samples.py",
        "-k", "1",
        "-n", "1",
        "-t", str(threads),
        "-f", dataset_file,
        "-p", work_dir,
    ]
    if no_patch:
        parts.append("-d")
    parts += ["-i", test_id]
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanity check: validate golden=100%% and no-patch=0%% for all dataset variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing dataset files (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=4,
        help="Number of parallel threads to pass to run_samples.py (default: 4)",
    )
    parser.add_argument(
        "-s", "--skip-existing",
        action="store_true",
        help="Skip variants where the work directory already exists (resumable runs)",
    )
    parser.add_argument(
        "--only-golden",
        action="store_true",
        help="Only run golden mode checks (skip no-patch)",
    )
    parser.add_argument(
        "--only-nopatch",
        action="store_true",
        help="Only run no-patch mode checks (skip golden)",
    )
    parser.add_argument(
        "-f", "--filter",
        dest="filter_str",
        default=None,
        help="Substring filter on dataset filename (e.g. 'nonagentic')",
    )
    args = parser.parse_args()

    if args.only_golden and args.only_nopatch:
        parser.error("--only-golden and --only-nopatch are mutually exclusive")

    # Discover dataset files
    dataset_files = discover_datasets(args.dataset_dir, args.filter_str)
    if not dataset_files:
        print(f"ERROR: No dataset files found in {args.dataset_dir} matching '{DATASET_GLOB}'")
        if args.filter_str:
            print(f"       (with filter: '{args.filter_str}')")
        return 1

    modes = []
    if not args.only_nopatch:
        modes.append(("golden", False))
    if not args.only_golden:
        modes.append(("nopatch", True))

    total_runs = len(dataset_files) * len(modes)
    print(f"Found {len(dataset_files)} dataset file(s), {len(modes)} mode(s) => {total_runs} total runs\n")

    # Track failures for final report
    failures: list[dict] = []

    for dataset_file in dataset_files:
        variant = variant_name(dataset_file)
        print(f"=== Variant: {variant} ===")

        for mode_label, no_patch in modes:
            work_dir = work_dir_name(variant, mode_label)
            composite_path = os.path.join(work_dir, "composite_report.json")

            # Skip if requested and work dir exists
            if args.skip_existing and os.path.exists(work_dir):
                print(f"  [{mode_label.upper()}] Skipping (work dir exists): {work_dir}")
            else:
                print(f"  [{mode_label.upper()}] Starting run -> {work_dir}")
                run_benchmark(dataset_file, work_dir, args.threads, no_patch)

            # Parse results
            passed, total, failing_tests, passing_tests = get_pass_stats(work_dir)

            if total == 0:
                print(f"  [{mode_label.upper()}] WARNING: 0 total problems found in report (run may have failed)")
                failures.append({
                    "mode": mode_label,
                    "dataset_file": dataset_file,
                    "work_dir": work_dir,
                    "passed": 0,
                    "total": 0,
                    "bad_tests": [],
                    "no_patch": no_patch,
                    "error": "No results found",
                })
                continue

            pct = 100.0 * passed / total if total else 0.0

            if mode_label == "golden":
                ok = (passed == total)
                bad_tests = failing_tests
                status = "PASS" if ok else "FAIL"
                print(f"  [{mode_label.upper()}] {status}: {passed}/{total} ({pct:.1f}%) passed")
                if not ok:
                    failures.append({
                        "mode": "golden",
                        "dataset_file": dataset_file,
                        "work_dir": work_dir,
                        "passed": passed,
                        "total": total,
                        "bad_tests": bad_tests,
                        "no_patch": no_patch,
                    })
            else:  # nopatch
                ok = (passed == 0)
                bad_tests = passing_tests
                status = "PASS" if ok else "FAIL"
                print(f"  [{mode_label.upper()}] {status}: {passed}/{total} ({pct:.1f}%) passed (expected 0)")
                if not ok:
                    failures.append({
                        "mode": "nopatch",
                        "dataset_file": dataset_file,
                        "work_dir": work_dir,
                        "passed": passed,
                        "total": total,
                        "bad_tests": bad_tests,
                        "no_patch": no_patch,
                    })

        print()

    # ---- Final report ----
    if not failures:
        print("╔══════════════════════════════════════╗")
        print("║  ALL CHECKS PASSED                   ║")
        print("╚══════════════════════════════════════╝")
        print(f"\n{total_runs} run(s) completed successfully.")
        return 0

    # Print failure details
    print("╔══════════════════════════════════════════════════════╗")
    print("║  SANITY CHECK FAILURES                               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    for failure in failures:
        mode = failure["mode"]
        fname = os.path.basename(failure["dataset_file"])
        passed = failure["passed"]
        total = failure["total"]
        work_dir = failure["work_dir"]
        no_patch = failure["no_patch"]
        bad_tests = failure.get("bad_tests", [])
        error = failure.get("error")

        if mode == "golden":
            pct = 100.0 * passed / total if total else 0.0
            print(f"[GOLDEN FAIL] {fname}")
            if error:
                print(f"  {error}")
            else:
                print(f"  Expected 100% pass rate, got {pct:.1f}% ({passed}/{total})")
                print(f"  Failing tests:")
                for t in bad_tests:
                    tid = t.get("test_id", "?")
                    cat = t.get("category", "?")
                    diff = t.get("difficulty", "?")
                    log = t.get("log", "")
                    print(f"    - {tid}  [{cat}/{diff}]")
                    if log:
                        print(f"        Log: {log}")
                print()
                print(f"  Re-run commands (for debugging):")
                for t in bad_tests:
                    tid = t.get("test_id", "?")
                    cmd = build_rerun_cmd(failure["dataset_file"], work_dir, args.threads, no_patch, tid)
                    print(f"    {cmd}")
        else:
            pct = 100.0 * passed / total if total else 0.0
            print(f"[NOPATCH FAIL] {fname}")
            if error:
                print(f"  {error}")
            else:
                print(f"  Expected 0% pass rate, got {pct:.1f}% ({passed}/{total})")
                print(f"  Tests that passed (should have failed):")
                for t in bad_tests:
                    tid = t.get("test_id", "?")
                    cat = t.get("category", "?")
                    diff = t.get("difficulty", "?")
                    log = t.get("log", "")
                    print(f"    - {tid}  [{cat}/{diff}]")
                    if log:
                        print(f"        Log: {log}")
                print()
                print(f"  Re-run commands:")
                for t in bad_tests:
                    tid = t.get("test_id", "?")
                    cmd = build_rerun_cmd(failure["dataset_file"], work_dir, args.threads, no_patch, tid)
                    print(f"    {cmd}")

        print()

    n_failed = len(failures)
    print(f"SUMMARY: {n_failed} variant(s) failed out of {total_runs} runs.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
