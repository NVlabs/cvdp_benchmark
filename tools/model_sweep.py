#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Model sweep tool for CVDP benchmark.

Runs one or more models/agents across one or more dataset variants and
produces a summary table (datasets × models/agents) plus a JSON report.

Usage:
  # Sweep two models across all public v1.1.0 nonagentic datasets:
  python tools/model_sweep.py -m openai-gpt-5.2 nemotron-ultra-253b -f nonagentic

  # Sweep an agent:
  python tools/model_sweep.py -g my-agent:latest -f code_generation

  # Use a custom model factory and resume partial run:
  python tools/model_sweep.py -m nemotron-super-49b -c custom_factory/custom_model_factory.py -s
"""

import argparse
import glob
import json
import os
import subprocess
import sys


DEFAULT_DATASET_DIR = os.path.expanduser("~/for_github/cvdp-benchmark-dataset")
DEFAULT_DATASET_VERSION = "v1.1.0"


def discover_datasets(
    dataset_dir: str,
    filter_str: str | None = None,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    include_additional: bool = False,
    include_examples: bool = False,
    include_with_solutions: bool = False,
) -> list[str]:
    """Return sorted list of matching public dataset files."""
    pattern = os.path.join(dataset_dir, f"cvdp_{dataset_version}_*.jsonl")
    files = sorted(glob.glob(pattern))
    if not include_with_solutions:
        files = [f for f in files if not os.path.basename(f).endswith("_with_solutions.jsonl")]
    if not include_additional:
        files = [f for f in files if "_additional" not in os.path.basename(f)]
    if not include_examples:
        prefix = f"cvdp_{dataset_version}_example_"
        files = [f for f in files if not os.path.basename(f).startswith(prefix)]
    if filter_str:
        files = [f for f in files if filter_str in os.path.basename(f)]
    return files


def variant_name(dataset_file: str, dataset_version: str = DEFAULT_DATASET_VERSION) -> str:
    """Extract a short variant name from a dataset filename.

    cvdp_v1.1.0_nonagentic_code_generation_no_commercial.jsonl
    -> nonagentic_code_generation_no_commercial
    """
    base = os.path.basename(dataset_file)
    name = base
    prefix = f"cvdp_{dataset_version}_"
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    return name


def _slug(name: str) -> str:
    """Sanitize a model/agent name for use in a directory name."""
    return name.replace("/", "_").replace(":", "-")


def work_dir_name(variant: str, model_or_agent: str) -> str:
    """Return the work directory name for a given variant and model/agent."""
    return f"work_sweep_{variant}_{_slug(model_or_agent)}"


def run_benchmark(
    dataset_file: str,
    work_dir: str,
    threads: int,
    n: int,
    k: int,
    model: str | None,
    agent: str | None,
    custom_factory: str | None,
    force_copilot: bool,
) -> bool:
    """
    Invoke run_samples.py for the given dataset/model/agent combination.

    Returns True if the subprocess succeeded, False otherwise.
    """
    cmd = [
        sys.executable, "run_samples.py",
        "-k", str(k),
        "-n", str(n),
        "-t", str(threads),
        "-f", dataset_file,
        "-p", work_dir,
        "-l",
    ]
    if model:
        cmd += ["-m", model]
    if agent:
        cmd += ["-g", agent]
    if custom_factory:
        cmd += ["-c", custom_factory]
    if force_copilot:
        cmd += ["--force-copilot"]

    print(f"  Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: run_samples.py exited with code {e.returncode}")
        return False


def get_pass_stats(work_dir: str) -> tuple[int, int]:
    """
    Parse composite_report.json in work_dir.

    Returns (passed, total).
    """
    report_path = os.path.join(work_dir, "composite_report.json")
    if not os.path.exists(report_path):
        print(f"  WARNING: No composite_report.json found at {report_path}")
        return 0, 0

    with open(report_path) as f:
        report = json.load(f)

    samples = report.get("samples", [])
    if not samples:
        return 0, 0

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

    return passed, total


def build_sweep_table(results: list[dict], model_agents: list[str]) -> None:
    """Print an ASCII grid: rows=datasets, columns=models/agents, cells=pass%."""
    variants = []
    seen = set()
    for r in results:
        v = r["variant"]
        if v not in seen:
            variants.append(v)
            seen.add(v)

    # Build lookup: (variant, model_or_agent) -> pass_pct string
    lookup: dict[tuple[str, str], str] = {}
    for r in results:
        key = (r["variant"], r["model_or_agent"])
        if r["total"] == 0:
            lookup[key] = "N/A"
        else:
            lookup[key] = f"{r['pass_pct']:.1f}%"

    # Column widths
    var_col = max(len("Dataset"), max((len(v) for v in variants), default=7))
    ma_cols = [max(len(ma), 7) for ma in model_agents]

    header = f"{'Dataset':<{var_col}}"
    for ma, w in zip(model_agents, ma_cols):
        header += f"  {ma:>{w}}"
    sep = "-" * len(header)

    print()
    print("=== Sweep Summary ===")
    print(sep)
    print(header)
    print(sep)
    for v in variants:
        row = f"{v:<{var_col}}"
        for ma, w in zip(model_agents, ma_cols):
            cell = lookup.get((v, ma), "-")
            row += f"  {cell:>{w}}"
        print(row)
    print(sep)
    print()


def write_json_report(results: list[dict], output_path: str) -> None:
    """Write structured JSON sweep report."""
    report = {"sweep": results}
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Sweep report written to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep one or more models/agents across dataset variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing dataset files (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--dataset-version",
        default=DEFAULT_DATASET_VERSION,
        help=f"Dataset version to discover (default: {DEFAULT_DATASET_VERSION})",
    )
    parser.add_argument(
        "-f", "--filter",
        dest="filter_str",
        default=None,
        help="Substring filter on dataset filename (e.g. 'nonagentic')",
    )
    parser.add_argument(
        "-m", "--models",
        nargs="+",
        default=[],
        metavar="MODEL",
        help="One or more model names to sweep (e.g. openai-gpt-5.2 nemotron-ultra-253b)",
    )
    parser.add_argument(
        "-g", "--agents",
        nargs="+",
        default=[],
        metavar="AGENT",
        help="One or more agent Docker image names to sweep",
    )
    parser.add_argument(
        "-c", "--custom-factory",
        default=None,
        help="Path to custom model factory (passed to run_samples.py -c)",
    )
    parser.add_argument(
        "--force-copilot",
        action="store_true",
        help="Pass --force-copilot to run_samples.py",
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=4,
        help="Number of parallel threads (default: 4)",
    )
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=1,
        help="Number of samples per run (default: 1)",
    )
    parser.add_argument(
        "-k", "--k-threshold",
        type=int,
        default=1,
        help="Pass@k threshold (default: 1)",
    )
    parser.add_argument(
        "-s", "--skip-existing",
        action="store_true",
        help="Skip if work directory already exists (resumable runs)",
    )
    parser.add_argument(
        "--include-additional",
        action="store_true",
        help="Include additional dataset variants in discovery",
    )
    parser.add_argument(
        "--include-examples",
        action="store_true",
        help="Include example dataset variants in discovery",
    )
    parser.add_argument(
        "--include-with-solutions",
        action="store_true",
        help="Include _with_solutions dataset variants in discovery",
    )
    parser.add_argument(
        "--output",
        default="sweep_report.json",
        help="Path to write sweep_report.json (default: sweep_report.json)",
    )
    args = parser.parse_args()

    if not args.models and not args.agents:
        parser.error("At least one of --models (-m) or --agents (-g) must be provided")

    # Discover dataset files
    dataset_files = discover_datasets(
        dataset_dir=args.dataset_dir,
        filter_str=args.filter_str,
        dataset_version=args.dataset_version,
        include_additional=args.include_additional,
        include_examples=args.include_examples,
        include_with_solutions=args.include_with_solutions,
    )
    if not dataset_files:
        print(f"ERROR: No dataset files found in {args.dataset_dir} matching 'cvdp_{args.dataset_version}_*.jsonl'")
        if args.filter_str:
            print(f"       (with filter: '{args.filter_str}')")
        return 1

    # Build flat list of (model_or_agent, is_agent) pairs
    targets: list[tuple[str, bool]] = []
    for m in args.models:
        targets.append((m, False))
    for a in args.agents:
        targets.append((a, True))

    total_runs = len(dataset_files) * len(targets)
    print(f"Found {len(dataset_files)} dataset file(s), {len(targets)} model/agent(s) => {total_runs} total runs\n")

    results: list[dict] = []

    for dataset_file in dataset_files:
        variant = variant_name(dataset_file, args.dataset_version)
        print(f"=== Variant: {variant} ===")

        for model_or_agent, is_agent in targets:
            work_dir = work_dir_name(variant, model_or_agent)
            label = f"{'AGENT' if is_agent else 'MODEL'}:{model_or_agent}"

            if args.skip_existing and os.path.exists(work_dir):
                print(f"  [{label}] Skipping (work dir exists): {work_dir}")
            else:
                print(f"  [{label}] Starting run -> {work_dir}")
                run_benchmark(
                    dataset_file=dataset_file,
                    work_dir=work_dir,
                    threads=args.threads,
                    n=args.n_samples,
                    k=args.k_threshold,
                    model=None if is_agent else model_or_agent,
                    agent=model_or_agent if is_agent else None,
                    custom_factory=args.custom_factory,
                    force_copilot=args.force_copilot,
                )

            passed, total = get_pass_stats(work_dir)
            pass_pct = 100.0 * passed / total if total else 0.0

            status = f"{passed}/{total} ({pass_pct:.1f}%)" if total else "N/A (no results)"
            print(f"  [{label}] {status}")

            results.append({
                "dataset_file": dataset_file,
                "variant": variant,
                "model_or_agent": model_or_agent,
                "work_dir": work_dir,
                "passed": passed,
                "total": total,
                "pass_pct": pass_pct,
            })

        print()

    # Summary table
    all_targets = [t[0] for t in targets]
    build_sweep_table(results, all_targets)

    # JSON report
    write_json_report(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
