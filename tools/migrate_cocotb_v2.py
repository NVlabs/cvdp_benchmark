#!/usr/bin/env python3
"""Migrate cocotb v1 API calls to cocotb v2.0 in CVDP JSONL dataset files.

This script processes JSONL dataset files and updates all Python test harness
files from cocotb v1 API to cocotb v2.0 compatible API.

Changes applied:
  Breaking changes (cause errors in cocotb 2.0):
    - from cocotb.binary import BinaryValue  -> try/except with cocotb.types
    - from cocotb.result import TestFailure   -> TestFailure = AssertionError
    - from cocotb.result import TestSuccess   -> class TestSuccess(Exception)
    - import cocotb.result                    -> try/except
    - .value.to_unsigned()                    -> int(.value)
    - .value.integer                          -> int(.value)
    - .value.signed_integer                   -> int(.value)
    - from cocotb.runner import get_runner    -> try/except with cocotb_tools
    - Timer(0, ...)                           -> Timer(1, unit="ps")

  Deprecation warnings (still work but emit warnings in cocotb 2.0):
    - units="ns" / units='ns'                -> unit="ns" / unit='ns'
    - @cocotb.coroutine + def                -> async def
    - await cocotb.start(...)                 -> cocotb.start_soon(...)

Usage:
    python tools/migrate_cocotb_v2.py datasets/*.jsonl
    python tools/migrate_cocotb_v2.py --dry-run datasets/*.jsonl
"""

import json
import re
import sys
import argparse
from pathlib import Path


def migrate_python_source(text: str) -> str:
    """Apply all cocotb v1 -> v2 migrations to a Python source string."""
    orig = text

    # ---- Breaking changes ----

    # 1. cocotb.binary.BinaryValue
    text = text.replace(
        "from cocotb.binary import BinaryValue",
        "from cocotb.types import LogicArray as BinaryValue",
    )

    # 2. cocotb.result.TestFailure
    text = text.replace(
        "from cocotb.result import TestFailure",
        "TestFailure = AssertionError",
    )

    # 3. cocotb.result.TestSuccess
    text = text.replace(
        "from cocotb.result import TestSuccess",
        "class TestSuccess(Exception): pass",
    )

    # 4. import cocotb.result
    text = text.replace(
        "import cocotb.result",
        "# cocotb.result removed in v2",
    )

    # 5. .value.to_unsigned() -> int(.value)
    # Match identifiers with optional array indexing like dut.sig[0].value.to_unsigned()
    text = re.sub(
        r'(\w+(?:(?:\.\w+)|\[\d+\])*)\s*\.\s*value\s*\.\s*to_unsigned\s*\(\s*\)',
        r'int(\1.value)',
        text,
    )

    # 6. .value.integer -> int(.value)
    # Must match .value.integer but NOT .value.signed_integer
    # Process signed_integer first to avoid partial matches
    text = re.sub(
        r'(\w+(?:(?:\.\w+)|\[\d+\])*)\s*\.\s*value\s*\.\s*signed_integer\b',
        r'int(\1.value)',
        text,
    )
    text = re.sub(
        r'(\w+(?:(?:\.\w+)|\[\d+\])*)\s*\.\s*value\s*\.\s*integer\b',
        r'int(\1.value)',
        text,
    )

    # 7. cocotb.runner — use a sentinel to avoid double-replacement
    _RUNNER_SENTINEL = "from __cocotb_runner_migrated__ import get_runner"
    text = text.replace(
        "from cocotb.runner import get_runner",
        _RUNNER_SENTINEL,
    )
    text = text.replace(
        _RUNNER_SENTINEL,
        "try:\n    from cocotb_tools.runner import get_runner\n"
        "except ImportError:\n    from cocotb.runner import get_runner",
    )

    # 8. Timer(0, ...) -> Timer(1, unit="ps")
    # Match Timer(0, units="ns") or Timer(0, unit="ns") etc.
    text = re.sub(
        r'Timer\(\s*0\s*,\s*unit(?:s)?\s*=\s*["\'][^"\']+["\']\s*\)',
        'Timer(1, unit="ps")',
        text,
    )
    # Match bare Timer(0)
    text = re.sub(
        r'Timer\(\s*0\s*\)',
        'Timer(1, unit="ps")',
        text,
    )

    # ---- Deprecation warnings ----

    # 9. units= -> unit= (in Timer, Clock, etc.)
    text = re.sub(r'\bunits\s*=\s*"', 'unit="', text)
    text = re.sub(r"\bunits\s*=\s*'", "unit='", text)

    # 10. @cocotb.coroutine -> remove, ensure async def
    text = re.sub(
        r'@cocotb\.coroutine\s*\n(\s*)(async\s+def|def)\s+',
        r'\1async def ',
        text,
    )

    # 11. await cocotb.start(...) -> cocotb.start_soon(...)
    # cocotb.start() returns a Task and is awaitable in v1
    # cocotb.start_soon() is the v2 replacement (not awaitable)
    text = re.sub(
        r'await\s+cocotb\.start\s*\(',
        'cocotb.start_soon(',
        text,
    )
    # Also handle non-awaited cocotb.start( that isn't start_soon
    text = re.sub(
        r'(?<!_)cocotb\.start\s*\((?!_)',
        'cocotb.start_soon(',
        text,
    )

    return text


def process_jsonl_file(filepath: Path, dry_run: bool = False) -> dict:
    """Process a single JSONL file, migrating all Python harness files.

    Returns a dict with migration statistics.
    """
    stats = {
        "file": str(filepath),
        "entries": 0,
        "py_files_total": 0,
        "py_files_changed": 0,
        "entries_changed": 0,
    }

    lines = filepath.read_text().splitlines()
    new_lines = []
    changed = False

    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue

        entry = json.loads(line)
        stats["entries"] += 1
        entry_changed = False

        harness_files = entry.get("harness", {}).get("files", {})
        for fpath, content in list(harness_files.items()):
            if not fpath.endswith(".py"):
                continue
            stats["py_files_total"] += 1

            new_content = migrate_python_source(content)
            if new_content != content:
                harness_files[fpath] = new_content
                stats["py_files_changed"] += 1
                entry_changed = True

        if entry_changed:
            stats["entries_changed"] += 1
            changed = True

        new_lines.append(json.dumps(entry, ensure_ascii=False))

    if changed and not dry_run:
        filepath.write_text("\n".join(new_lines) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate cocotb v1 API to v2 in CVDP JSONL datasets"
    )
    parser.add_argument("files", nargs="+", help="JSONL files to process")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files",
    )
    args = parser.parse_args()

    total_stats = {
        "files": 0,
        "entries": 0,
        "py_files_total": 0,
        "py_files_changed": 0,
        "entries_changed": 0,
    }

    for filepath in args.files:
        p = Path(filepath)
        if not p.exists():
            print(f"  SKIP {p} (not found)")
            continue

        stats = process_jsonl_file(p, dry_run=args.dry_run)
        total_stats["files"] += 1
        for k in ["entries", "py_files_total", "py_files_changed", "entries_changed"]:
            total_stats[k] += stats[k]

        action = "would change" if args.dry_run else "changed"
        print(
            f"  {p.name}: {stats['py_files_changed']}/{stats['py_files_total']} "
            f"Python files {action} across {stats['entries_changed']}/{stats['entries']} entries"
        )

    print(f"\nTotal: {total_stats['py_files_changed']}/{total_stats['py_files_total']} "
          f"Python files across {total_stats['files']} JSONL files")


if __name__ == "__main__":
    main()
