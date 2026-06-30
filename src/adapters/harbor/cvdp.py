# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert CVDP JSONL datasets to Harbor tasks and import Harbor scores.

The generated task layout mirrors the Harbor adapter in
``evals/adapters/cvdp``:

* CVDP context files are baked into ``environment/workspace/code`` and become
  visible under ``/sandbox/workspace/code``.
* CVDP harness files are written to ``tests/src`` and are only copied into
  ``/tests`` after the agent has finished.
* The verifier runs ``tests/test.sh`` and writes ``/logs/verifier/reward.json``.
"""

from __future__ import annotations

import base64
import datetime
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


ADAPTER_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = ADAPTER_DIR / "templates"

SKIP_HARNESS_FILES = {"docker-compose.yml", "Dockerfile"}
STANDARD_DIRS = ("rtl", "verif", "docs", "rundir")
INSTRUCTION_PATH_REWRITES = (
    ("/code/docs/", "docs/"),
    ("/code/rtl/", "rtl/"),
    ("/code/verif/", "verif/"),
    ("/code/rundir/", "rundir/"),
    ("/code/", ""),
)
HARNESS_PATH_REWRITES = (
    (r"(?<!\w)/code(?=/|$)", "/sandbox/workspace/code"),
    (r"(?<!\w)/src(?=/|$)", "/tests/src"),
    (r"(?<!\w)/rundir(?=/|$)", "/sandbox/workspace/code/rundir"),
)
DEFAULT_PLAN_WORKDIR = "/sandbox/workspace/code/rundir"


@dataclass
class HarborConversionResult:
    """Summary returned by a Harbor conversion."""

    output_dir: Path
    split: str
    count: int = 0
    task_dirs: list[Path] = field(default_factory=list)


def ensure_dict(value: Any) -> dict:
    """HuggingFace fields may arrive as JSON strings or dictionaries."""
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    return value


def task_dir_name(task_id: str) -> str:
    """Convert ``cvdp_agentic_fixed_arbiter_0001`` to ``fixed_arbiter_0001``."""
    return re.sub(r"^cvdp_agentic_", "", task_id).lower()


def split_name(name: str) -> str:
    """Derive the Harbor split directory name from a dataset/config name."""
    if "no_commercial" in name:
        return "no_commercial"
    if "commercial" in name:
        return "commercial"
    return "no_commercial"


def extract_category(categories: Any) -> str:
    """Extract the ``cidXXX`` category from a CVDP categories list."""
    if isinstance(categories, str):
        categories = json.loads(categories)
    for cat in categories:
        if isinstance(cat, str) and re.match(r"^cid\d+$", cat):
            return cat
    raise ValueError(f"No cidXXX category found in {categories}")


def extract_difficulty(categories: Any) -> str:
    """Extract CVDP difficulty from a categories list."""
    if isinstance(categories, str):
        categories = json.loads(categories)
    for cat in categories:
        if cat in {"easy", "medium", "hard"}:
            return cat
    return "medium"


def extract_output_paths(harness: dict) -> list[str]:
    """Extract expected RTL output paths from harness .env VERILOG_SOURCES."""
    env_content = None
    for key, val in harness.items():
        if key.endswith(".env") and val is not None:
            env_content = val
            break
    if not env_content:
        return []

    paths = []
    for line in env_content.splitlines():
        m = re.match(r"^VERILOG_SOURCES\s*=\s*\"?(.+?)\"?\s*$", line)
        if m:
            for src in m.group(1).split():
                src = src.strip('"')
                if src.startswith("/code/"):
                    src = src[len("/code/") :]
                elif src.startswith("./"):
                    src = src[2:]
                paths.append(src)
    return paths


def build_workspace_hint(prompt: str, harness: dict) -> str:
    """Build the optional workspace-layout hint from the reference adapter."""
    output_paths = extract_output_paths(harness)
    missing = [p for p in output_paths if p not in prompt]

    lines = [
        "",
        "## Workspace layout",
        "",
        "Your working directory is organised as follows:",
        "",
        "- `docs/` - specifications and documentation",
        "- `rtl/` - RTL source files (Verilog / SystemVerilog)",
        "- `verif/` - testbenches and verification files",
        "- `rundir/` - simulation working directory",
    ]

    if missing:
        lines.append("")
        lines.append(
            "Place your output in the appropriate directory. "
            "The expected output file path(s):"
        )
        for path in missing:
            lines.append(f"- `{path}`")

    return "\n".join(lines)


def remap_env(env_text: str) -> str:
    """Remap CVDP .env to bash-compatible syntax for Harbor."""
    lines = []
    for line in env_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\w+)\s*=\s*(.*)", line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if key == "HASH":
            continue
        value = value.replace("/code/", "/sandbox/workspace/code/")
        value = re.sub(r"(?<!\w)/src(?=/|$)", "/tests/src", value)
        lines.append(f'{key}="{value}"')
    return "\n".join(lines) + "\n"


def remap_harness_paths(text: str) -> str:
    """Remap hidden-harness absolute paths into Harbor container paths.

    Harness files (other than ``.env``) reference the original CVDP container
    layout (``/code``, ``/src``, ``/rundir``); rewrite those to where Harbor
    actually mounts them so replayed commands resolve.
    """
    remapped = text
    for pattern, replacement in HARNESS_PATH_REWRITES:
        remapped = re.sub(pattern, replacement, remapped)
    return remapped


def normalize_compose_path(value: str) -> str:
    """Convert a docker-compose path value to its verifier-container path."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    value = value.replace("./src/", "/tests/src/")
    value = value.replace("./src", "/tests/src")
    return remap_harness_paths(value)


def _compose_command_str(command: Any) -> str:
    """A docker-compose ``command`` may be a string or an exec-form list."""
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command)


def extract_verifier_plan(harness: dict) -> list[tuple[str, str, str]]:
    """Flatten docker-compose service commands into a replayable plan.

    CVDP grades by running one or more docker-compose services — a cocotb
    ``test_runner.py`` and, on some commercial tasks, a raw simulator command
    (e.g. ``xrun``). Harbor does not run docker-compose inside the verifier, so
    we parse the compose YAML and flatten every service's ``command`` — with its
    ``env_file`` and ``working_dir`` — into a plan that ``test.sh`` replays in
    the single verifier container.

    Parsing the YAML (rather than scraping lines) preserves block-scalar
    commands (``command: >`` / ``command: |``); returned commands may span
    multiple lines and are base64-encoded when written to the plan file so each
    record stays on one line.
    """
    compose = harness.get("docker-compose.yml")
    if not compose:
        return []

    doc = yaml.safe_load(compose)
    if not isinstance(doc, dict):
        return []

    plan: list[tuple[str, str, str]] = []
    for service in (doc.get("services") or {}).values():
        if not isinstance(service, dict) or "command" not in service:
            continue

        command = normalize_compose_path(_compose_command_str(service["command"]))
        if not command:
            continue

        env_file = service.get("env_file") or ""
        if isinstance(env_file, list):
            env_file = env_file[0] if env_file else ""
        env_file = normalize_compose_path(str(env_file)) if env_file else ""

        workdir = service.get("working_dir")
        workdir = (
            normalize_compose_path(str(workdir)) if workdir else DEFAULT_PLAN_WORKDIR
        )

        plan.append((env_file, workdir, command))

    return plan


def normalize_instruction_paths(prompt: str) -> str:
    """Rewrite original ``/code/...`` paths into workspace-relative paths."""
    normalized = prompt
    for old, new in INSTRUCTION_PATH_REWRITES:
        normalized = normalized.replace(old, new)
    return normalized


def docker_image_config_key(split: str) -> str:
    """Return the CVDP image config key for a Harbor split."""
    if split == "commercial":
        return "VERIF_EDA_IMAGE"
    return "OSS_SIM_IMAGE"


def configured_docker_image(split: str) -> str:
    """Read the verifier image from CVDP's central configuration."""
    from src.config_manager import config

    return config.get(docker_image_config_key(split))


def write_docker_environment(env_dir: Path, split: str) -> None:
    """Write Harbor's thin Docker wrapper around the configured CVDP image."""
    image = configured_docker_image(split)
    dockerfile = (
        f"FROM {image}\n\n"
        "COPY workspace/ /sandbox/workspace/\n"
        "WORKDIR /sandbox/workspace/code\n"
    )
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")


def row_to_harbor_fields(row: dict) -> tuple[dict, dict, str]:
    """Normalize agentic and non-agentic CVDP rows to Harbor inputs."""
    if "context" in row and "prompt" in row:
        context = ensure_dict(row["context"])
        harness = ensure_dict(row.get("harness", {}))
        if "files" in harness and isinstance(harness["files"], dict):
            harness = harness["files"]
        return context, harness, row["prompt"]

    if "input" in row:
        input_data = ensure_dict(row["input"])
        context = ensure_dict(input_data.get("context", {}))
        harness = ensure_dict(row.get("harness", {}))
        if "files" in harness and isinstance(harness["files"], dict):
            harness = harness["files"]
        return context, harness, input_data.get("prompt", "")

    raise ValueError(f"Unsupported CVDP row shape for id {row.get('id', '<unknown>')}")


def convert_task(
    row: dict,
    output_dir: Path,
    split: str,
    workspace_hint: bool = False,
) -> Path:
    """Convert one CVDP row into a Harbor task directory."""
    task_id = row["id"]
    dirname = task_dir_name(task_id)
    category = extract_category(row["categories"])
    task_dir = output_dir / split / category / dirname
    if task_dir.exists():
        shutil.rmtree(task_dir)

    context, harness, prompt = row_to_harbor_fields(row)

    normalized_prompt = normalize_instruction_paths(prompt)
    instruction = normalized_prompt.strip() + "\n"
    if workspace_hint:
        instruction += build_workspace_hint(normalized_prompt, harness) + "\n"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")

    shutil.copy2(TEMPLATES_DIR / "task.toml", task_dir / "task.toml")

    env_dir = task_dir / "environment"
    write_docker_environment(env_dir, split)

    code_dir = env_dir / "workspace" / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for directory in STANDARD_DIRS:
        (code_dir / directory).mkdir(parents=True, exist_ok=True)

    for filepath, content in context.items():
        if content is None:
            continue
        dest = code_dir / filepath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATES_DIR / "test.sh", tests_dir / "test.sh")

    # CVDP's canonical grading is the docker-compose service command(s). Harbor
    # can't run docker-compose in the verifier, so flatten them into a plan that
    # test.sh replays. Without this, commercial coverage tasks whose grading is a
    # raw simulator command (not a test_runner.py) would never run.
    verifier_plan = extract_verifier_plan(harness)
    if verifier_plan:
        # Commands are base64-encoded so multi-line / quoted commands survive as
        # a single tab-separated record; env_file and workdir stay plain.
        plan_lines = [
            "\t".join(
                (env_file, workdir, base64.b64encode(command.encode()).decode())
            )
            for env_file, workdir, command in verifier_plan
        ]
        (tests_dir / ".harbor_verifier_plan").write_text(
            "\n".join(plan_lines) + "\n", encoding="utf-8"
        )

    for filepath, content in harness.items():
        if filepath in SKIP_HARNESS_FILES or content is None:
            continue
        if Path(filepath).name.startswith(".env"):
            content = remap_env(content)
        else:
            content = remap_harness_paths(content)
        dest = tests_dir / filepath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    return task_dir


def convert_dataset(
    dataset_path: Union[str, Path],
    output_dir: Union[str, Path],
    split: Optional[str] = None,
    workspace_hint: bool = False,
) -> HarborConversionResult:
    """Convert a CVDP JSONL dataset to Harbor task directories."""
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    resolved_split = split or split_name(dataset_path.name)
    result = HarborConversionResult(output_dir=output_dir, split=resolved_split)

    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task_dir = convert_task(
                row=row,
                output_dir=output_dir,
                split=resolved_split,
                workspace_hint=workspace_hint,
            )
            result.count += 1
            result.task_dirs.append(task_dir)

    return result


def load_dataset_index(dataset_path: Union[str, Path]) -> dict[str, dict]:
    """Index CVDP rows by Harbor task directory name."""
    index = {}
    with Path(dataset_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            name = task_dir_name(row["id"])
            index[name] = {
                "id": row["id"],
                "category": extract_category(row["categories"]),
                "difficulty": extract_difficulty(row["categories"]),
            }
    return index


def strip_trial_suffix(name: str) -> str:
    """Strip Harbor's random trial suffix from ``task__AbCd123`` names."""
    return re.sub(r"__[A-Za-z0-9]+$", "", name)


def parse_iso_timestamp(value: Optional[str]) -> Optional[datetime.datetime]:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(normalized)
    except ValueError:
        return None


def elapsed_seconds(start: Optional[str], finish: Optional[str]) -> float:
    started = parse_iso_timestamp(start)
    finished = parse_iso_timestamp(finish)
    if not started or not finished:
        return 0.0
    return max((finished - started).total_seconds(), 0.0)


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def task_name_from_trial(trial_dir: Path, trial_result: dict) -> str:
    task_id = trial_result.get("task_id") or {}
    task_path = task_id.get("path")
    if not task_path:
        task_path = trial_result.get("config", {}).get("task", {}).get("path")
    if task_path:
        return Path(task_path).name
    return strip_trial_suffix(trial_dir.name)


def reward_from_trial(trial_dir: Path, trial_result: dict) -> dict:
    rewards = trial_result.get("verifier_result", {}).get("rewards")
    if isinstance(rewards, dict):
        return rewards
    return read_json_if_exists(trial_dir / "verifier" / "reward.json")


def build_raw_results_from_harbor_job(
    job_dir: Union[str, Path],
    dataset_path: Union[str, Path],
    reward_threshold: float = 1.0,
) -> dict:
    """Convert a Harbor job directory into CVDP ``raw_result.json`` shape."""
    job_dir = Path(job_dir)
    dataset_index = load_dataset_index(dataset_path)
    raw_results = {}

    for entry in sorted(job_dir.iterdir()):
        if not entry.is_dir():
            continue

        trial_result = read_json_if_exists(entry / "result.json")
        rewards = reward_from_trial(entry, trial_result)
        task_name = task_name_from_trial(entry, trial_result)
        metadata = dataset_index.get(task_name)
        if metadata is None:
            continue

        reward = rewards.get("reward")
        if reward is None:
            error_msg = "Harbor reward.json missing top-level reward"
            test_result = 1
            harbor_score = None
        else:
            error_msg = None
            harbor_score = float(reward)
            test_result = 0 if harbor_score >= reward_threshold else 1

        verifier_timing = trial_result.get("verifier", {})
        execution = elapsed_seconds(
            verifier_timing.get("started_at"),
            verifier_timing.get("finished_at"),
        )
        log_path = entry / "verifier" / "test-stdout.txt"
        if not log_path.exists():
            log_path = entry / "verifier" / "reward.json"

        raw_results[metadata["id"]] = {
            "category": metadata["category"],
            "difficulty": metadata["difficulty"],
            "tests": [
                {
                    "result": test_result,
                    "log": str(log_path) if log_path.exists() else None,
                    "error_msg": error_msg,
                    "execution": execution,
                    "harbor_score": harbor_score,
                    "harbor_rewards": rewards,
                }
            ],
            "errors": test_result,
        }

    return raw_results
