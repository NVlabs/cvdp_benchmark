# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import tempfile
import unittest
from pathlib import Path

from src.adapters.harbor.cvdp import (
    build_raw_results_from_harbor_job,
    convert_dataset,
)


class HarborAdapterConversionTest(unittest.TestCase):

    def test_converts_agentic_row_to_harbor_layout(self):
        row = {
            "id": "cvdp_agentic_widget_0001",
            "categories": ["cid003", "easy"],
            "prompt": "Write /code/rtl/widget.sv using /code/docs/spec.md",
            "context": {
                "docs/spec.md": "spec",
                "verif/widget_tb.sv": "tb",
            },
            "harness": {
                "docker-compose.yml": "services: {}",
                "Dockerfile": "FROM scratch",
                "src/.env": (
                    "SIM = icarus\n"
                    "VERILOG_SOURCES = /code/rtl/widget.sv ./verif/widget_tb.sv\n"
                    "PYTHONPATH = /src\n"
                    "HASH = deadbeef\n"
                ),
                "src/test_runner.py": "print('run')\n",
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "cvdp_agentic_code_generation_no_commercial.jsonl"
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = Path(tmp) / "datasets" / "cvdp"

            result = convert_dataset(dataset, output, workspace_hint=True)

            task_dir = output / "no_commercial" / "cid003" / "widget_0001"
            self.assertEqual(result.count, 1)
            self.assertTrue((task_dir / "task.toml").is_file())
            self.assertTrue((task_dir / "environment" / "Dockerfile").is_file())
            self.assertTrue((task_dir / "tests" / "test.sh").is_file())
            self.assertTrue((task_dir / "environment" / "workspace" / "code" / "rtl").is_dir())
            self.assertEqual(
                (task_dir / "environment" / "workspace" / "code" / "docs" / "spec.md").read_text(encoding="utf-8"),
                "spec",
            )
            self.assertFalse((task_dir / "tests" / "docker-compose.yml").exists())

            instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
            self.assertIn("rtl/widget.sv", instruction)
            self.assertIn("docs/spec.md", instruction)
            self.assertNotIn("/code/", instruction)
            self.assertIn("## Workspace layout", instruction)

            env = (task_dir / "tests" / "src" / ".env").read_text(encoding="utf-8")
            self.assertIn('VERILOG_SOURCES="/sandbox/workspace/code/rtl/widget.sv ./verif/widget_tb.sv"', env)
            self.assertIn('PYTHONPATH="/tests/src"', env)
            self.assertNotIn("HASH", env)

    def test_converts_nonagentic_row_when_shape_is_unambiguous(self):
        row = {
            "id": "cvdp_copilot_gadget_0002",
            "categories": ["cid004", "medium"],
            "input": {
                "prompt": "Create gadget",
                "context": {"docs/spec.md": "spec"},
            },
            "output": {"context": {"rtl/gadget.sv": ""}},
            "harness": {"files": {"src/test_runner.py": "print('run')\n"}},
        }

        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "custom.jsonl"
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = Path(tmp) / "harbor"

            convert_dataset(dataset, output, split="no_commercial")

            task_dir = output / "no_commercial" / "cid004" / "cvdp_copilot_gadget_0002"
            self.assertTrue((task_dir / "instruction.md").is_file())
            self.assertTrue((task_dir / "tests" / "src" / "test_runner.py").is_file())
            self.assertTrue((task_dir / "environment" / "workspace" / "code" / "docs" / "spec.md").is_file())

    def test_extracts_compose_command_into_verifier_plan(self):
        # A commercial coverage task grades via the docker-compose service
        # command (a raw simulator invocation), not a test_runner.py. The
        # converter must flatten that command into the replayable plan, with
        # container paths remapped and the command base64-encoded.
        row = {
            "id": "cvdp_agentic_coverage_0001",
            "categories": ["cid012", "hard"],
            "prompt": "Improve coverage",
            "context": {"rtl/dut.sv": "module dut; endmodule"},
            "harness": {
                "docker-compose.yml": (
                    "services:\n"
                    "  sim:\n"
                    "    image: cadence\n"
                    "    working_dir: /code/rundir\n"
                    "    env_file: ./src/.env\n"
                    "    command: xrun -coverage all /src/*.sv /code/verif/*.sv\n"
                ),
                "src/.env": "SIM = xcelium\n",
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "cvdp_agentic_code_generation_commercial.jsonl"
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = Path(tmp) / "harbor"

            convert_dataset(dataset, output, split="commercial")

            task_dir = output / "commercial" / "cid012" / "coverage_0001"
            plan_path = task_dir / "tests" / ".harbor_verifier_plan"
            self.assertTrue(plan_path.is_file())
            # docker-compose.yml is parsed for the plan, not copied as a harness file.
            self.assertFalse((task_dir / "tests" / "docker-compose.yml").exists())

            env_file, workdir, command_b64 = (
                plan_path.read_text(encoding="utf-8").strip().split("\t")
            )
            command = base64.b64decode(command_b64).decode()
            self.assertEqual(workdir, "/sandbox/workspace/code/rundir")
            self.assertEqual(env_file, "/tests/src/.env")
            # Container paths remapped into Harbor's layout.
            self.assertIn("/tests/src/*.sv", command)
            self.assertIn("/sandbox/workspace/code/verif/*.sv", command)
            # No -timescale injection on this branch.
            self.assertNotIn("-timescale", command)


class HarborAdapterReportImportTest(unittest.TestCase):

    def test_imports_harbor_job_rewards_as_raw_results(self):
        row = {
            "id": "cvdp_agentic_widget_0001",
            "categories": ["cid003", "easy"],
            "prompt": "Write widget",
            "context": {},
            "harness": {},
            "patch": {"rtl/widget.sv": ""},
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

            trial = root / "jobs" / "job1" / "widget_0001__AbC123"
            (trial / "verifier").mkdir(parents=True)
            (trial / "verifier" / "reward.json").write_text('{"reward": 1, "accuracy": 1}\n', encoding="utf-8")
            (trial / "verifier" / "test-stdout.txt").write_text("ok\n", encoding="utf-8")
            (trial / "result.json").write_text(
                json.dumps({
                    "task_id": {"path": "datasets/cvdp/no_commercial/cid003/widget_0001"},
                    "verifier_result": {"rewards": {"reward": 1, "accuracy": 1}},
                    "verifier": {
                        "started_at": "2026-06-07T00:00:00Z",
                        "finished_at": "2026-06-07T00:00:03Z",
                    },
                }),
                encoding="utf-8",
            )

            raw_results = build_raw_results_from_harbor_job(root / "jobs" / "job1", dataset)

            self.assertEqual(set(raw_results), {"cvdp_agentic_widget_0001"})
            result = raw_results["cvdp_agentic_widget_0001"]
            self.assertEqual(result["category"], "cid003")
            self.assertEqual(result["difficulty"], "easy")
            self.assertEqual(result["errors"], 0)
            self.assertEqual(result["tests"][0]["result"], 0)
            self.assertEqual(result["tests"][0]["harbor_score"], 1.0)
            self.assertEqual(result["tests"][0]["execution"], 3.0)


if __name__ == "__main__":
    unittest.main()
