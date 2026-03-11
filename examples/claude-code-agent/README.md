# Claude Code Agent for CVDP

A Docker-based agent that uses [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) to solve CVDP hardware verification challenges in an agentic workflow.

## Prerequisites

- Docker
- An Anthropic API key (`sk-ant-...`)
- (Commercial only) Access to Cadence EDA tools and a license server

## Building

```bash
cd examples/claude-code-agent/

# Build both images (non-commercial + commercial)
./build_agent.sh

# Or build only the non-commercial image
docker build -t claude-code-agent .
```

To override the default license server for the commercial build:

```bash
CDS_LIC_FILE='port@server' LM_LICENSE_FILE='port@server' ./build_agent.sh
```

This produces two Docker images:

| Image | Base | Use case |
|-------|------|----------|
| `claude-code-agent` | Ubuntu 22.04 | OSS simulators (Icarus, Verilator) |
| `claude-code-agent-commercial` | RockyLinux 9 | Cadence Xcelium, cocotb, coverage tools |

## Running

Set your API key and run the benchmark from the repo root:

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Single run
python run_benchmark.py -f dataset.jsonl -l -g claude-code-agent

# Single datapoint
python run_benchmark.py -f dataset.jsonl -i cvdp_agentic_issue_0001 -l -g claude-code-agent

# Multi-sample Pass@k evaluation
python run_samples.py -f dataset.jsonl -l -g claude-code-agent -n 5 -k 1

# Commercial image (with EDA tools)
python run_benchmark.py -f dataset.jsonl -l -g claude-code-agent-commercial
```

## Configuration

| Environment Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | **Required.** Anthropic API key. |
| `CLAUDE_CODE_MAX_TURNS` | Optional. Limit the number of agent turns for fair comparison. |
| `DOCKER_TIMEOUT_AGENT` | Optional. Agent container timeout in seconds (set in `.env`). |

## How It Works

1. The benchmark mounts challenge files into the container at `/code/` (docs, RTL, verification files, and `prompt.json`).
2. `agent.py` reads the task from `/code/prompt.json`.
3. Claude Code CLI is invoked in non-interactive project mode (`claude -p --dangerously-skip-permissions`) with the task piped via stdin.
4. Claude iteratively reads, modifies, and tests files in the `/code/` workspace.
5. Web search/fetch are disabled to ensure the agent works only with local files.
6. The container exits and the benchmark evaluates the agent's changes against the test harness.

## Debugging

```bash
# Run a benchmark first to generate work directories
python run_benchmark.py -f dataset.jsonl -i cvdp_agentic_issue_0001 -l -g claude-code-agent

# Inspect agent changes
cat work/cvdp_agentic_issue_0001/harness/1/agent_changes.patch

# Interactive debug shell inside the agent container
cd work/cvdp_agentic_issue_0001/harness/1/
./run_docker_agent.sh -d
```

## See Also

- [Agentic Workflow Guide](../../README_AGENTIC.md) — full documentation on the agentic benchmark workflow
- [Example Agent](../agent/) — minimal reference agent implementation
