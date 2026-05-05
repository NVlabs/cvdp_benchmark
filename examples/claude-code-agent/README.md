# Claude Code Agent for CVDP

A Docker-based agent that uses [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) to solve CVDP hardware verification challenges in an agentic workflow.

## Prerequisites

- Docker
- A Claude Code auth method. For benchmark containers, prefer an environment-token auth path such as `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN`.
- (Commercial only) Access to Cadence EDA tools and a license server

## Building

```bash
cd examples/claude-code-agent/

# Build both images (non-commercial + commercial)
./build_agent.sh

# Or build only the non-commercial image
docker build -t claude-code-agent .

# Optionally pin the Claude Code CLI version
docker build --build-arg CLAUDE_CODE_VERSION=2.1.126 -t claude-code-agent .
```

To override the default license server for the commercial build:

```bash
CDS_LIC_FILE='port@server' LM_LICENSE_FILE='port@server' ./build_agent.sh
```

This produces two Docker images:

| Image | Base | Use case |
|-------|------|----------|
| `claude-code-agent` | `nvidia/cvdp-sim:v1.0.0` | OSS simulators (Icarus, Verilator, Yosys) |
| `claude-code-agent-commercial` | RockyLinux 9 | Cadence Xcelium, cocotb, coverage tools |

## Running

Set one Claude auth path and run the benchmark from the repo root:

```bash
# Direct Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# Or, for Claude subscription/SSO auth, generate a long-lived token once:
claude setup-token
export CLAUDE_CODE_OAUTH_TOKEN=...

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

| Setting | Type | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Environment variable | Direct Anthropic API key. Preferred for reproducible API-billed runs. |
| `ANTHROPIC_AUTH_TOKEN` | Environment variable | Bearer token for an LLM gateway or proxy. |
| `CLAUDE_CODE_OAUTH_TOKEN` | Environment variable | Long-lived OAuth token from `claude setup-token`; useful for Claude subscription or SSO-backed auth in non-interactive containers. |
| `CVDP_CLAUDE_AUTH_MODE` | Environment variable | Optional. `env` (default), `oauth`, `helper`, or `host`. |
| `CVDP_CLAUDE_SETTINGS` | Environment variable | Optional. Claude settings JSON string or path, primarily for `apiKeyHelper` auth. |
| `CVDP_CLAUDE_MODEL` | Environment variable | Optional. Model passed to `claude --model`. |
| `CVDP_CLAUDE_MAX_BUDGET_USD` | Environment variable | Optional. Cost cap passed to `claude --max-budget-usd`. |
| `CVDP_CLAUDE_USE_BARE` | Environment variable | Optional. Override bare mode. By default, bare mode is used only with API-key or `apiKeyHelper` auth. |
| `CVDP_CLAUDE_ALLOWED_TOOLS` | Environment variable | Optional. Tool allow-list passed to `claude --allowedTools`. |
| `CVDP_CLAUDE_DISALLOWED_TOOLS` | Environment variable | Optional. Tool deny-list passed to `claude --disallowedTools`. |
| `CVDP_CLAUDE_ENABLE_TELEMETRY` | Environment variable | Optional. Set to `1` to opt in to Claude Code telemetry. Telemetry is off by default for benchmark logs. |
| `CVDP_CLAUDE_INCLUDE_WORKSPACE_RULES` | Environment variable | Optional. Set to `0` to disable the Claude wrapper's benchmark workspace rules. |
| `CLAUDE_CODE_VERSION` | Docker build argument | Optional. Pin the Claude Code npm package version for reproducibility. |
| `CLAUDE_CODE_MAX_TURNS` | Environment variable | Optional. Limit the number of agent turns for fair comparison. |
| `DOCKER_TIMEOUT_AGENT` | Environment variable | Optional. Agent container timeout in seconds (set in `.env`). Defaults to 600; longer agentic runs often need a value such as 1800. |

### Auth Modes

`CVDP_CLAUDE_AUTH_MODE=env` is the default. It accepts `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN`, or configured cloud-provider auth environment variables. The Claude image declares its supported pass-through environment variables with the `org.cvdp.agent.env` Docker label, so auth values are not written into generated compose files.

Use `CVDP_CLAUDE_AUTH_MODE=oauth` to require `CLAUDE_CODE_OAUTH_TOKEN` specifically. This is the preferred path when you want to use an existing Claude subscription or SSO-backed organization without copying host login state into the container.

Use `CVDP_CLAUDE_AUTH_MODE=helper` with `CVDP_CLAUDE_SETTINGS` when Claude Code should call an `apiKeyHelper` script. The helper script must be available inside the container.

Use `CVDP_CLAUDE_AUTH_MODE=host` only as an explicit best-effort fallback. The Claude image declares conditional host-auth mounts with the `org.cvdp.agent.mounts` Docker label, so the benchmark mounts `~/.claude/.credentials.json`, `~/.claude/settings.json`, and `~/.claude.json` read-only only when host auth is requested and those files exist. The wrapper copies them into a temporary writable container home. This is most likely to work for Linux file-backed Claude Code credentials. It is not a portable replacement for macOS Keychain or other host credential stores, and it should not be the default benchmark path.

### Agent Image Metadata

The benchmark runner is agent-neutral. Agent images can declare runtime needs with Docker labels:

```dockerfile
LABEL org.cvdp.agent.env="ANTHROPIC_API_KEY,CLAUDE_CODE_OAUTH_TOKEN,CVDP_CLAUDE_AUTH_MODE"
LABEL org.cvdp.agent.mounts="env:CVDP_CLAUDE_AUTH_MODE=host:~/.claude.json:/host-claude/.claude.json:ro"
```

Use `CVDP_AGENT_ENV` or `CVDP_AGENT_MOUNTS` to add pass-through environment variables or host mounts without rebuilding an image. `CVDP_AGENT_MOUNTS` uses the same semicolon-separated format as the label.

## How It Works

1. The benchmark mounts challenge files into the container at `/code/` (docs, RTL, verification files, and `prompt.json`).
2. `agent.py` reads the task from `/code/prompt.json` and prepends Claude-specific benchmark workspace rules unless disabled.
3. Claude Code CLI is invoked in non-interactive mode with the task piped via stdin. API-key and `apiKeyHelper` runs use `--bare` by default for reproducibility; OAuth-token and host-auth runs do not because bare mode does not read OAuth/keychain credentials.
4. Claude iteratively reads, modifies, and tests files in the `/code/` workspace.
5. The workspace rules protect benchmark infrastructure and generated artifacts without exposing golden patch paths or expected solution files.
6. Prompt history and session persistence are disabled for benchmark runs. Web search/fetch are disabled where supported by the CLI, but the container still needs network access for Claude API calls.
7. The container exits and the benchmark evaluates the agent's accepted project-file changes against the test harness.

## Agent Status

The benchmark records agent execution metadata separately from harness test results. Agent statuses include `completed`, `completed_no_patch`, `failed`, `timeout`, and `budget_exceeded`. Timeouts use `DOCKER_TIMEOUT_AGENT`.

Harness pass rate is computed from harness tests only. Agent failures, timeouts, ignored generated artifacts, and benchmark-contract violations are reported in metadata instead of being inserted as synthetic failed tests.

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
