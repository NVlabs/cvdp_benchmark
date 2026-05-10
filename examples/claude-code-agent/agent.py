#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Claude Code agent implementation for the CVDP agentic workflow.
This agent reads prompt.json and uses Claude Code to make changes to files.
"""

import json
import subprocess
import sys
import os
import shutil

RUNTIME_HOME = "/tmp/claude-home"
HOST_CLAUDE_MOUNT = "/host-claude"

VALID_AUTH_MODES = {"env", "oauth", "helper", "host"}
CLOUD_PROVIDER_FLAGS = (
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)

BENCHMARK_WORKSPACE_RULES = """Benchmark workspace rules:
- Work inside /code and use the task prompt plus repository files to infer the required deliverables.
- Modify only project files needed for the task. The benchmark does not provide additional expected solution paths beyond the task and repository content.
- Do not modify benchmark infrastructure or generated artifacts, including prompt.json, Docker compose files, run scripts, hidden/reference solution files, run directories, generated logs, or cache directories.
- Do not alter benchmark harness infrastructure or generated output to force a passing score. If the task asks for RTL, verification code, documentation, assertions, or testbench work, make those changes in normal project files.
"""


def _has_env(env, key):
    value = env.get(key)
    return value is not None and value != ""


def _parse_bool(value, default=False):
    if value is None or value == "":
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    print(f"Warning: ignoring boolean value '{value}'")
    return default


def _copy_file_if_present(src, dst, mode=None):
    if not os.path.exists(src):
        return False

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    if mode is not None:
        os.chmod(dst, mode)
    return True


def _prepare_claude_home(env, auth_mode):
    os.makedirs(RUNTIME_HOME, exist_ok=True)
    claude_config_dir = os.path.join(RUNTIME_HOME, ".claude")
    os.makedirs(claude_config_dir, exist_ok=True)

    env["HOME"] = RUNTIME_HOME
    env["CLAUDE_CONFIG_DIR"] = claude_config_dir

    if auth_mode != "host":
        return False

    copied_credentials = False

    try:
        copied_credentials = _copy_file_if_present(
            os.path.join(HOST_CLAUDE_MOUNT, ".credentials.json"),
            os.path.join(claude_config_dir, ".credentials.json"),
            0o600,
        ) or copied_credentials
        copied_credentials = _copy_file_if_present(
            os.path.join(HOST_CLAUDE_MOUNT, ".claude.json"),
            os.path.join(RUNTIME_HOME, ".claude.json"),
            0o600,
        ) or copied_credentials
        copied_credentials = _copy_file_if_present(
            os.path.join(HOST_CLAUDE_MOUNT, ".claude.json"),
            os.path.join(claude_config_dir, ".claude.json"),
            0o600,
        ) or copied_credentials
        copied_settings = _copy_file_if_present(
            os.path.join(HOST_CLAUDE_MOUNT, "settings.json"),
            os.path.join(claude_config_dir, "settings.json"),
            0o600,
        )
        if copied_settings:
            print("Copied host Claude settings into the runtime config directory")
    except Exception as e:
        print(f"ERROR: Could not copy host Claude auth files: {e}", file=sys.stderr)
        sys.exit(1)

    if copied_credentials:
        print("Copied host Claude auth files into the runtime config directory")

    return copied_credentials


def _has_cloud_provider_auth(env):
    return any(_parse_bool(env.get(key), False) for key in CLOUD_PROVIDER_FLAGS)


def _resolve_auth_mode(env, host_auth_ready):
    auth_mode = env.get("CVDP_CLAUDE_AUTH_MODE", "env").strip().lower()
    if auth_mode not in VALID_AUTH_MODES:
        print(
            "ERROR: CVDP_CLAUDE_AUTH_MODE must be one of: env, oauth, helper, host",
            file=sys.stderr,
        )
        sys.exit(1)

    has_api_key = _has_env(env, "ANTHROPIC_API_KEY")
    has_auth_token = _has_env(env, "ANTHROPIC_AUTH_TOKEN")
    has_oauth_token = _has_env(env, "CLAUDE_CODE_OAUTH_TOKEN")
    has_settings = _has_env(env, "CVDP_CLAUDE_SETTINGS")
    has_cloud = _has_cloud_provider_auth(env)

    if auth_mode == "env":
        if has_cloud:
            return auth_mode, "cloud-provider"
        if has_auth_token:
            return auth_mode, "auth-token"
        if has_api_key:
            return auth_mode, "api-key"
        if has_oauth_token:
            return auth_mode, "oauth-token"

        print(
            "ERROR: no Claude auth was provided. Set ANTHROPIC_API_KEY, "
            "ANTHROPIC_AUTH_TOKEN, CLAUDE_CODE_OAUTH_TOKEN, or provider auth env vars.",
            file=sys.stderr,
        )
        sys.exit(1)

    if auth_mode == "oauth":
        if has_api_key or has_auth_token or has_cloud:
            print(
                "ERROR: oauth auth mode was requested, but a higher-precedence "
                "Claude auth method is also set. Unset API key, auth token, or "
                "cloud-provider auth variables for a pure OAuth-token run.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not has_oauth_token:
            print("ERROR: oauth auth mode requires CLAUDE_CODE_OAUTH_TOKEN", file=sys.stderr)
            sys.exit(1)
        return auth_mode, "oauth-token"

    if auth_mode == "helper":
        if has_api_key or has_auth_token or has_oauth_token or has_cloud:
            print(
                "ERROR: helper auth mode was requested, but another Claude auth "
                "method is also set. Unset other Claude auth variables so the "
                "apiKeyHelper is used.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not has_settings:
            print(
                "ERROR: helper auth mode requires CVDP_CLAUDE_SETTINGS with a "
                "settings JSON string or path containing apiKeyHelper",
                file=sys.stderr,
            )
            sys.exit(1)
        return auth_mode, "helper"

    if auth_mode == "host":
        if has_api_key or has_auth_token or has_oauth_token or has_cloud:
            print(
                "ERROR: host auth mode was requested, but environment-token auth "
                "is also set. Unset Claude auth env vars for a host-auth run.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not host_auth_ready:
            print(
                "ERROR: host auth mode requires mounted Claude auth files under /host-claude",
                file=sys.stderr,
            )
            sys.exit(1)
        return auth_mode, "host"

    # Unreachable because auth_mode was already validated.
    raise AssertionError(auth_mode)


def _bare_mode_default(auth_path):
    return auth_path in {"api-key", "helper"}


def _sanitize_command_for_log(command):
    redacted = []
    skip_next = False
    for arg in command:
        if skip_next:
            redacted.append("<redacted>")
            skip_next = False
            continue
        redacted.append(arg)
        if arg == "--settings":
            skip_next = True
    return " ".join(redacted)


def _build_claude_command(env, auth_path):
    requested_bare = env.get("CVDP_CLAUDE_USE_BARE")
    use_bare = _parse_bool(requested_bare, _bare_mode_default(auth_path))

    if use_bare and auth_path not in {"api-key", "helper"}:
        print(
            "ERROR: bare mode only supports ANTHROPIC_API_KEY or apiKeyHelper auth",
            file=sys.stderr,
        )
        sys.exit(1)

    command = ["claude"]
    if use_bare:
        command.append("--bare")

    command.extend([
        "-p",
        "--dangerously-skip-permissions",
        "--no-session-persistence",
    ])

    if not use_bare:
        command.extend(["--setting-sources", "user"])

    if _has_env(env, "CVDP_CLAUDE_SETTINGS"):
        command.extend(["--settings", env["CVDP_CLAUDE_SETTINGS"]])

    if _has_env(env, "CVDP_CLAUDE_MODEL"):
        command.extend(["--model", env["CVDP_CLAUDE_MODEL"]])

    if _has_env(env, "CVDP_CLAUDE_MAX_BUDGET_USD"):
        command.extend(["--max-budget-usd", env["CVDP_CLAUDE_MAX_BUDGET_USD"]])

    if _has_env(env, "CVDP_CLAUDE_ALLOWED_TOOLS"):
        command.extend(["--allowedTools", env["CVDP_CLAUDE_ALLOWED_TOOLS"]])

    if _has_env(env, "CVDP_CLAUDE_DISALLOWED_TOOLS"):
        command.extend(["--disallowedTools", env["CVDP_CLAUDE_DISALLOWED_TOOLS"]])

    return command, use_bare


def _apply_benchmark_workspace_rules(task, env):
    include_rules = _parse_bool(env.get("CVDP_CLAUDE_INCLUDE_WORKSPACE_RULES"), True)
    if not include_rules:
        return task

    if task.lstrip().startswith("Benchmark workspace rules:"):
        return task

    return f"{BENCHMARK_WORKSPACE_RULES}\nTask:\n{task}"


def main():
    """Main agent function"""
    print("Starting Claude Code agent...")

    # Read the task from prompt.json
    try:
        with open("/code/prompt.json", "r") as f:
            prompt_data = json.load(f)
            task = prompt_data.get("prompt", "")
    except Exception as e:
        print(f"Error reading prompt.json: {e}")
        sys.exit(1)

    if not task:
        print("No task found in prompt.json. Exiting.")
        sys.exit(1)

    task = _apply_benchmark_workspace_rules(task, os.environ)

    # Check for max turns configuration
    max_turns = os.environ.get('CLAUDE_CODE_MAX_TURNS', None)
    if max_turns:
        print(f"Configuring Claude to use up to {max_turns} agent turns for fair comparison")
        # Prepend turn requirement instructions to the task
        turn_requirement = f"""IMPORTANT: For fair comparison with other agents, you may use up to {max_turns} agent turns to complete this task.

An "agent turn" is one response message from you that may contain multiple tool calls. Stop once the implementation is correct; do not spend extra turns only to use the available budget. When useful:

1. Take an iterative, exploratory approach
2. Break the work into many small steps
3. Read and analyze files thoroughly before making changes
4. Test and verify your changes incrementally
5. Use multiple turns to explore, plan, implement, test, and refine
6. Be thorough and methodical, but finish early when the task is complete

Your goal is to produce high-quality results within the turn budget.

Task:
"""
        task = turn_requirement + task

    print(f"Task: {task[:200]}..." if len(task) > 200 else f"Task: {task}")

    # Change to the code directory where files are mounted
    os.chdir("/code")

    # Setup environment and validate one supported auth path.
    env = os.environ.copy()
    requested_auth_mode = env.get("CVDP_CLAUDE_AUTH_MODE", "env").strip().lower()
    host_auth_ready = _prepare_claude_home(env, requested_auth_mode)
    auth_mode, auth_path = _resolve_auth_mode(env, host_auth_ready)
    print(f"Using Claude Code auth mode: {auth_mode}")

    if _parse_bool(env.get("CVDP_CLAUDE_ENABLE_TELEMETRY"), False):
        env['CLAUDE_CODE_ENABLE_TELEMETRY'] = '1'
        print("Claude Code telemetry is enabled by request")
    else:
        env.setdefault('CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC', '1')

    env.setdefault('CLAUDE_CODE_SKIP_PROMPT_HISTORY', '1')
    env.setdefault('CLAUDE_CODE_DISABLE_WEB_SEARCH', '1')
    env.setdefault('CLAUDE_CODE_DISABLE_WEB_FETCH', '1')
    print("Claude prompt history is disabled for this benchmark run")
    print("Claude web search and fetch tools are disabled when supported by the CLI")

    command, use_bare = _build_claude_command(env, auth_path)
    print(f"Claude bare mode: {'enabled' if use_bare else 'disabled'}")

    # Run Claude in non-interactive mode
    # Use stdin to pass prompt to avoid command-line length issues
    # --dangerously-skip-permissions = bypass all permission prompts
    print("\nInvoking Claude Code...")
    print("=" * 80)
    print(f"Command: echo '<task>' | {_sanitize_command_for_log(command)}")
    print(f"Task length: {len(task)} characters")
    print("Output will appear when task completes...")
    print("=" * 80)
    sys.stdout.flush()
    sys.stderr.flush()

    # Use subprocess to pipe the task via stdin
    # This avoids command-line argument length issues
    try:
        result = subprocess.run(
            command,
            input=task,
            text=True,
            env=env,
            cwd="/code",
            capture_output=False  # Let output go directly to stdout/stderr
        )
        sys.exit(result.returncode)
    except Exception as e:
        print(f"\nError running Claude: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
