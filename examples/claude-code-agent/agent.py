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

def main():
    """Main agent function"""
    print("Starting Claude Code agent...")

    # Copy Claude auth files from mounted read-only location to writable location
    # This allows Claude to update its config files during execution
    import shutil
    import subprocess as sp
    home = os.path.expanduser("~")

    # Don't pre-create .claude.json - let Claude CLI create it on first use
    # Just ensure the home directory is writable
    claude_json_path = f"{home}/.claude.json"
    if not os.path.exists(claude_json_path):
        print(f"Note: {claude_json_path} will be created by Claude CLI on first use")

    if os.path.exists("/host-claude/.claude.json"):
        print("Copying Claude auth files to writable location...")
        try:
            # Use cp command to avoid python permission issues
            sp.run(["cp", "/host-claude/.claude.json", f"{home}/.claude.json"], check=True)
            sp.run(["chmod", "644", f"{home}/.claude.json"], check=True)
            print("Copied .claude.json")
        except Exception as e:
            print(f"Warning: Could not copy .claude.json: {e}")

    if os.path.exists("/host-claude/.claude"):
        try:
            # Use cp command to avoid python permission issues
            if os.path.exists(f"{home}/.claude"):
                sp.run(["rm", "-rf", f"{home}/.claude"], check=True)
            sp.run(["cp", "-r", "/host-claude/.claude", f"{home}/.claude"], check=True)
            sp.run(["chmod", "-R", "755", f"{home}/.claude"], check=True)
            print("Copied .claude directory")
        except Exception as e:
            print(f"Warning: Could not copy .claude directory: {e}")

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

    # Check for max turns configuration
    max_turns = os.environ.get('CLAUDE_CODE_MAX_TURNS', None)
    if max_turns:
        print(f"Configuring Claude to use {max_turns} agent turns for fair comparison")
        # Prepend turn requirement instructions to the task
        turn_requirement = f"""IMPORTANT: For fair comparison with other agents, you should aim to use approximately {max_turns} agent turns to complete this task.

An "agent turn" is one response message from you that may contain multiple tool calls. Do NOT rush to complete the task. Instead:

1. Take an iterative, exploratory approach
2. Break the work into many small steps
3. Read and analyze files thoroughly before making changes
4. Test and verify your changes incrementally
5. Use multiple turns to explore, plan, implement, test, and refine
6. Be thorough and methodical rather than trying to solve everything at once

Your goal is to produce high-quality results while using the full turn budget available to you.

Task:
"""
        task = turn_requirement + task

    print(f"Task: {task[:200]}..." if len(task) > 200 else f"Task: {task}")

    # Change to the code directory where files are mounted
    os.chdir("/code")

    # Setup environment - ensure we have ANTHROPIC_API_KEY
    env = os.environ.copy()

    # Check for ANTHROPIC_API_KEY in environment
    if 'ANTHROPIC_API_KEY' not in env:
        print("ERROR: ANTHROPIC_API_KEY not found in environment!")
        print("Available env vars:", [k for k in env.keys() if 'KEY' in k or 'API' in k])
        sys.exit(1)

    # Validate it's an actual Anthropic key
    api_key = env['ANTHROPIC_API_KEY']
    if not api_key.startswith('sk-ant-'):
        print(f"WARNING: ANTHROPIC_API_KEY does not start with 'sk-ant-' (starts with '{api_key[:10]}...')")
        print("This may not be a valid Anthropic API key")
        sys.exit(1)

    print(f"Using Anthropic API key: {api_key[:20]}...")

    # Use /tmp for Claude config to avoid home directory permission issues
    env['CLAUDE_CONFIG_DIR'] = '/tmp/.claude'

    # Enable telemetry for debugging
    env['CLAUDE_CODE_ENABLE_TELEMETRY'] = '1'
    env['OTEL_LOG_USER_PROMPTS'] = '1'
    # Log to console for now (we can add proper OTLP later)
    env['OTEL_LOGS_EXPORTER'] = 'console'
    env['OTEL_METRICS_EXPORTER'] = 'console'

    # Disable web access tools to prevent looking up solutions online
    # This ensures the agent can only work with local files
    env['CLAUDE_CODE_DISABLE_WEB_SEARCH'] = '1'
    env['CLAUDE_CODE_DISABLE_WEB_FETCH'] = '1'
    print("Web search and fetch tools are DISABLED - agent can only use local files")

    # Run Claude in non-interactive mode
    # Use stdin to pass prompt to avoid command-line length issues
    # --dangerously-skip-permissions = bypass all permission prompts
    print("\nInvoking Claude Code...")
    print("=" * 80)
    print(f"Command: echo '<task>' | claude -p --dangerously-skip-permissions")
    print(f"Task length: {len(task)} characters")
    print("Output will appear when task completes...")
    print("=" * 80)
    sys.stdout.flush()
    sys.stderr.flush()

    # Use subprocess to pipe the task via stdin
    # This avoids command-line argument length issues
    try:
        result = subprocess.run(
            ["claude", "-p", "--dangerously-skip-permissions"],
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
