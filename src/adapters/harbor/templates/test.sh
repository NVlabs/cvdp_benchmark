#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
ROOT="/tests/src"
CODE_ROOT="/sandbox/workspace/code"
RUNDIR="$CODE_ROOT/rundir"
PLAN="/tests/.harbor_verifier_plan"
OUT="/tmp/test_out.txt"
ARTIFACTS="/logs/verifier/artifacts"
: > "$OUT"

mkdir -p "$ROOT" "$RUNDIR" "$ARTIFACTS"

save_artifacts() {
    local dir f rel

    for dir in rtl verif docs rundir; do
        if [ -d "$CODE_ROOT/$dir" ]; then
            find "$CODE_ROOT/$dir" -type f | while read -r f; do
                rel="${f#"$CODE_ROOT"/}"
                mkdir -p "$ARTIFACTS/$(dirname "$rel")"
                cp "$f" "$ARTIFACTS/$rel"
            done
        fi
    done

    find "$CODE_ROOT" -maxdepth 1 -type f \( -name "*.v" -o -name "*.sv" -o -name "*.svh" -o -name "*.vh" -o -name "*.txt" -o -name "*.md" -o -name "*.json" \) | while read -r f; do
        rel="${f#"$CODE_ROOT"/}"
        cp "$f" "$ARTIFACTS/$rel"
    done
}

setup_compat_paths() {
    if [ ! -e /code ] || [ -L /code ]; then
        ln -sfn "$CODE_ROOT" /code
    fi
    if [ ! -e /src ] || [ -L /src ]; then
        ln -sfn "$ROOT" /src
    fi
    if [ ! -e /rundir ] || [ -L /rundir ]; then
        ln -sfn "$RUNDIR" /rundir
    fi
}

setup_compat_paths

# Save the agent's generated workspace before the verifier runs (so we capture
# its state, not the simulator outputs this script is about to produce).
save_artifacts
echo "Saved agent artifacts to $ARTIFACTS" >> "$OUT"

run_command() {
    local env_file="$1"
    local workdir="$2"
    local command="$3"

    {
        echo "=== COMMAND: $command ==="
        echo "=== WORKDIR: $workdir ==="
        if [ -n "$env_file" ]; then
            echo "=== ENV: $env_file ==="
        fi
    } >> "$OUT"

    (
        set -a
        if [ -n "$env_file" ] && [ -f "$env_file" ]; then
            source "$env_file"
        fi
        set +a
        mkdir -p "$workdir"
        cd "$workdir"
        bash -lc "$command"
    ) >> "$OUT" 2>&1
}

# Replay the verifier plan emitted by the converter. Each record is
# "env_file<TAB>workdir<TAB>base64(command)"; the command is base64-encoded so
# multi-line / quoted commands survive as a single line.
run_plan() {
    local status=0
    local line env_file workdir command_b64 rest command

    # Split on TAB by hand: `read` with IFS=$'\t' would treat the tab as
    # whitespace and silently drop an empty leading env_file.
    while IFS= read -r line || [ -n "$line" ]; do
        if [ -z "$line" ]; then
            continue
        fi
        env_file="${line%%$'\t'*}"
        rest="${line#*$'\t'}"
        workdir="${rest%%$'\t'*}"
        command_b64="${rest#*$'\t'}"
        if [ -z "$command_b64" ]; then
            continue
        fi
        command="$(printf '%s' "$command_b64" | base64 -d)"
        if ! run_command "$env_file" "$workdir" "$command"; then
            status=1
        fi
    done < "$PLAN"

    return "$status"
}

status=0

if [ -f "$PLAN" ]; then
    if ! run_plan; then
        status=1
    fi
else
    # Every CVDP task is converted with a plan; a missing one means the
    # conversion is broken, so fail loudly rather than silently pass.
    echo "No verifier plan found at $PLAN" >> "$OUT"
    status=1
fi

if [ "$status" -eq 0 ]; then
    echo '{"reward": 1, "accuracy": 1}' > /logs/verifier/reward.json
else
    echo '{"reward": 0, "accuracy": 0}' > /logs/verifier/reward.json
fi

cat "$OUT"
