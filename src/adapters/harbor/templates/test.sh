#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
cd /sandbox/workspace/code
ROOT="/tests/src"
OUT="/tmp/test_out.txt"
ARTIFACTS="/logs/verifier/artifacts"
: > "$OUT"
mkdir -p "$ARTIFACTS"

save_artifacts() {
    local dir f rel

    for dir in rtl verif docs rundir; do
        if [ -d "$dir" ]; then
            find "$dir" -type f | while read -r f; do
                rel="${f#./}"
                mkdir -p "$ARTIFACTS/$(dirname "$rel")"
                cp "$f" "$ARTIFACTS/$rel"
            done
        fi
    done

    find . -maxdepth 1 -type f \( -name "*.v" -o -name "*.sv" -o -name "*.svh" -o -name "*.vh" -o -name "*.txt" -o -name "*.md" -o -name "*.json" \) | while read -r f; do
        rel="${f#./}"
        cp "$f" "$ARTIFACTS/$rel"
    done
}

save_artifacts
echo "Saved agent artifacts to $ARTIFACTS" >> "$OUT"

choose_runners() {
    if [ -f "$ROOT/test_runner.py" ]; then
        printf '%s\n' "$ROOT/test_runner.py"
        return
    fi
    find "$ROOT" -type f -name 'test_runner*.py' | sort
}

resolve_env_file() {
    local runner="$1"
    local dir base suffix
    dir="$(dirname "$runner")"
    base="$(basename "$runner" .py)"

    if [ "$base" != "test_runner" ]; then
        suffix="${base#test_runner_}"
        if [ -f "$dir/.env_${suffix}" ]; then
            printf '%s\n' "$dir/.env_${suffix}"
            return
        fi
    fi

    if [ -f "$dir/.env" ]; then
        printf '%s\n' "$dir/.env"
        return
    fi

    if [ -f "$ROOT/.env" ]; then
        printf '%s\n' "$ROOT/.env"
    fi
}

status=0
mapfile -t RUNNERS < <(choose_runners)

if [ "${#RUNNERS[@]}" -eq 0 ]; then
    echo "No test_runner*.py found under $ROOT" > "$OUT"
    status=1
else
    for runner in "${RUNNERS[@]}"; do
        env_file="$(resolve_env_file "$runner" || true)"
        {
            echo "=== RUNNER: $runner ==="
            if [ -n "$env_file" ]; then
                echo "=== ENV: $env_file ==="
            fi
        } >> "$OUT"

        if ! (
            set -a
            if [ -n "$env_file" ] && [ -f "$env_file" ]; then
                source "$env_file"
            fi
            set +a
            pytest -s --log-cli-level=INFO "$runner" -v
        ) >> "$OUT" 2>&1; then
            status=1
        fi
    done
fi

if [ "$status" -eq 0 ]; then
    echo '{"reward": 1, "accuracy": 1}' > /logs/verifier/reward.json
else
    echo '{"reward": 0, "accuracy": 0}' > /logs/verifier/reward.json
fi

cat "$OUT"
