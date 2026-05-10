#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

BUILD_COMMERCIAL="${BUILD_COMMERCIAL:-0}"

usage() {
    echo "Usage: $0 [--commercial]"
    echo ""
    echo "Builds claude-code-agent by default. Pass --commercial or set"
    echo "BUILD_COMMERCIAL=1 to also build claude-code-agent-commercial."
}

while [ $# -gt 0 ]; do
    case "$1" in
        --commercial)
            BUILD_COMMERCIAL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

echo "Building Claude Code agent Docker image..."
echo ""

# Build non-commercial version (no EDA tools)
echo "Building claude-code-agent (non-commercial)..."
docker build -t claude-code-agent .
echo "✓ claude-code-agent build complete"
echo ""

# Build commercial version (with EDA tools)
if [ "$BUILD_COMMERCIAL" = "1" ]; then
    echo "Building claude-code-agent-commercial (with EDA tool paths)..."
    docker build \
        -f Dockerfile.commercial \
        -t claude-code-agent-commercial .
    echo "✓ claude-code-agent-commercial build complete"
    echo ""
fi

echo "=========================================="
echo "Build complete! Docker image is ready:"
echo "  - claude-code-agent (non-commercial)"
if [ "$BUILD_COMMERCIAL" = "1" ]; then
    echo "  - claude-code-agent-commercial (with EDA tool paths)"
fi
echo "=========================================="
echo ""
echo "To run the agent with the benchmark:"
echo "  Non-commercial: ./run_benchmark.py -f <dataset.jsonl> -l -g claude-code-agent"
if [ "$BUILD_COMMERCIAL" = "1" ]; then
    echo "  Commercial:     ./run_benchmark.py -f <dataset.jsonl> -l -g claude-code-agent-commercial"
fi
