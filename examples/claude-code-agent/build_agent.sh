#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Default license server
CDS_LIC_FILE="${CDS_LIC_FILE:-5280@10.4.120.82}"
LM_LICENSE_FILE="${LM_LICENSE_FILE:-5280@10.4.120.82}"

echo "Building Claude Code agent Docker images..."
echo ""

# Build non-commercial version (no EDA tools)
echo "1. Building claude-code-agent (non-commercial)..."
docker build -t claude-code-agent .
echo "✓ claude-code-agent build complete"
echo ""

# Build commercial version (with EDA tools)
echo "2. Building claude-code-agent-commercial (with EDA tools)..."
echo "   Using license server: $CDS_LIC_FILE"
docker build \
    --build-arg CDS_LIC_FILE="$CDS_LIC_FILE" \
    --build-arg LM_LICENSE_FILE="$LM_LICENSE_FILE" \
    -f Dockerfile.commercial \
    -t claude-code-agent-commercial .
echo "✓ claude-code-agent-commercial build complete"
echo ""

echo "=========================================="
echo "Build complete! Docker images are ready:"
echo "  - claude-code-agent (non-commercial)"
echo "  - claude-code-agent-commercial (with EDA tools)"
echo "=========================================="
echo ""
echo "To run the agent with the benchmark:"
echo "  Non-commercial: ./run_benchmark.py -f <dataset.jsonl> -l -g claude-code-agent"
echo "  Commercial:     ./run_benchmark.py -f <dataset.jsonl> -l -g claude-code-agent-commercial"
echo ""
echo "To use a different license server:"
echo "  CDS_LIC_FILE='port@server' ./build_agent.sh"
