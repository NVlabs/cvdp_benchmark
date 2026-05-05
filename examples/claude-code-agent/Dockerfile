FROM nvidia/cvdp-sim:v1.0.0

ARG DEBIAN_FRONTEND=noninteractive

LABEL org.cvdp.agent.env="CLAUDE_CODE_MAX_TURNS,ANTHROPIC_API_KEY,ANTHROPIC_AUTH_TOKEN,CLAUDE_CODE_OAUTH_TOKEN,ANTHROPIC_BASE_URL,ANTHROPIC_CUSTOM_HEADERS,CVDP_CLAUDE_AUTH_MODE,CVDP_CLAUDE_SETTINGS,CVDP_CLAUDE_MODEL,CVDP_CLAUDE_MAX_BUDGET_USD,CVDP_CLAUDE_USE_BARE,CVDP_CLAUDE_ALLOWED_TOOLS,CVDP_CLAUDE_DISALLOWED_TOOLS,CVDP_CLAUDE_ENABLE_TELEMETRY,CVDP_CLAUDE_INCLUDE_WORKSPACE_RULES,CLAUDE_CODE_API_KEY_HELPER_TTL_MS,CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC,CLAUDE_CODE_SKIP_PROMPT_HISTORY,DISABLE_AUTOUPDATER"
LABEL org.cvdp.agent.mounts="env:CVDP_CLAUDE_AUTH_MODE=host:~/.claude/.credentials.json:/host-claude/.credentials.json:ro;env:CVDP_CLAUDE_AUTH_MODE=host:~/.claude/settings.json:/host-claude/settings.json:ro;env:CVDP_CLAUDE_AUTH_MODE=host:~/.claude.json:/host-claude/.claude.json:ro"

# Install Node.js from the Ubuntu 24.04 base image repositories. Claude Code
# requires Node >= 18, which the CVDP simulation image's base distribution provides.
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs \
    npm && \
    rm -rf /var/lib/apt/lists/*

# Verify Node.js version
RUN node --version && npm --version

# Install Claude Code CLI via npm.
RUN npm install -g @anthropic-ai/claude-code

# Verify Claude and the inherited OSS verification toolchain.
RUN command -v iverilog && \
    command -v verilator && \
    command -v yosys && \
    python3 -m pytest --version && \
    which claude && \
    claude --version

# Create a non-root user for Claude (required for --dangerously-skip-permissions)
RUN useradd -m -u 1001 -s /bin/bash claude && \
    mkdir -p /home/claude/.config/claude && \
    chown -R claude:claude /home/claude && \
    chmod 755 /home/claude && \
    chmod -R u+w /home/claude

ENV HOME=/home/claude

# Copy agent wrapper
WORKDIR /app
COPY agent.py .
RUN chmod +x agent.py && chown claude:claude agent.py

# Create /code directory and set permissions
RUN mkdir -p /code && chown -R claude:claude /code

# Set environment variable to indicate we're in a container
ENV CLAUDE_CODE_CONTAINER=1

# Switch to non-root user
USER claude

# Set the entrypoint
ENTRYPOINT ["python3", "/app/agent.py"]
