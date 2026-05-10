FROM nvidia/cvdp-sim:v1.0.0

ARG DEBIAN_FRONTEND=noninteractive

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
