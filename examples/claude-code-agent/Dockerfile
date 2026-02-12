FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x (required for Claude Code which needs >= 18)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Verify Node.js version
RUN node --version && npm --version

# Install Claude Code CLI via npm
RUN npm install -g @anthropic-ai/claude-code

# Verify Claude installation
RUN which claude && claude --version

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
