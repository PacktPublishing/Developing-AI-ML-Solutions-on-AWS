FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:$PATH"

RUN apt-get update && apt-get install -y \
    curl ca-certificates git python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# ttyd serves the terminal in the browser over WebSocket; pick the binary for
# the build architecture (arm64 laptop locally, amd64 on Fargate).
RUN ARCH=$(uname -m) && \
    curl -fsSL "https://github.com/tsl0922/ttyd/releases/latest/download/ttyd.${ARCH}" \
    -o /usr/local/bin/ttyd && chmod +x /usr/local/bin/ttyd

# uv for the SQL helper's environment
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# Claude Code CLI
RUN curl -fsSL https://claude.ai/install.sh | bash && \
    cp /root/.local/bin/claude /usr/local/bin/claude && \
    chmod 755 /usr/local/bin/claude

# Restricted shell: the browser terminal cannot escape to bash with `!`,
# while Claude's own Bash tool is unaffected.
RUN printf '#!/bin/sh\necho "Shell access is not available."\nexit 1\n' \
      > /usr/local/bin/noshell && chmod +x /usr/local/bin/noshell

# Non-root: claude --dangerously-skip-permissions refuses to run as root
RUN useradd -m analyst

WORKDIR /app
COPY CLAUDE.md pyproject.toml launch-claude docker-entrypoint.sh ./
COPY skills/ skills/
COPY config/ config/
# One pre-built environment for the SQL helper; CLAUDE.md tells the model to
# reuse it with `uv run`, never to create new venvs.
RUN uv sync --no-dev && \
    cp launch-claude /usr/local/bin/launch-claude && \
    chmod +x /usr/local/bin/launch-claude docker-entrypoint.sh && \
    mkdir -p /home/analyst/state && chown -R analyst:analyst /app /home/analyst

USER analyst
EXPOSE 7681
ENTRYPOINT ["./docker-entrypoint.sh"]
