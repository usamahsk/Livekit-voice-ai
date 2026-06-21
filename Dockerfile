# syntax=docker/dockerfile:1

# Use the official UV Python base image with Python 3.13 on Debian Bookworm
ARG PYTHON_VERSION=3.13
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

# Keeps Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Compile Python source to bytecode (.pyc) during install
ENV UV_COMPILE_BYTECODE=1

# CRITICAL FIX: Explicitly pin the home and HuggingFace cache directories globally
# This forces model_q8.onnx to download directly into /app/.cache instead of /root/.cache
ENV HOME=/app
ENV HF_HOME=/app/.cache/huggingface

# --- Build stage ---
FROM base AS build

# Install build dependencies required for Python packages with native extensions
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /app

# Copy just the dependency files first for efficient layer caching
COPY pyproject.toml uv.lock ./
RUN mkdir -p src

# Install Python dependencies using UV's lock file
RUN uv sync --locked

# Copy all remaining application files into the container
COPY . .

# Pre-download ML models directly into the newly pinned /app/.cache directory
RUN uv run python -m livekit.agents download-files

# --- Production stage ---
FROM base

# Create a non-privileged user that the app will run under
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Copy the application AND the pre-downloaded models in /app/.cache with correct ownership
COPY --from=build --chown=appuser:appuser /app /app

WORKDIR /app

# Switch to the non-privileged user for security
USER appuser

# Run the AgentServer using UV
CMD ["uv", "run", "src/agent.py", "start"]