# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build argument - set EXTRAS=local to include sentence-transformers
# (adds ~1 GB; only needed for local offline embeddings)
#
# Examples:
#   docker build .                          # BM25-only + OpenAI-compatible API
#   docker build --build-arg EXTRAS=local . # + sentence-transformers
# ---------------------------------------------------------------------------
ARG EXTRAS=""

FROM python:3.12-slim AS base

# System deps: gcc needed by some transitive build steps (e.g. tiktoken)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata first for better layer caching
COPY pyproject.toml ./
COPY README.md ./
COPY groundmemory/ ./groundmemory/

# Install the package (and optional extras if requested via build arg)
ARG EXTRAS
RUN if [ -n "$EXTRAS" ]; then \
        pip install --no-cache-dir ".[$EXTRAS]"; \
    else \
        pip install --no-cache-dir .; \
    fi

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------

# All workspace data is written here - mount a host directory at this path
ENV GROUNDMEMORY_ROOT_DIR=/data

# Default workspace name (override via GROUNDMEMORY_WORKSPACE env var)
ENV GROUNDMEMORY_WORKSPACE=default

# Default to BM25-only; override in .env to switch to openai-compatible API
ENV GROUNDMEMORY_EMBEDDING__PROVIDER=none

EXPOSE 4242

VOLUME ["/data"]

CMD ["groundmemory-mcp"]