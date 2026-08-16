# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────────────────────
# AI Hedge Finance v2 — Production Image
#
# Build:
#   docker build -t ai-hedge-finance:latest .
#
# Run (trade):
#   docker run --env-file .env ai-hedge-finance:latest ahf-trade
#
# Run (train):
#   docker run --env-file .env ai-hedge-finance:latest ahf-train
# ──────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.11

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies (rl extra required for model inference)
RUN uv sync --frozen --no-dev --extra rl --no-install-project

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/

# Install project
RUN uv sync --frozen --no-dev --extra rl

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

WORKDIR /app

# Create non-root user
RUN addgroup --system ahf && adduser --system --ingroup ahf ahf

# Copy installed env from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs

# Create runtime data directories
RUN mkdir -p data/logs data/models && chown -R ahf:ahf /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER ahf

# Health check: import the package
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "from ahf.core.settings import AHFSettings; AHFSettings()" || exit 1

# Default entrypoint: trade (override in docker-compose or k8s)
ENTRYPOINT ["python", "-m", "ahf.entrypoints.trade"]
CMD []
