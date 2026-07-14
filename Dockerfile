# ---- builder: resolve deps into /app/.venv ----
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock ./
# --no-dev: prod deps only (no dvc/pytest/ruff). Containers run the pipeline
# modules directly; dvc repro stays a host concern (make repro). torch resolves
# to the CPU wheel on Linux via [tool.uv.sources] in pyproject (no CUDA/NVIDIA).
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# ---- runtime: slim, non-root, PYTHONPATH=src ----
FROM python:3.13-slim-bookworm AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src PYTHONUNBUFFERED=1
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY configs ./configs
# data/ and models/ arrive as volumes; mlruns owned by app so the named MLflow
# volume inherits uid-1000 ownership (non-root writes the sqlite db).
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/data /app/models /app/mlruns \
 && chown -R app /app
USER app
CMD ["python", "-m", "recsys.pipeline.train"]
