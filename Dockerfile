# ---- builder: resolve as deps em /app/.venv ----
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock ./
# --no-dev: apenas deps de prod (sem dvc/pytest/ruff). Os containers rodam os módulos
# do pipeline diretamente; dvc repro continua sendo responsabilidade do host (make repro).
# No Linux, torch resolve para o wheel CPU via [tool.uv.sources] no pyproject (sem CUDA/NVIDIA).
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# ---- runtime: slim, non-root, PYTHONPATH=src ----
FROM python:3.13-slim-bookworm AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src PYTHONUNBUFFERED=1
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY configs ./configs
# data/, models/ e mlruns precisam ser graváveis pelo usuário app; evitar um chown -R
# sobre toda a árvore do projeto, pois isso pode ficar extremamente lento quando a
# imagem contém um .venv grande.
RUN useradd --create-home --uid 1000 app \
 && install -d -o app -g app /app/data /app/models /app/mlruns
USER app
CMD ["python", "-m", "recsys.pipeline.train"]
