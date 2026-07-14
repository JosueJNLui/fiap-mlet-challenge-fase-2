"""App FastAPI: expõe o modelo final com /health, /recommend e /docs (Swagger nativo)."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request

from recsys.api.serving import ServingState, load_serving, recommend_for_user
from recsys.config import Settings, load_settings

logger = logging.getLogger(__name__)


def _utc_now_z() -> str:
    """Timestamp UTC ISO-8601 com sufixo ``Z`` (ex.: ``2026-07-14T12:00:00.000000Z``)."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Cria o app; carrega modelo + artefatos no startup (readiness via /health)."""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.serving = load_serving(settings)
            app.state.ready = True
            logger.info("serving pronto: %d itens", app.state.serving.n_items)
        except Exception as exc:  # noqa: BLE001 - pipeline não rodou / artefato ausente
            app.state.serving = None
            app.state.ready = False
            logger.error("falha ao carregar serving; /health retornará 503: %s", exc)
        yield

    app = FastAPI(title="RecSys API", lifespan=lifespan)

    @app.middleware("http")
    async def add_headers(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = uuid.uuid4().hex
        response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.6f}"
        return response

    @app.get("/health")
    async def health(request: Request):
        if not request.app.state.ready:
            raise HTTPException(status_code=503, detail={"status": "loading"})
        return {"status": "ok", "timestamp": _utc_now_z()}

    @app.get("/recommend")
    async def recommend(request: Request, user_id: int):
        state: ServingState | None = request.app.state.serving
        if state is None:
            raise HTTPException(status_code=503, detail="modelo ainda não carregado")
        try:
            recs = recommend_for_user(state, user_id, k=10)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"user_id {user_id} desconhecido") from exc
        return {"user_id": user_id, "recommendations": recs}

    return app


app = create_app()
