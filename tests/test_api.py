"""Testes da API FastAPI (serving do modelo)."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from recsys.api.app import create_app
from recsys.api.serving import build_serving_artifact
from recsys.config import Paths, Settings
from recsys.io import INTERACTIONS, save_model
from recsys.models.baselines import PopularityRecommender
from recsys.preprocessing.data import reindex

N_USERS, N_ITEMS = 6, 20


def _toy_interactions() -> pd.DataFrame:
    """Interações densas via ``reindex`` (mesma contiguidade do pipeline); ids crus deslocados.

    Item ``i`` é atribuído ao usuário ``i % N_USERS`` — garante que todos os itens aparecem
    (item_idx contíguo) e que cada usuário vê ~3-4 itens (>=10 não vistos p/ recomendar).
    """
    rows = [
        (100 + (i % N_USERS), 1000 + i, float(1 + i % 5), i)
        for i in range(N_ITEMS)
    ]
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "t"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="s")
    df, _, _ = reindex(df.drop(columns="t"))
    return df


def _app(tmp_path) -> tuple[object, pd.DataFrame]:
    """Monta artefatos num tmp dir e devolve o app FastAPI + as interações usadas."""
    df = _toy_interactions()
    df.to_parquet(tmp_path / INTERACTIONS, index=False)
    save_model(PopularityRecommender(N_ITEMS).fit(df), tmp_path, "bpr")
    # dagshub_token=None força o fallback local (init > .env), sem tocar a rede.
    settings = Settings(paths=Paths(processed=tmp_path, models=tmp_path), dagshub_token=None)
    build_serving_artifact(settings)
    return create_app(settings), df


def test_health_ok_with_headers(tmp_path) -> None:
    app, _ = _app(tmp_path)
    with TestClient(app) as client:  # o context manager dispara o lifespan (carrega o modelo)
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["timestamp"].endswith("Z")
    assert "X-Request-ID" in resp.headers
    assert "X-Process-Time" in resp.headers


def test_recommend_known_user(tmp_path) -> None:
    app, df = _app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/recommend", params={"user_id": 100})
    assert resp.status_code == 200
    recs = resp.json()["recommendations"]
    assert len(recs) == 10
    seen = set(df[df["user"] == 100]["item"])
    assert all(r["item_id"] not in seen for r in recs)


def test_recommend_unknown_user_404(tmp_path) -> None:
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/recommend", params={"user_id": 999999}).status_code == 404
