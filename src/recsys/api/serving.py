"""Camada de serving: artefato self-contained + carga do modelo para a API.

A API depende só de ``models/`` (não do dataset de treino): o pipeline persiste um
``serving.pkl`` com os mapeamentos id-cru↔índice e os itens já vistos por usuário. O modelo
vem do Model Registry (alias ``production``) quando há credenciais, com fallback para o
pickle local ``models/bpr.pkl``.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import Settings
from recsys.evaluation.metrics import seen_items_by_user
from recsys.io import INTERACTIONS, load_model
from recsys.models.base import Recommender

logger = logging.getLogger(__name__)

SERVING = "serving.pkl"


def build_serving_artifact(settings: Settings) -> Path:
    """Deriva mapas + seen do ``interactions.parquet`` e faz pickle em ``models/serving.pkl``."""
    df = pd.read_parquet(settings.paths.processed / INTERACTIONS)
    items = df[["item_idx", "item"]].drop_duplicates().sort_values("item_idx")
    artifact = {
        "user_to_idx": {int(u): int(i) for u, i in zip(df["user"], df["user_idx"], strict=True)},
        "item_ids": [int(x) for x in items["item"]],
        "seen_by_user": {int(u): {int(i) for i in s} for u, s in seen_items_by_user(df).items()},
        "n_items": int(df["item_idx"].nunique()),
    }
    settings.paths.models.mkdir(parents=True, exist_ok=True)
    path = settings.paths.models / SERVING
    with path.open("wb") as fh:
        pickle.dump(artifact, fh)
    return path


class _RegistryRecommender(Recommender):
    """Adapta o pyfunc do Registry à interface ``Recommender`` (herda ``recommend``)."""

    name = "BPR"

    def __init__(self, pyfunc, n_items: int) -> None:
        self._pyfunc = pyfunc
        self.n_items = n_items

    def fit(self, train: pd.DataFrame) -> _RegistryRecommender:  # noqa: ARG002 - serving-only
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        # int32: a signature foi inferida das colunas do parquet; o MLflow recusa int64.
        frame = pd.DataFrame(
            {"user_idx": np.asarray(users, dtype="int32"), "item_idx": np.asarray(items, "int32")}
        )
        return np.asarray(self._pyfunc.predict(frame))


def _force_cpu(model: Recommender) -> None:
    """Reseta o device de modelos torch (o ``device`` mps/cuda é picklado no treino)."""
    # só o BPR carrega um nn.Module; guardado por hasattr.
    if hasattr(model, "model") and hasattr(model, "device"):
        model.device = "cpu"
        model.model.to("cpu")


def load_model_prod(settings: Settings, n_items: int) -> Recommender:
    """Carrega o modelo do Registry (alias ``production``); fallback para o pickle local."""
    try:
        import mlflow

        from recsys.tracking import init_mlflow

        init_mlflow(settings)
        pyfunc = mlflow.pyfunc.load_model(f"models:/{settings.served_model}@production")
        logger.info("modelo carregado do Registry (%s@production)", settings.served_model)
        return _RegistryRecommender(pyfunc, n_items)
    except Exception as exc:  # noqa: BLE001 - sem creds/rede/alias → fallback local
        # lower() casa com o nome do pickle para BPR/SVD/Bias/Popularity (não p/ GlobalMean,
        # que nunca é servido); se isso mudar, use o mapa nome↔rótulo de evaluate.py.
        local = settings.served_label.lower()
        logger.warning("Registry indisponível (%s); usando pickle local models/%s.pkl", exc, local)
        model = load_model(settings.paths.models, local)
        _force_cpu(model)
        return model


def _fetch_serving_artifact(settings: Settings) -> Path:
    """Baixa ``serving.pkl`` do run da versão ``production`` (API sem artefatos locais)."""
    import mlflow
    from mlflow import MlflowClient

    from recsys.tracking import init_mlflow

    init_mlflow(settings)
    version = MlflowClient().get_model_version_by_alias(settings.served_model, "production")
    logger.info("baixando %s do run %s", SERVING, version.run_id)
    return Path(mlflow.artifacts.download_artifacts(run_id=version.run_id, artifact_path=SERVING))


@dataclass
class ServingState:
    """Tudo que a API precisa em memória para responder recomendações."""

    model: Recommender
    user_to_idx: dict[int, int]
    item_ids: list[int]
    seen_by_user: dict[int, set[int]]
    n_items: int


def load_serving(settings: Settings) -> ServingState:
    """Carrega o artefato de serving + o modelo (Registry/local) num estado em memória."""
    path = settings.paths.models / SERVING
    if not path.exists():
        path = _fetch_serving_artifact(settings)
    with path.open("rb") as fh:
        art = pickle.load(fh)
    model = load_model_prod(settings, art["n_items"])
    return ServingState(
        model=model,
        user_to_idx=art["user_to_idx"],
        item_ids=art["item_ids"],
        seen_by_user=art["seen_by_user"],
        n_items=art["n_items"],
    )


def recommend_for_user(state: ServingState, user_id: int, k: int = 10) -> list[dict]:
    """Top-k itens (item_id cru + score) para ``user_id``; ``KeyError`` se desconhecido."""
    if user_id not in state.user_to_idx:
        raise KeyError(user_id)
    user_idx = state.user_to_idx[user_id]
    seen = state.seen_by_user.get(user_idx, set())
    idx = state.model.recommend(user_idx, k, seen, state.n_items)
    scores = state.model.predict(np.full(len(idx), user_idx), idx)
    return [
        {"item_id": state.item_ids[int(i)], "score": float(s)}
        for i, s in zip(idx, scores, strict=True)
    ]
