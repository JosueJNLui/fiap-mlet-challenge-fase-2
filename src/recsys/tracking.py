"""Integração com MLflow no DagsHub (tracking + registro de modelos).

O pipeline exige DagsHub: sem credenciais, ``init_mlflow`` falha alto (sem fallback
local). Autenticação é feita via variáveis ``MLFLOW_TRACKING_*`` para funcionar em
execuções headless (DVC/Docker), sem o fluxo interativo do pacote ``dagshub``.
"""

from __future__ import annotations

import os

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from recsys.config import Settings
from recsys.models.base import Recommender


class RecommenderPyfunc(mlflow.pyfunc.PythonModel):
    """Adaptador pyfunc: expõe ``Recommender.predict`` para o MLflow (todos os modelos)."""

    def __init__(self, model: Recommender) -> None:
        self.model = model

    def predict(self, context, model_input: pd.DataFrame, params=None):  # noqa: ARG002
        """Prevê scores a partir de um DataFrame com colunas ``user_idx``/``item_idx``."""
        return self.model.predict(
            model_input["user_idx"].to_numpy(),
            model_input["item_idx"].to_numpy(),
        )


def init_mlflow(settings: Settings) -> str:
    """Aponta o MLflow para o servidor DagsHub. Exige ``DAGSHUB_TOKEN`` no ``.env``."""
    if not settings.dagshub_token:
        raise RuntimeError(
            "DAGSHUB_TOKEN ausente: defina no .env. O pipeline exige tracking no DagsHub."
        )
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.dagshub_user or settings.dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.dagshub_token
    uri = f"https://dagshub.com/{settings.dagshub_repo_owner}/{settings.dagshub_repo_name}.mlflow"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)
    return uri


def log_recommender(model: Recommender, example: pd.DataFrame, registered_model_name: str) -> None:
    """Loga o modelo como pyfunc, com signature/input_example, e registra no Registry."""
    preds = model.predict(example["user_idx"].to_numpy(), example["item_idx"].to_numpy())
    signature = infer_signature(example, preds)
    mlflow.pyfunc.log_model(
        name="model",
        python_model=RecommenderPyfunc(model),
        signature=signature,
        input_example=example,
        registered_model_name=registered_model_name,
    )


def _alias_ndcg(client: MlflowClient, name: str, alias: str) -> float | None:
    """NDCG@10 da versão apontada por ``alias``, ou ``None`` se o alias não existe."""
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except Exception:  # noqa: BLE001 - alias/modelo inexistente
        return None
    return client.get_run(mv.run_id).data.metrics.get("ndcg_at_10")


def promote_to_production(label: str, ndcg: float) -> None:
    """Promove a última versão de ``MovieLens_<label>_Reco`` no Model Registry.

    Toda versão vira ``staging``; o alias ``production`` só migra para ela se o
    NDCG@10 informado bater o da produção atual (ou se ainda não houver produção).
    """
    client = MlflowClient()
    name = f"MovieLens_{label}_Reco"
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        return
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(name, "staging", latest.version)
    current = _alias_ndcg(client, name, "production")
    if current is None or ndcg > current:
        client.set_registered_model_alias(name, "production", latest.version)
