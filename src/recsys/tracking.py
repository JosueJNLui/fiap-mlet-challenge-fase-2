"""Integração com MLflow no DagsHub (tracking + registro de modelos).

O pipeline exige DagsHub: sem credenciais, ``init_mlflow`` falha alto (sem fallback
local). Autenticação é feita via variáveis ``MLFLOW_TRACKING_*`` para funcionar em
execuções headless (DVC/Docker), sem o fluxo interativo do pacote ``dagshub``.
"""

from __future__ import annotations

import os

import mlflow
import pandas as pd
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
