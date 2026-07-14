"""Testes da camada de config e persistência do pipeline."""

from __future__ import annotations

import contextlib

import mlflow
from mlflow import MlflowClient

from recsys.config import Settings, load_settings
from recsys.io import load_model, save_model
from recsys.models.baselines import GlobalMeanRecommender
from recsys.tracking import promote_to_production


def test_settings_loads_yaml_defaults() -> None:
    s = load_settings()
    assert s.data.n_users_sample == 20000
    assert s.data.min_user_ratings == 20
    assert s.models["bpr"]["emb_dim"] == 128
    assert s.mlflow.experiment_name == "MovieLens-Reco-Etapa2-Modelagem"


def test_env_overrides_yaml(monkeypatch) -> None:
    monkeypatch.setenv("DATA__N_USERS_SAMPLE", "5")
    assert Settings().data.n_users_sample == 5


def test_model_roundtrip(tmp_path) -> None:
    model = GlobalMeanRecommender()
    save_model(model, tmp_path, "gm")
    assert load_model(tmp_path, "gm").name == model.name


def _register_version(client: MlflowClient, name: str, ndcg: float) -> str:
    """Cria uma run com o NDCG dado e registra uma nova versão de ``name``."""
    with mlflow.start_run() as run:
        mlflow.log_metric("ndcg_at_10", ndcg)
    with contextlib.suppress(Exception):  # já existe
        client.create_registered_model(name)
    mv = client.create_model_version(name, source=run.info.artifact_uri, run_id=run.info.run_id)
    return mv.version


def test_promote_to_production(tmp_path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    client = MlflowClient()
    name = "MovieLens_BPR_Reco"
    _register_version(client, name, 0.10)  # v1
    _register_version(client, name, 0.20)  # v2 (melhor)

    promote_to_production("BPR", 0.20)
    assert int(client.get_model_version_by_alias(name, "production").version) == 2
    assert int(client.get_model_version_by_alias(name, "staging").version) == 2

    _register_version(client, name, 0.05)  # v3 (pior)
    promote_to_production("BPR", 0.05)
    assert int(client.get_model_version_by_alias(name, "production").version) == 2  # inalterado
    assert int(client.get_model_version_by_alias(name, "staging").version) == 3
