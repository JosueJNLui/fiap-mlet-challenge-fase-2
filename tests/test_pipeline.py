"""Testes da camada de config e persistência do pipeline."""

from __future__ import annotations

from recsys.config import Settings, load_settings
from recsys.io import load_model, save_model
from recsys.models.baselines import GlobalMeanRecommender


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
