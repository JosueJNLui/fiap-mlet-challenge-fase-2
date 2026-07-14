"""Stage 3: treina os 5 modelos (Factory), avalia, loga no MLflow e salva em models/."""

from __future__ import annotations

import re
from pathlib import Path

import mlflow
import pandas as pd

from recsys.api.serving import build_serving_artifact
from recsys.config import Settings, load_settings
from recsys.evaluation.metrics import seen_items_by_user
from recsys.io import META, TEST, TRAIN, read_json, save_model, write_json
from recsys.models.base import Recommender
from recsys.models.factory import create_recommender
from recsys.preprocessing.data import item_train_counts
from recsys.tracking import init_mlflow, log_recommender

# (nome na Factory, kwargs-builder, rótulo p/ run/registro).
Spec = tuple[str, dict, str]


def _specs(s: Settings) -> list[Spec]:
    """Configuração dos 5 modelos; hiperparâmetros vêm do config.yaml."""
    return [
        ("global_mean", {}, "GlobalMean"),
        ("bias", dict(s.models.get("bias", {})), "Bias"),
        ("svd", dict(s.models.get("svd", {})), "SVD"),
        ("popularity", {}, "Popularity"),
        ("bpr", {"params": dict(s.models.get("bpr", {})), "seed": s.seed}, "BPR"),
    ]


def _dataset_version(raw_dvc: Path) -> str:
    """Extrai o hash DVC de data/raw.dvc para rastrear a versão do dataset."""
    if not raw_dvc.exists():
        return "unknown"
    match = re.search(r"md5:\s*([0-9a-f]+\.dir)", raw_dvc.read_text())
    return match.group(1) if match else "unknown"


def _flat_params(kwargs: dict, n_users: int, n_items: int, seed: int) -> dict:
    """Achata os kwargs do modelo (incl. o dict aninhado do BPR) para log de params."""
    flat = {"n_users": n_users, "n_items": n_items, "seed": seed}
    for key, value in kwargs.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def _train_one(
    spec: Spec, train: pd.DataFrame, test: pd.DataFrame, meta: dict, s: Settings,
) -> dict:
    """Treina, avalia, loga a run MLflow e persiste o modelo + suas métricas."""
    name, kwargs, label = spec
    n_users, n_items = meta["n_users"], meta["n_items"]
    model: Recommender = create_recommender(name, n_users, n_items, **kwargs)
    model.fit(train)
    metrics = model.evaluate(
        test, seen_by_user=seen_items_by_user(train), n_items=n_items,
        train_counts=item_train_counts(train, n_items), like_threshold=s.eval.like_threshold,
        ranking_n_users=s.eval.ranking_n_users, item_batch=s.eval.ranking_item_batch, seed=s.seed,
    )
    save_model(model, s.paths.models, name)
    write_json(s.paths.models / f"{name}.metrics.json", metrics)
    with mlflow.start_run(run_name=label):
        mlflow.log_params(_flat_params(kwargs, n_users, n_items, s.seed))
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "etapa": "2", "stage": "modeling", "model": label, "seed": s.seed,
            "dataset_version_dvc": _dataset_version(s.paths.raw.parent / "raw.dvc"),
            "n_users": n_users, "n_items": n_items, "ranking_protocol": "full_catalog",
        })
        log_recommender(model, test[["user_idx", "item_idx"]].head(5), f"MovieLens_{label}_Reco")
    print(f"train: {label} — ndcg@10={metrics.get('ndcg_at_10', float('nan')):.4f}")
    return metrics


def main() -> None:
    """Treina todos os modelos e registra uma run MLflow por modelo."""
    s = load_settings()
    init_mlflow(s)
    train = pd.read_parquet(s.paths.processed / TRAIN)
    test = pd.read_parquet(s.paths.processed / TEST)
    meta = read_json(s.paths.processed / META)
    for spec in _specs(s):
        _train_one(spec, train, test, meta, s)
    build_serving_artifact(s)  # artefato self-contained p/ a API (Task 6)


if __name__ == "__main__":
    main()
