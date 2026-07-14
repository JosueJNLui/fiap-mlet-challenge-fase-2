"""Persistência de artefatos do pipeline (modelos via pickle, JSON, caminhos)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from recsys.models.base import Recommender

INTERACTIONS = "interactions.parquet"
META = "meta.json"
TRAIN = "train.parquet"
TEST = "test.parquet"


def save_model(model: Recommender, models_dir: Path, name: str) -> Path:
    """Serializa um recomendador para ``<models_dir>/<name>.pkl``."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.pkl"
    with path.open("wb") as fh:
        pickle.dump(model, fh)
    return path


def load_model(models_dir: Path, name: str) -> Recommender:
    """Carrega o recomendador serializado em ``<models_dir>/<name>.pkl``."""
    with (models_dir / f"{name}.pkl").open("rb") as fh:
        return pickle.load(fh)


def write_json(path: Path, obj: object) -> None:
    """Escreve ``obj`` como JSON indentado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def read_json(path: Path) -> dict:
    """Lê um JSON como dict."""
    return json.loads(Path(path).read_text())
