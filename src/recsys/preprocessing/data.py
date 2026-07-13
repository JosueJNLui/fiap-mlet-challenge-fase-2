"""Carga e preparação do dataset MovieLens (dtypes enxutos, filtro, reindexação)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_ratings(path: Path) -> pd.DataFrame:
    """Carrega rating.csv com dtypes enxutos e timestamp como datetime."""
    dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32"}
    df = pd.read_csv(path / "rating.csv", dtype=dtypes, parse_dates=["timestamp"])
    return df.rename(columns={"userId": "user", "movieId": "item"})


def filter_by_activity(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Mantém apenas usuários e itens com número mínimo de interações."""
    items_ok = df["item"].value_counts()
    df = df[df["item"].isin(items_ok[items_ok >= min_item].index)]
    users_ok = df["user"].value_counts()
    return df[df["user"].isin(users_ok[users_ok >= min_user].index)]


def sample_users(df: pd.DataFrame, n_users: int, seed: int) -> pd.DataFrame:
    """Amostra um subconjunto de usuários (n_users <= 0 mantém todos)."""
    if n_users <= 0 or df["user"].nunique() <= n_users:
        return df
    rng = np.random.default_rng(seed)
    chosen = rng.choice(df["user"].unique(), size=n_users, replace=False)
    return df[df["user"].isin(chosen)]


def reindex(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Mapeia user/item para índices contíguos 0..n-1; retorna df e cardinalidades."""
    df = df.copy()
    df["user_idx"] = df["user"].astype("category").cat.codes
    df["item_idx"] = df["item"].astype("category").cat.codes
    return df, int(df["user_idx"].nunique()), int(df["item_idx"].nunique())


def item_train_counts(train: pd.DataFrame, n_items: int) -> np.ndarray:
    """Contagem de interações por item no treino (base p/ popularidade e novidade)."""
    return np.bincount(train["item_idx"].to_numpy(), minlength=n_items)
