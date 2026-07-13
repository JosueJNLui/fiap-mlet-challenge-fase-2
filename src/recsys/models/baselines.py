"""Baselines: média global, vieses regularizados e popularidade."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.models.base import Recommender


class GlobalMeanRecommender(Recommender):
    """Baseline ingênuo: prevê a nota média global do treino."""

    name = "GlobalMean"

    def fit(self, train: pd.DataFrame) -> GlobalMeanRecommender:
        self.mu = float(train["rating"].mean())
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.full(len(users), self.mu, dtype=float)


class BiasRecommender(Recommender):
    """Baseline de vieses regularizados: mu + b_user + b_item."""

    name = "BiasBaseline"

    def __init__(self, n_users: int, n_items: int, reg: float = 10.0) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.reg = reg

    def _bias(self, df: pd.DataFrame, key: str, n: int, baseline: np.ndarray) -> np.ndarray:
        """Viés regularizado por grupo: soma(resíduo) / (reg + contagem)."""
        resid = df["rating"].to_numpy() - self.mu - baseline
        grp = pd.Series(resid).groupby(df[key].to_numpy())
        bias = grp.sum() / (self.reg + grp.size())
        out = np.zeros(n, dtype=float)
        out[bias.index.to_numpy()] = bias.to_numpy()
        return out

    def fit(self, train: pd.DataFrame) -> BiasRecommender:
        self.mu = float(train["rating"].mean())
        self.b_item = self._bias(train, "item_idx", self.n_items, 0.0)
        self.b_user = self._bias(
            train, "user_idx", self.n_users, self.b_item[train["item_idx"]],
        )
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return self.mu + self.b_user[users] + self.b_item[items]


class PopularityRecommender(Recommender):
    """Baseline de ranking: pontua itens pela contagem de interações no treino.

    Referência top-K padrão. Não é um preditor de nota — o score é ``log1p`` da
    popularidade (preserva a ordem), então suas métricas de regressão são apenas
    ilustrativas; o valor está no ranking.
    """

    name = "Popularity"

    def __init__(self, n_items: int) -> None:
        self.n_items = n_items

    def fit(self, train: pd.DataFrame) -> PopularityRecommender:
        counts = np.bincount(train["item_idx"].to_numpy(), minlength=self.n_items)
        self.item_score = np.log1p(counts.astype(float))
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return self.item_score[items]
