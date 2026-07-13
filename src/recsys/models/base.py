"""Interface comum dos recomendadores (Template Method + Strategy).

``Recommender`` é a estratégia comum: todo modelo implementa os passos primitivos
``fit`` e ``predict``. ``evaluate`` é o Template Method — orquestra a mesma sequência
de avaliação (erro de nota + ranking/diversidade) para qualquer modelo, e ``recommend``
tem uma implementação concreta padrão reutilizável (top-k por score).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from recsys.evaluation.metrics import (
    ranking_metrics,
    regression_metrics,
    score_candidates,
)


class Recommender(ABC):
    """Estratégia comum de recomendação por predição de score."""

    name: str = "Recommender"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> Recommender:
        """Ajusta o modelo aos dados de treino (passo primitivo)."""

    @abstractmethod
    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Prevê o score de cada par (user, item) (passo primitivo)."""

    def recommend(
        self, user: int, k: int, seen: set[int], n_items: int, item_batch: int = 2048,
    ) -> np.ndarray:
        """Top-k itens não vistos por score decrescente (implementação padrão)."""
        candidates = np.array([i for i in range(n_items) if i not in seen], dtype=np.int64)
        if len(candidates) == 0:
            return candidates
        scores = score_candidates(self, user, candidates, item_batch)
        return candidates[np.argsort(-scores)[:k]]

    def evaluate(
        self, test: pd.DataFrame, *, seen_by_user: dict[int, set[int]], n_items: int,
        train_counts: np.ndarray, k: int = 10, like_threshold: float = 4.0,
        ranking_n_users: int = 500, item_batch: int = 2048, seed: int = 42,
    ) -> dict[str, float]:
        """Template Method: erro de nota + ranking/diversidade em catálogo completo."""
        y_pred = np.clip(
            self.predict(test["user_idx"].to_numpy(), test["item_idx"].to_numpy()),
            0.5, 5.0,
        )
        metrics = regression_metrics(test["rating"].to_numpy(), y_pred)
        metrics.update(
            ranking_metrics(
                self, test, seen_by_user, n_items, train_counts,
                k=k, like_threshold=like_threshold, n_users=ranking_n_users,
                item_batch=item_batch, seed=seed,
            )
        )
        return metrics
