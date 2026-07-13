"""Métricas de regressão, ranking e diversidade para recomendação.

As funções de ranking/diversidade compartilham uma única passada de scoring por
usuário (cara: rankeia todo o catálogo não visto), evitando pontuar duas vezes.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ScoreModel(Protocol):
    """Qualquer objeto que pontue pares (users, items) — basta um ``predict``."""

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray: ...


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, MSE e R² das notas previstas."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _dcg(relevances: np.ndarray) -> float:
    """Discounted Cumulative Gain de um vetor de relevâncias já ordenado."""
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(ranked_rel: np.ndarray, k: int) -> float:
    """NDCG@k a partir das relevâncias na ordem prevista."""
    ideal = np.sort(ranked_rel)[::-1]
    idcg = _dcg(ideal[:k])
    return _dcg(ranked_rel[:k]) / idcg if idcg > 0 else 0.0


def seen_items_by_user(df: pd.DataFrame) -> dict[int, set[int]]:
    """Conjunto de itens vistos por usuário (candidatos de ranking = não vistos)."""
    return df.groupby("user_idx")["item_idx"].agg(set).to_dict()


def score_candidates(
    model: ScoreModel, user: int, candidates: np.ndarray, item_batch: int,
) -> np.ndarray:
    """Pontua um usuário contra muitos candidatos, em lotes (memória controlada)."""
    scores = np.empty(len(candidates), dtype=np.float64)
    for start in range(0, len(candidates), item_batch):
        batch = candidates[start : start + item_batch]
        users = np.full(len(batch), user, dtype=np.int64)
        scores[start : start + len(batch)] = model.predict(users, batch)
    return scores


def _rank_one_user(
    model: ScoreModel, user: int, positives: list[int], seen: set[int],
    n_items: int, k: int, item_batch: int,
) -> tuple[float, float, float, np.ndarray]:
    """Rankeia todo o catálogo não visto; devolve P@k, R@k, NDCG@k e o top-k previsto."""
    candidates = np.array([i for i in range(n_items) if i not in seen], dtype=np.int64)
    if len(candidates) == 0:
        return 0.0, 0.0, 0.0, np.empty(0, dtype=np.int64)
    order = np.argsort(-score_candidates(model, user, candidates, item_batch))
    ranked = candidates[order]
    ranked_rel = np.isin(ranked, positives).astype(int)
    hits = ranked_rel[:k].sum()
    return hits / k, hits / len(positives), ndcg_at_k(ranked_rel, k), ranked[:k]


def _gini(counts: np.ndarray) -> float:
    """Coeficiente de Gini da distribuição de frequências (0=uniforme, 1=concentrado)."""
    if counts.sum() == 0:
        return 0.0
    sorted_c = np.sort(counts)
    n = len(sorted_c)
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * sorted_c)) / (n * sorted_c.sum()))


def _liked_users(test: pd.DataFrame, like_threshold: float) -> pd.Series:
    """Itens curtidos (rating >= limiar) por usuário no teste."""
    liked = test[test["rating"] >= like_threshold]
    return liked.groupby("user_idx")["item_idx"].agg(list)


def ranking_metrics(
    model: ScoreModel, test: pd.DataFrame, seen_by_user: dict[int, set[int]],
    n_items: int, train_counts: np.ndarray, *, k: int = 10,
    like_threshold: float = 4.0, n_users: int = 500, item_batch: int = 2048,
    seed: int = 42,
) -> dict[str, float]:
    """Precision/Recall/NDCG @k + cobertura, novidade e Gini sobre o catálogo completo."""
    liked = _liked_users(test, like_threshold)
    users = liked.index.to_numpy()
    if len(users) > n_users:
        users = np.random.default_rng(seed).choice(users, size=n_users, replace=False)

    rows, recommended = [], []
    for u in users:
        p, r, n, top_k = _rank_one_user(
            model, u, liked[u], seen_by_user.get(u, set()), n_items, k, item_batch,
        )
        rows.append((p, r, n))
        recommended.append(top_k)
    return _aggregate(rows, recommended, k, n_items, train_counts)


def _aggregate(
    rows: list[tuple], recommended: list[np.ndarray], k: int,
    n_items: int, train_counts: np.ndarray,
) -> dict[str, float]:
    """Consolida ranking (média) e diversidade (sobre o top-k agregado)."""
    p, r, n = np.mean(rows, axis=0)
    rec = np.concatenate(recommended) if recommended else np.empty(0, dtype=np.int64)
    rec_counts = np.bincount(rec, minlength=n_items)
    log_pop = np.log1p(train_counts[rec]) if len(rec) else np.array([0.0])
    return {
        f"precision_at_{k}": float(p),
        f"recall_at_{k}": float(r),
        f"ndcg_at_{k}": float(n),
        "coverage": float((rec_counts > 0).sum() / n_items),
        "novelty": float(np.mean(log_pop)),
        "gini": _gini(rec_counts.astype(float)),
    }
