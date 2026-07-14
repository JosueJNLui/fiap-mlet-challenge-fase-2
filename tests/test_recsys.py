"""Checks mínimos que falham se a lógica central quebrar (não é suíte completa)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.evaluation.metrics import ndcg_at_k, seen_items_by_user
from recsys.models import (
    BiasRecommender,
    BPRRecommender,
    GlobalMeanRecommender,
    PopularityRecommender,
    SVDRecommender,
    create_recommender,
)
from recsys.preprocessing import (
    RandomHoldout,
    TemporalLeaveLastFraction,
    UniformNegativeSampler,
)


def _toy_ratings(n_users: int = 30, n_items: int = 15, seed: int = 0) -> pd.DataFrame:
    """DataFrame sintético denso o suficiente para treinar/avaliar rapidamente."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        items = rng.choice(n_items, size=8, replace=False)
        for t, i in enumerate(items):
            rows.append((u, int(i), float(rng.integers(1, 6)), t))
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "t"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="s")
    df["user_idx"] = df["user"]
    df["item_idx"] = df["item"]
    return df


def test_factory_returns_correct_types():
    assert isinstance(create_recommender("global_mean", 5, 5), GlobalMeanRecommender)
    assert isinstance(create_recommender("bias", 5, 5), BiasRecommender)
    assert isinstance(create_recommender("svd", 5, 5), SVDRecommender)
    assert isinstance(create_recommender("popularity", 5, 5), PopularityRecommender)
    assert isinstance(create_recommender("bpr", 5, 5), BPRRecommender)


def test_factory_unknown_name_raises():
    with pytest.raises(ValueError):
        create_recommender("nope", 5, 5)


def test_temporal_split_has_no_future_leak():
    df = _toy_ratings()
    train, test = TemporalLeaveLastFraction(0.25).split(df)
    for user in df["user_idx"].unique():
        tr = train[train["user_idx"] == user]["timestamp"]
        te = test[test["user_idx"] == user]["timestamp"]
        if len(tr) and len(te):
            assert tr.max() <= te.min()


def test_split_covers_all_test_users_in_train():
    df = _toy_ratings()
    for strategy in (TemporalLeaveLastFraction(0.25), RandomHoldout(0.25, seed=1)):
        train, test = strategy.split(df)
        assert set(test["user_idx"]).issubset(set(train["user_idx"]))


def test_negatives_never_include_seen_items():
    df = _toy_ratings()
    seen = seen_items_by_user(df)
    sampler = UniformNegativeSampler(n_items=15, seen_by_user=seen, seed=3)
    users = df["user_idx"].unique()
    negs = sampler.sample(users, n_neg=4)
    for row, user in enumerate(users):
        assert not (set(negs[row]) & seen[int(user)])


def test_ndcg_at_k_hand_example():
    assert ndcg_at_k(np.array([1, 0, 1]), k=3) == pytest.approx(0.9197207891)


def test_sampler_is_reproducible():
    seen = {0: {1, 2}}
    a = UniformNegativeSampler(10, seen, seed=7).sample(np.array([0]), 3)
    b = UniformNegativeSampler(10, seen, seed=7).sample(np.array([0]), 3)
    assert np.array_equal(a, b)


def test_evaluate_template_reports_ranking_and_diversity():
    df = _toy_ratings()
    train, test = TemporalLeaveLastFraction(0.25).split(df)
    counts = np.bincount(train["item_idx"].to_numpy(), minlength=15)
    model = PopularityRecommender(15).fit(train)
    metrics = model.evaluate(
        test, seen_by_user=seen_items_by_user(train), n_items=15,
        train_counts=counts, ranking_n_users=30, like_threshold=4.0,
    )
    for key in ("rmse", "ndcg_at_10", "coverage", "novelty", "gini"):
        assert key in metrics


def test_bpr_trains_and_predicts_reproducibly():
    df = _toy_ratings()
    params = {"epochs": 2, "emb_dim": 8, "batch_size": 64, "val_users": 10}
    preds = []
    for _ in range(2):
        model = BPRRecommender(30, 15, params=params, device="cpu", seed=11).fit(df)
        preds.append(model.predict(np.array([0, 1, 2]), np.array([0, 1, 2])))
    assert preds[0].shape == (3,)
    assert np.allclose(preds[0], preds[1])
