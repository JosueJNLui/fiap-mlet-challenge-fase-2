"""Fatoração latente via TruncatedSVD (scikit-learn) sobre resíduos centrados."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from recsys.models.base import Recommender


class SVDRecommender(Recommender):
    """Fatoração latente (TruncatedSVD) sobre resíduos centrados na média."""

    name = "SVD"

    def __init__(
        self, n_users: int, n_items: int, n_components: int = 50, seed: int = 42,
    ) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.n_components = n_components
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> SVDRecommender:
        self.mu = float(train["rating"].mean())
        matrix = csr_matrix(
            (train["rating"].to_numpy() - self.mu,
             (train["user_idx"].to_numpy(), train["item_idx"].to_numpy())),
            shape=(self.n_users, self.n_items),
        )
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.seed)
        self.user_factors = svd.fit_transform(matrix)
        self.item_factors = svd.components_.T
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        dot = np.sum(self.user_factors[users] * self.item_factors[items], axis=1)
        return self.mu + dot
