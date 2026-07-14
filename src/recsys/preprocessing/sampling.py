"""Estratégias de amostragem de negativos (Strategy pattern).

O treino BPR precisa de itens "negativos" (não interagidos) por positivo. A estratégia
é intercambiável; ``UniformNegativeSampler`` amostra uniformemente do catálogo inteiro
e filtra os itens já vistos pelo usuário — a mesma distribuição de candidatos contra a
qual o modelo é avaliado no ranking em catálogo completo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class NegativeSampler(ABC):
    """Amostra itens negativos para um lote de usuários positivos."""

    @abstractmethod
    def sample(self, users: np.ndarray, n_neg: int) -> np.ndarray:
        """Devolve matriz [len(users), n_neg] de índices de itens negativos."""


class UniformNegativeSampler(NegativeSampler):
    """Amostra uniforme do catálogo, filtrando itens já vistos por usuário."""

    def __init__(
        self, n_items: int, seen_by_user: dict[int, set[int]], seed: int = 42,
    ) -> None:
        self.n_items = n_items
        self.seen_by_user = seen_by_user
        self.rng = np.random.default_rng(seed)

    def sample(self, users: np.ndarray, n_neg: int) -> np.ndarray:
        out = np.empty((len(users), n_neg), dtype=np.int64)
        for row, user in enumerate(users):
            seen = self.seen_by_user.get(int(user), ())
            out[row] = self._sample_unseen(seen, n_neg)
        return out

    def _sample_unseen(self, seen, n_neg: int) -> np.ndarray:
        """Rejeição simples: sorteia com folga e descarta itens vistos."""
        picked: list[int] = []
        while len(picked) < n_neg:
            draw = self.rng.integers(0, self.n_items, size=(n_neg - len(picked)) * 4)
            picked.extend(int(i) for i in draw if i not in seen)
        return np.array(picked[:n_neg], dtype=np.int64)
