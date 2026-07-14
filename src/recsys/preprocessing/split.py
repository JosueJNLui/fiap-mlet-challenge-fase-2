"""Estratégias de split treino/teste (Strategy pattern).

As estratégias são intercambiáveis: ``TemporalLeaveLastFraction`` respeita a ordem
do tempo (padrão do projeto, evita vazar o futuro); ``RandomHoldout`` é a alternativa
sem ordenação temporal, útil como controle. Ambas garantem que todo usuário do teste
também aparece no treino (sem IDs desconhecidos nas embeddings).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SplitStrategy(ABC):
    """Estratégia de separação treino/teste por usuário."""

    @abstractmethod
    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Devolve (treino, teste) preservando a interface de colunas do df."""


class TemporalLeaveLastFraction(SplitStrategy):
    """Separa a fração temporal final (mais recente) de cada usuário para teste."""

    def __init__(self, test_frac: float = 0.2) -> None:
        self.test_frac = test_frac

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df.sort_values(["user_idx", "timestamp"])
        grp = df.groupby("user_idx")
        rank = grp.cumcount()
        sizes = grp["item_idx"].transform("size")
        n_test = np.maximum((sizes * self.test_frac).astype(int), 1)
        is_test = rank >= (sizes - n_test)
        return df[~is_test].copy(), df[is_test].copy()


class RandomHoldout(SplitStrategy):
    """Separa uma fração aleatória de cada usuário para teste (sem ordem temporal)."""

    def __init__(self, test_frac: float = 0.2, seed: int = 42) -> None:
        self.test_frac = test_frac
        self.seed = seed

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        shuffled = df.sample(frac=1.0, random_state=self.seed)
        grp = shuffled.groupby("user_idx")
        n_test = np.maximum((grp["item_idx"].transform("size") * self.test_frac).astype(int), 1)
        is_test = grp.cumcount() < n_test
        return shuffled[~is_test].copy(), shuffled[is_test].copy()
