"""Factory de recomendadores: mapeia um nome ao produto concreto."""

from __future__ import annotations

from recsys.models.base import Recommender
from recsys.models.baselines import (
    BiasRecommender,
    GlobalMeanRecommender,
    PopularityRecommender,
)
from recsys.models.bpr import BPRRecommender
from recsys.models.svd import SVDRecommender


def create_recommender(
    name: str, n_users: int, n_items: int, **params,
) -> Recommender:
    """Cria um recomendador pelo nome. Aceita: global_mean, bias, svd, popularity, bpr.

    Args:
        name: identificador do produto.
        n_users: número de usuários (cardinalidade contígua).
        n_items: número de itens (cardinalidade contígua).
        **params: hiperparâmetros específicos do produto.

    Returns:
        Instância concreta de ``Recommender``.

    Raises:
        ValueError: se ``name`` não corresponder a nenhum produto.
    """
    builders = {
        "global_mean": lambda: GlobalMeanRecommender(),
        "bias": lambda: BiasRecommender(n_users, n_items, **params),
        "svd": lambda: SVDRecommender(n_users, n_items, **params),
        "popularity": lambda: PopularityRecommender(n_items),
        "bpr": lambda: BPRRecommender(n_users, n_items, **params),
    }
    if name not in builders:
        raise ValueError(f"Recomendador desconhecido: {name!r}. Opções: {sorted(builders)}")
    return builders[name]()
