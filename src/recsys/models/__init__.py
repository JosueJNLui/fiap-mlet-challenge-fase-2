"""Modelos de recomendação: baselines, SVD, BPR neural e factory."""

from recsys.models.base import Recommender
from recsys.models.baselines import (
    BiasRecommender,
    GlobalMeanRecommender,
    PopularityRecommender,
)
from recsys.models.bpr import BPRRecommender
from recsys.models.factory import create_recommender
from recsys.models.svd import SVDRecommender

__all__ = [
    "BPRRecommender",
    "BiasRecommender",
    "GlobalMeanRecommender",
    "PopularityRecommender",
    "Recommender",
    "SVDRecommender",
    "create_recommender",
]
