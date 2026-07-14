"""Modelo neural de ranking: BPR Matrix Factorization (PyTorch).

Score ``p_u·q_i`` (BPR-MF puro, **sem termos de viés**) treinado com a loss pairwise BPR
sobre pares (positivo, negativo). Vieses foram removidos de propósito: na loss pairwise
``μ`` e ``b_u`` se cancelam (mesmo usuário nos dois lados) e ``b_i`` acaba absorvendo a
popularidade global e dominando o score — o modelo colapsa para "recomende o popular".
Sem eles, a ordenação depende só da personalização ``p_u·q_i``.

Os negativos vêm do ``UniformNegativeSampler`` (catálogo inteiro, filtrando itens vistos)
— a mesma distribuição de candidatos usada na avaliação em catálogo completo. O early
stopping seleciona o checkpoint pelo **mesmo NDCG@10 de catálogo completo que é reportado**
(medido num subconjunto de validação, por custo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from recsys.evaluation.metrics import ndcg_at_k, score_candidates, seen_items_by_user
from recsys.models.base import Recommender
from recsys.preprocessing.sampling import UniformNegativeSampler
from recsys.preprocessing.split import TemporalLeaveLastFraction


def resolve_device(prefer: str | None = None) -> str:
    """Escolhe o dispositivo (mps/cuda/cpu), respeitando ``prefer`` se disponível."""
    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class BPRMatrixFactorization(nn.Module):
    """Fatoração de matriz para ranking: score = <p_u, q_i> (BPR-MF puro, sem vieses)."""

    def __init__(self, n_users: int, n_items: int, emb_dim: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(users) * self.item_emb(items)).sum(-1)


DEFAULT_PARAMS: dict = {
    "emb_dim": 64,
    "lr": 5e-3,
    "weight_decay": 1e-5,
    "n_neg": 5,
    "batch_size": 2048,
    "epochs": 30,
    "patience": 5,
    "val_users": 200,
    "val_item_batch": 4096,
    "like_threshold": 4.0,
}


class BPRRecommender(Recommender):
    """Modelo neural de embeddings treinado com BPR, otimizado para NDCG@10."""

    name = "BPR"

    def __init__(
        self, n_users: int, n_items: int, *, params: dict | None = None,
        device: str | None = None, seed: int = 42,
    ) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.device = resolve_device(device)
        self.seed = seed
        self.history: dict = {}

    def fit(self, train: pd.DataFrame) -> BPRRecommender:
        torch.manual_seed(self.seed)
        tr, val = TemporalLeaveLastFraction(0.1).split(train)
        self.model = BPRMatrixFactorization(
            self.n_users, self.n_items, self.params["emb_dim"],
        ).to(self.device)
        loader = self._positive_loader(tr)
        sampler = UniformNegativeSampler(self.n_items, seen_items_by_user(tr), self.seed)
        self.history = self._train_loop(loader, sampler, val, seen_items_by_user(tr))
        return self

    def _positive_loader(self, tr: pd.DataFrame) -> DataLoader:
        """Loader dos pares positivos (rating >= limiar) usados no BPR."""
        pos = tr[tr["rating"] >= self.params["like_threshold"]]
        ds = TensorDataset(
            torch.tensor(pos["user_idx"].to_numpy(), dtype=torch.long),
            torch.tensor(pos["item_idx"].to_numpy(), dtype=torch.long),
        )
        return DataLoader(ds, batch_size=self.params["batch_size"], shuffle=True)

    def _train_loop(self, loader, sampler, val, seen_by_user) -> dict:
        """Treina com BPR e faz early stopping pelo NDCG@10 de catálogo completo."""
        opt = self._optimizer()
        history = {"train_loss": [], "val_ndcg": []}
        best_ndcg, best_state, since = -1.0, None, 0
        for epoch in range(1, self.params["epochs"] + 1):
            loss = self._train_epoch(loader, sampler, opt)
            ndcg = self._val_ndcg(val, seen_by_user)
            history["train_loss"].append(loss)
            history["val_ndcg"].append(ndcg)
            print(f"  epoca {epoch:02d} | BPR loss {loss:.4f} | val NDCG@10 {ndcg:.4f}")
            if ndcg > best_ndcg:
                best_ndcg, best_state, since = ndcg, self._snapshot(), 0
            else:
                since += 1
                if since >= self.params["patience"]:
                    print(f"  early stopping na epoca {epoch}")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        history["best_val_ndcg"] = best_ndcg
        return history

    def _optimizer(self) -> torch.optim.Optimizer:
        """Adam com L2 nos embeddings (regularização padrão do BPR-MF, sem vieses)."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.params["lr"], weight_decay=self.params["weight_decay"],
        )

    def _train_epoch(self, loader, sampler, opt) -> float:
        """Uma época de BPR: -logsigmoid(score_pos - score_neg) por negativo amostrado."""
        self.model.train()
        n_neg = self.params["n_neg"]
        total, seen = 0.0, 0
        for users, pos_items in loader:
            neg = torch.tensor(sampler.sample(users.numpy(), n_neg), dtype=torch.long)
            loss = self._bpr_loss(users, pos_items, neg, n_neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(users)
            seen += len(users)
        return total / seen

    def _bpr_loss(self, users, pos_items, neg, n_neg) -> torch.Tensor:
        """Loss BPR média sobre (positivo, n_neg negativos) do lote."""
        users = users.to(self.device)
        pos_score = self.model(users, pos_items.to(self.device))
        rep_users = users.unsqueeze(1).expand(-1, n_neg).reshape(-1)
        neg_score = self.model(rep_users, neg.to(self.device).reshape(-1)).view(-1, n_neg)
        diff = pos_score.unsqueeze(1) - neg_score
        return -torch.nn.functional.logsigmoid(diff).mean()

    @torch.no_grad()
    def _val_ndcg(self, val: pd.DataFrame, seen_by_user) -> float:
        """NDCG@10 em catálogo completo sobre um subconjunto de usuários de validação."""
        self.model.eval()
        liked = val[val["rating"] >= self.params["like_threshold"]]
        liked = liked.groupby("user_idx")["item_idx"].agg(list)
        if len(liked) == 0:
            return 0.0
        users = liked.index.to_numpy()
        rng = np.random.default_rng(self.seed)
        if len(users) > self.params["val_users"]:
            users = rng.choice(users, size=self.params["val_users"], replace=False)
        return float(np.mean([self._user_ndcg(u, liked[u], seen_by_user) for u in users]))

    def _user_ndcg(self, user, positives, seen_by_user) -> float:
        """NDCG@10 de um usuário rankeando todo o catálogo não visto."""
        seen = seen_by_user.get(int(user), set())
        candidates = np.array([i for i in range(self.n_items) if i not in seen], dtype=np.int64)
        if len(candidates) == 0:
            return 0.0
        scores = score_candidates(self, user, candidates, self.params["val_item_batch"])
        ranked_rel = np.isin(candidates[np.argsort(-scores)], positives).astype(int)
        return ndcg_at_k(ranked_rel, 10)

    @torch.no_grad()
    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        self.model.eval()
        u = torch.tensor(np.asarray(users), dtype=torch.long, device=self.device)
        i = torch.tensor(np.asarray(items), dtype=torch.long, device=self.device)
        return self.model(u, i).cpu().numpy()

    def _snapshot(self) -> dict:
        """Cópia dos pesos em CPU (checkpoint do melhor NDCG)."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
