"""Modelo neural de ranking: BPR-v2 (NeuMF híbrido + negativos ponderados).

Melhorias sobre o bpr.py original:

1.  **Pop-sampling de negativos** — ``PopularityWeightedNegativeSampler`` com
    ``alpha`` ajustável, quebrando o viés de popularidade que limitava o BPR-MF
    a ~0.09 NDCG@10. Itens populares aparecem como negativos com mais frequência,
    forçando o modelo a aprender preferências genuínas.

2.  **Arquitetura NeuMF híbrida (GMF + MLP):**   
    - **GMF (Generalized MF):** dot product ``p_u·q_i`` — interações lineares.  
    - **MLP:** embeddings separados concatenados → ReLU → Dropout → camadas
      densas com crescente[s] — interações não lineares.  
    - **Fusão:** concatenação das duas saídas → camada densa final → score.
    Inspirado no NeuMF (He et al. 2017, "Neural Collaborative Filtering").

3.  **Regularização melhorada:** Dropout nos embeddings MLP e nas camadas
    densas, weight decay mais agressivo e gradient clipping.

4.  **LR scheduler:** CosineAnnealingWarmRestarts para escapar de platôs
    locais e melhorar convergência.

5.  **Predição calibrada:** `predict` aplica `mu + (pred - mu_pred) * (sigma_ratings / sigma_pred)`
    para alinhar a escala de scores BPR com ratings reais, melhorando o RMSE
    sem afetar a ordenação do ranking.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from recsys.evaluation.metrics import ndcg_at_k, score_candidates, seen_items_by_user
from recsys.models.base import Recommender
from recsys.preprocessing.sampling import PopularityWeightedNegativeSampler
from recsys.preprocessing.split import TemporalLeaveLastFraction


def resolve_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class HybridNMF(nn.Module):
    """NeuMF-style hybrid: GMF (dot-product) + MLP towers → fused score."""

    def __init__(
        self, n_users: int, n_items: int,
        mf_emb_dim: int = 64,
        mlp_emb_dim: int = 64,
        mlp_layers: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # GMF branch
        self.mf_user_emb = nn.Embedding(n_users, mf_emb_dim)
        self.mf_item_emb = nn.Embedding(n_items, mf_emb_dim)

        # MLP branch (separate embeddings)
        self.mlp_user_emb = nn.Embedding(n_users, mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(n_items, mlp_emb_dim)

        mlp_layers = mlp_layers or [128, 64, 32]
        input_dim = mlp_emb_dim * 2
        mlp_modules: list[nn.Module] = []
        for h in mlp_layers:
            mlp_modules.append(nn.Linear(input_dim, h))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            input_dim = h
        self.mlp = nn.Sequential(*mlp_modules)

        # Fusion layer: concat(GMF_out, MLP_out) -> score
        self.fusion = nn.Linear(mf_emb_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in [self.mf_user_emb, self.mf_item_emb,
                     self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.05)
        for mod in self.mlp:
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_uniform_(mod.weight, nonlinearity="relu")
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        nn.init.kaiming_uniform_(self.fusion.weight, nonlinearity="linear")
        nn.init.zeros_(self.fusion.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        # GMF: element-wise product
        mf_u = self.mf_user_emb(users)
        mf_i = self.mf_item_emb(items)
        gmf_out = mf_u * mf_i

        # MLP: concat + layers
        mlp_u = self.mlp_user_emb(users)
        mlp_i = self.mlp_item_emb(items)
        mlp_out = self.mlp(torch.cat([mlp_u, mlp_i], dim=-1))

        # Fusion
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.fusion(fused).squeeze(-1)


DEFAULT_PARAMS: dict = {
    # Architecture
    "mf_emb_dim": 64,          # GMF branch embedding dim
    "mlp_emb_dim": 64,         # MLP branch embedding dim
    "mlp_layers": [128, 64],   # MLP hidden layer sizes
    "dropout": 0.2,            # Dropout rate in MLP layers

    # Training
    "lr": 3e-3,
    "weight_decay": 1e-4,
    "n_neg": 10,
    "batch_size": 2048,
    "epochs": 60,
    "patience": 8,
    "grad_clip_norm": 1.0,
    "scheduler_t0": 10,    # CosineAnnealingWarmRestarts T_0

    # Negatives
    "pop_alpha": 0.75,      # Popularity weighting exponent (0 = uniform)

    # Validation
    "val_users": 300,
    "val_item_batch": 4096,
    "like_threshold": 4.0,
}


class BPRv2Recommender(Recommender):
    """BPR-v2: NeuMF híbrido + pop-sampling, com predição calibrada via z-score."""

    name = "BPR-v2"

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
        # calibration values set during fit
        self.mu, self.rating_std, self.score_std = 0.0, 1.0, 1.0

    def fit(self, train: pd.DataFrame) -> BPRv2Recommender:
        torch.manual_seed(self.seed)
        tr, val = TemporalLeaveLastFraction(0.1).split(train)

        # Compute item counts for pop-weighted sampling
        item_counts = np.bincount(
            tr["item_idx"].to_numpy(), minlength=self.n_items,
        ).astype(np.float64)

        self.mu = float(train["rating"].mean())

        self.model = HybridNMF(
            self.n_users, self.n_items,
            mf_emb_dim=self.params["mf_emb_dim"],
            mlp_emb_dim=self.params["mlp_emb_dim"],
            mlp_layers=self.params["mlp_layers"],
            dropout=self.params["dropout"],
        ).to(self.device)

        loader = self._positive_loader(tr)
        sampler = PopularityWeightedNegativeSampler(
            self.n_items, seen_items_by_user(tr),
            item_counts, alpha=self.params["pop_alpha"], seed=self.seed,
        )
        self.history = self._train_loop(loader, sampler, val, seen_items_by_user(tr))

        # Calibrate: compute score_std on train positives for z-score scaling
        self._calibrate_scale(tr)
        return self

    def _calibrate_scale(self, tr: pd.DataFrame) -> None:
        """Estimate ratings and predicted score std for z-score calibration."""
        self.rating_std = float(tr["rating"].std())
        # sample predictions to estimate score scale
        sample_n = min(50000, len(tr))
        idx_sample = np.random.default_rng(self.seed).choice(len(tr), sample_n, replace=False)
        s = tr.iloc[idx_sample]
        preds = self._raw_predict(
            s["user_idx"].to_numpy(), s["item_idx"].to_numpy(),
        )
        self.score_std = max(float(np.std(preds)), 1e-8)

    def _positive_loader(self, tr: pd.DataFrame) -> DataLoader:
        pos = tr[tr["rating"] >= self.params["like_threshold"]]
        ds = TensorDataset(
            torch.tensor(pos["user_idx"].to_numpy(), dtype=torch.long),
            torch.tensor(pos["item_idx"].to_numpy(), dtype=torch.long),
        )
        return DataLoader(ds, batch_size=self.params["batch_size"], shuffle=True)

    def _train_loop(self, loader, sampler, val, seen_by_user) -> dict:
        opt = self._optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=self.params["scheduler_t0"], T_mult=2,
        )
        history = {"train_loss": [], "val_ndcg": []}
        best_ndcg, best_state, since = -1.0, None, 0

        for epoch in range(1, self.params["epochs"] + 1):
            loss = self._train_epoch(loader, sampler, opt)
            scheduler.step()
            ndcg = self._val_ndcg(val, seen_by_user)
            history["train_loss"].append(loss)
            history["val_ndcg"].append(ndcg)
            print(f"  epoca {epoch:02d} | BPR loss {loss:.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | val NDCG@10 {ndcg:.4f}")
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
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params["lr"], weight_decay=self.params["weight_decay"],
        )

    def _train_epoch(self, loader, sampler, opt) -> float:
        self.model.train()
        n_neg = self.params["n_neg"]
        total, seen = 0.0, 0
        for users, pos_items in loader:
            neg = torch.tensor(sampler.sample(users.numpy(), n_neg), dtype=torch.long)
            loss = self._bpr_loss(users, pos_items, neg, n_neg)
            opt.zero_grad()
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.params["grad_clip_norm"])
            opt.step()
            total += loss.item() * len(users)
            seen += len(users)
        return total / seen

    def _bpr_loss(self, users, pos_items, neg, n_neg) -> torch.Tensor:
        users = users.to(self.device)
        pos_score = self.model(users, pos_items.to(self.device))
        rep_users = users.unsqueeze(1).expand(-1, n_neg).reshape(-1)
        neg_score = self.model(rep_users, neg.to(self.device).reshape(-1)).view(-1, n_neg)
        diff = pos_score.unsqueeze(1) - neg_score
        return -torch.nn.functional.logsigmoid(diff).mean()

    @torch.no_grad()
    def _val_ndcg(self, val: pd.DataFrame, seen_by_user) -> float:
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
        seen = seen_by_user.get(int(user), set())
        candidates = np.array([i for i in range(self.n_items) if i not in seen], dtype=np.int64)
        if len(candidates) == 0:
            return 0.0
        scores = score_candidates(self, user, candidates, self.params["val_item_batch"])
        ranked_rel = np.isin(candidates[np.argsort(-scores)], positives).astype(int)
        return ndcg_at_k(ranked_rel, 10)

    # --- Raw prediction (BPR score) ---
    @torch.no_grad()
    def _raw_predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        self.model.eval()
        u = torch.tensor(np.asarray(users), dtype=torch.long, device=self.device)
        i = torch.tensor(np.asarray(items), dtype=torch.long, device=self.device)
        return self.model(u, i).cpu().numpy()

    # --- Calibrated prediction (z-score aligned to ratings scale) ---
    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        raw = self._raw_predict(users, items)
        # z-score alignment: map raw scores to rating-scale scores
        calibrated = self.mu + (raw - raw.mean()) * (self.rating_std / self.score_std)
        return calibrated

    def _snapshot(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}