"""Stage 1: carrega o dataset cru, filtra por atividade, amostra e reindexa."""

from __future__ import annotations

from recsys.config import load_settings
from recsys.io import INTERACTIONS, META, write_json
from recsys.preprocessing.data import (
    filter_by_activity,
    load_ratings,
    reindex,
    sample_users,
)


def main() -> None:
    """Escreve ``interactions.parquet`` (com user/item crus + índices) e ``meta.json``."""
    s = load_settings()
    df = load_ratings(s.paths.raw)
    df = filter_by_activity(df, s.data.min_user_ratings, s.data.min_item_ratings)
    df = sample_users(df, s.data.n_users_sample, s.seed)
    df, n_users, n_items = reindex(df)

    s.paths.processed.mkdir(parents=True, exist_ok=True)
    df.to_parquet(s.paths.processed / INTERACTIONS, index=False)
    meta = {"n_users": n_users, "n_items": n_items, "n_rows": len(df)}
    write_json(s.paths.processed / META, meta)
    print(f"preprocess: {len(df)} interações, {n_users} usuários, {n_items} itens")


if __name__ == "__main__":
    main()
