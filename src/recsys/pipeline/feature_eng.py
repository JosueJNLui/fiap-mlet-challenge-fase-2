"""Stage 2: split temporal treino/teste (sem vazar o futuro).

ponytail: nó separado apenas para satisfazer os 4 stages exigidos pelo pipeline DVC
(Task 3). Sua única "feature" é o split temporal — dobraria em preprocess.py se o DVC
não pedisse a separação.
"""

from __future__ import annotations

import pandas as pd

from recsys.config import load_settings
from recsys.io import INTERACTIONS, TEST, TRAIN
from recsys.preprocessing.split import TemporalLeaveLastFraction


def main() -> None:
    """Escreve ``train.parquet`` e ``test.parquet`` a partir de ``interactions.parquet``."""
    s = load_settings()
    df = pd.read_parquet(s.paths.processed / INTERACTIONS)
    train, test = TemporalLeaveLastFraction(s.data.test_frac).split(df)

    train.to_parquet(s.paths.processed / TRAIN, index=False)
    test.to_parquet(s.paths.processed / TEST, index=False)
    print(f"feature_eng: treino={len(train)} teste={len(test)}")


if __name__ == "__main__":
    main()
