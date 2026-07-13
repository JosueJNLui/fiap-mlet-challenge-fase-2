"""Stage 4: agrega as métricas dos modelos em comparison.csv + metrics.json.

Consome as métricas que ``train`` já calculou (uma por modelo em models/), monta a
tabela de comparação, escolhe os melhores e loga uma run-resumo no MLflow. Não
re-avalia (o ranking do BPR é caro): a avaliação acontece uma vez, no stage de treino.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from recsys.config import load_settings
from recsys.io import read_json, write_json
from recsys.tracking import init_mlflow

NAME_TO_LABEL = {
    "global_mean": "GlobalMean",
    "bias": "Bias",
    "svd": "SVD",
    "popularity": "Popularity",
    "bpr": "BPR",
}
COMPARISON_CSV = Path("comparison.csv")
METRICS_JSON = Path("metrics.json")


def main() -> None:
    """Escreve comparison.csv + metrics.json e loga a run de comparação."""
    s = load_settings()
    rows = {
        label: read_json(s.paths.models / f"{name}.metrics.json")
        for name, label in NAME_TO_LABEL.items()
        if (s.paths.models / f"{name}.metrics.json").exists()
    }
    if not rows:
        raise FileNotFoundError("Nenhuma métrica em models/; rode `train` antes de `evaluate`.")

    comparison = pd.DataFrame(rows).T
    comparison.to_csv(COMPARISON_CSV)
    best = {
        "best_rmse_model": str(comparison["rmse"].idxmin()),
        "best_ndcg_model": str(comparison["ndcg_at_10"].idxmax()),
    }
    write_json(METRICS_JSON, {"models": rows, **best})

    init_mlflow(s)
    with mlflow.start_run(run_name="model_comparison_summary"):
        mlflow.log_artifact(str(COMPARISON_CSV))
        mlflow.set_tags(best)
    print(f"evaluate: melhor RMSE={best['best_rmse_model']} melhor NDCG={best['best_ndcg_model']}")


if __name__ == "__main__":
    main()
