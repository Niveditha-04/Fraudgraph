"""
Per-model threshold analysis for Phase 3 (base Elliptic).

The originally-reported precision/recall for GraphSAGE and GAT are both at
the default 0.5 classification threshold. AUC-PR itself is threshold-free
and stands as reported, but a precision/recall pair at an untuned threshold
is not a fair basis for comparing two models -- GAT's very different
precision/recall balance next to GraphSAGE's could be an artifact of
threshold choice, not a genuine difference in what each model learned.

This retrains both models (same seed, same config as models/run_phase3.py,
so the default-threshold numbers reproduce exactly) to get the raw
per-node probabilities Phase 3 didn't persist, picks the F1-maximizing
threshold on the VALIDATION set only, and reports precision/recall at that
threshold on the test set -- never tuning the threshold against the same
data the final numbers are reported on.
"""
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from models.data_utils import prepare_data
from models.gnn_models import GAT, GraphSAGE
from models.metrics import compute_metrics
from models.train_gnn import train_gnn

SEED = 42


def best_f1_threshold(y_val: np.ndarray, val_probs: np.ndarray) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_val, (val_probs >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


def evaluate_at_threshold(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = (probs >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def run_model(name: str, model: torch.nn.Module, pdata) -> dict:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    val_metrics, test_metrics, _, raw = train_gnn(model, pdata, epochs=300, patience=30, verbose=False)
    print(f"reproduced default-threshold test metrics: {test_metrics}")
    assert abs(test_metrics.precision - {"GraphSAGE": 0.6725, "GAT": 0.1149}[name]) < 0.01, (
        f"{name}: retrained default-threshold precision doesn't match the committed Phase 3 result -- "
        "seed/config drift, investigate before trusting the threshold analysis"
    )

    threshold = best_f1_threshold(raw["val_y"], raw["val_probs"])
    at_default = evaluate_at_threshold(raw["test_y"], raw["test_probs"], 0.5)
    at_f1max = evaluate_at_threshold(raw["test_y"], raw["test_probs"], threshold)

    print(f"F1-max threshold (selected on val set): {threshold:.2f}")
    print(f"  at 0.5:              precision={at_default['precision']:.4f} recall={at_default['recall']:.4f} f1={at_default['f1']:.4f}")
    print(f"  at {threshold:.2f} (F1-max): precision={at_f1max['precision']:.4f} recall={at_f1max['recall']:.4f} f1={at_f1max['f1']:.4f}")

    return {
        "auc_pr": test_metrics.auc_pr,
        "default_threshold": at_default,
        "f1_max_threshold": at_f1max,
    }


def main():
    torch.manual_seed(SEED)
    pdata = prepare_data()
    in_dim = pdata.x.shape[1]

    torch.manual_seed(SEED)
    sage_results = run_model("GraphSAGE", GraphSAGE(in_dim), pdata)

    torch.manual_seed(SEED)
    gat_results = run_model("GAT", GAT(in_dim), pdata)

    results = {"graphsage": sage_results, "gat": gat_results}
    with open("models/results/threshold_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved to models/results/threshold_analysis.json")
    return results


if __name__ == "__main__":
    main()
