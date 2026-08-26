"""
Evaluates the ORIGINAL hybrid score (0.7*GNN + 0.3*Benford) against ground
truth, using precision/recall/AUC-PR -- not just its two components in
isolation. This script is what proved that weighting underperforms the GNN
alone (see the results printed below), which is why models/hybrid_config.py
now sets HYBRID_WEIGHT_BENFORD=0. The 0.7/0.3 constants here are deliberately
hardcoded, not imported from hybrid_config -- this script's job is to
document what the rejected weighting did, not to track whatever the current
production weighting is.

Uses the per-node scores already cached by models/run_phase4.py
(models/results/phase4_node_scores.npz) -- no retraining, no new LLM calls,
just analysis of data that already exists on disk.
"""
import json

import numpy as np

from models.metrics import compute_metrics

SCORES_PATH = "models/results/phase4_node_scores.npz"
HYBRID_WEIGHT_GNN = 0.7  # the original, rejected weighting -- see module docstring
HYBRID_WEIGHT_BENFORD = 0.3


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    finite = x[np.isfinite(x)]
    lo, hi = finite.min(), finite.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def main():
    scores = np.load(SCORES_PATH)
    gnn_probs = scores["gnn_probs"]
    benford_scores = scores["benford_scores"]
    y = scores["y"]
    test_mask = scores["test_mask"]

    # same subsetting Phase 4 used: test nodes with both a finite Benford
    # score and a labeled (non-"unknown") ground truth
    test_idx = np.where(test_mask)[0]
    valid = test_idx[np.isfinite(benford_scores[test_idx]) & (y[test_idx] != 2)]

    gnn_test = gnn_probs[valid]
    benford_test = benford_scores[valid]
    y_test = y[valid]

    gnn_norm = min_max_normalize(gnn_test)
    benford_norm = min_max_normalize(benford_test)
    hybrid = HYBRID_WEIGHT_GNN * gnn_norm + HYBRID_WEIGHT_BENFORD * benford_norm

    gnn_metrics = compute_metrics(y_test, gnn_test)
    benford_metrics = compute_metrics(y_test, benford_norm)
    hybrid_metrics = compute_metrics(y_test, hybrid)

    print(f"n = {len(y_test)} test nodes\n")
    print(f"{'score':<20} {'precision':>10} {'recall':>10} {'auc_pr':>10}")
    print(f"{'GNN alone':<20} {gnn_metrics.precision:>10.4f} {gnn_metrics.recall:>10.4f} {gnn_metrics.auc_pr:>10.4f}")
    print(f"{'Benford alone':<20} {benford_metrics.precision:>10.4f} {benford_metrics.recall:>10.4f} {benford_metrics.auc_pr:>10.4f}")
    print(f"{'Hybrid (0.7/0.3)':<20} {hybrid_metrics.precision:>10.4f} {hybrid_metrics.recall:>10.4f} {hybrid_metrics.auc_pr:>10.4f}")

    delta = hybrid_metrics.auc_pr - gnn_metrics.auc_pr
    print(f"\nhybrid AUC-PR vs. GNN-alone AUC-PR: {delta:+.4f}")
    if delta <= 0:
        print(
            "Honest finding: the hybrid score does NOT beat the GNN alone here. "
            "Benford's near-zero correlation with the GNN score (Phase 4: r=0.042) "
            "means blending it in adds noise rather than complementary signal, at "
            "least at this weighting and on this (already-undertrained) Elliptic++ "
            "GraphSAGE model."
        )
    else:
        print("The hybrid score does modestly improve on the GNN alone.")

    results = {
        "n_test": int(len(y_test)),
        "gnn_alone": {"precision": gnn_metrics.precision, "recall": gnn_metrics.recall, "auc_pr": gnn_metrics.auc_pr},
        "benford_alone": {"precision": benford_metrics.precision, "recall": benford_metrics.recall, "auc_pr": benford_metrics.auc_pr},
        "hybrid": {"precision": hybrid_metrics.precision, "recall": hybrid_metrics.recall, "auc_pr": hybrid_metrics.auc_pr},
        "hybrid_beats_gnn_alone": bool(delta > 0),
    }
    with open("models/results/hybrid_score_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved to models/results/hybrid_score_evaluation.json")


if __name__ == "__main__":
    main()
