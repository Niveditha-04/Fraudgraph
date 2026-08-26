"""
Phase 3 orchestration: train baseline + GraphSAGE + GAT, report precision/
recall/AUC-PR for all three on the same held-out temporal test split, and
enforce the gate: best GNN's test AUC-PR must exceed the baseline's.
"""
import json
import os

import torch

from models.data_utils import prepare_data
from models.gnn_models import GAT, GraphSAGE
from models.metrics import Metrics
from models.train_baseline import train_baseline
from models.train_gnn import train_gnn

SEED = 42


def metrics_to_dict(m: Metrics) -> dict:
    return {"precision": m.precision, "recall": m.recall, "auc_pr": m.auc_pr, "n_pos": m.n_pos, "n_total": m.n_total}


def main():
    torch.manual_seed(SEED)

    pdata = prepare_data()
    in_dim = pdata.x.shape[1]

    print("\n" + "=" * 70)
    print("1/3 -- Logistic Regression baseline (no graph)")
    print("=" * 70)
    baseline_test = train_baseline(pdata)
    print(f"baseline (test): {baseline_test}")

    print("\n" + "=" * 70)
    print("2/3 -- GraphSAGE (weighted cross-entropy)")
    print("=" * 70)
    torch.manual_seed(SEED)
    sage = GraphSAGE(in_dim)
    sage_val, sage_test, sage_hist, _ = train_gnn(sage, pdata, epochs=300, patience=30)
    print(f"GraphSAGE (val):  {sage_val}")
    print(f"GraphSAGE (test): {sage_test}")

    print("\n" + "=" * 70)
    print("3/3 -- GAT (weighted cross-entropy)")
    print("=" * 70)
    torch.manual_seed(SEED)
    gat = GAT(in_dim)
    gat_val, gat_test, gat_hist, _ = train_gnn(gat, pdata, epochs=300, patience=30)
    print(f"GAT (val):  {gat_val}")
    print(f"GAT (test): {gat_test}")

    print("\n" + "=" * 70)
    print("Phase 3 results summary (test split, t=40-49)")
    print("=" * 70)
    rows = [
        ("Logistic Regression (baseline)", baseline_test),
        ("GraphSAGE", sage_test),
        ("GAT", gat_test),
    ]
    print(f"{'model':<32} {'precision':>10} {'recall':>10} {'auc_pr':>10}")
    for name, m in rows:
        print(f"{name:<32} {m.precision:>10.4f} {m.recall:>10.4f} {m.auc_pr:>10.4f}")

    best_gnn_name, best_gnn = max([("GraphSAGE", sage_test), ("GAT", gat_test)], key=lambda t: t[1].auc_pr)
    print(f"\nbest GNN: {best_gnn_name} (test AUC-PR = {best_gnn.auc_pr:.4f})")
    print(f"baseline: Logistic Regression (test AUC-PR = {baseline_test.auc_pr:.4f})")

    gate_pass = best_gnn.auc_pr > baseline_test.auc_pr
    print(f"\nGATE (best GNN AUC-PR > baseline AUC-PR): {'PASS' if gate_pass else 'FAIL'}")

    results = {
        "seed": SEED,
        "baseline": metrics_to_dict(baseline_test),
        "graphsage": {"val": metrics_to_dict(sage_val), "test": metrics_to_dict(sage_test)},
        "gat": {"val": metrics_to_dict(gat_val), "test": metrics_to_dict(gat_test)},
        "best_gnn": best_gnn_name,
        "gate_pass": gate_pass,
    }
    os.makedirs("models/results", exist_ok=True)
    with open("models/results/phase3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results to models/results/phase3_results.json")

    if not gate_pass:
        raise SystemExit(1)

    return results


if __name__ == "__main__":
    main()
