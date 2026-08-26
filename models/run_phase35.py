"""
Phase 3.5 (optional stretch): re-run baseline + GraphSAGE + GAT on the
Elliptic++ wallet-address graph, same methodology as Phase 3 (temporal
split, weighted loss, precision/recall/AUC-PR). Results are reported
side-by-side with base Elliptic in the README, not as a replacement.
"""
import json
import os

import torch

from models.data_utils_pp import prepare_pp_data
from models.gnn_models import GAT, GraphSAGE
from models.metrics import Metrics
from models.train_baseline import train_baseline
from models.train_gnn import train_gnn

# hidden_dim/epoch budget reduced from the Phase 3 defaults (128 dim, up to
# 300 epochs) for this graph specifically. Root cause, diagnosed rather than
# guessed: this is the user's everyday laptop (Chrome with many tabs,
# WhatsApp, Webex, the Claude desktop app all resident) sharing RAM with
# training, not a dedicated ML machine -- confirmed via `sysctl vm.swapusage`
# showing heavy swap use that did NOT drop after killing the training
# process, meaning the pressure was ambient/system-wide, not caused by this
# script's own memory footprint alone. A full-batch run at hidden_dim=128
# was observed thrashing (CPU utilization ~18% of wall-clock -- the OS
# paging, not the model computing). NeighborLoader mini-batch sampling (the
# standard production fix for graphs at this scale, ~823K nodes) was
# attempted but requires pyg-lib or torch-sparse, neither of which has a
# working installable build in this environment against this torch version
# (see models/train_gnn_sampled.py, implemented but unused for this reason).
# hidden_dim=32 alone was smoke-tested clean (no thrashing) for a few
# epochs but still degraded over a sustained 300-epoch run under ambient
# load. The user was asked how to proceed (AskUserQuestion) and chose a
# smaller/faster config over freeing up RAM or stopping outright: hidden_dim
# 16, a tight epoch cap. Justified by the smoke test already showing
# GraphSAGE beat the baseline within 2-3 epochs on this data, so a large
# epoch budget is unlikely to be strictly necessary here. This is a real,
# disclosed compute-constraint tradeoff -- reported honestly in the README
# alongside the numbers it produced, not silently applied.
WALLET_GRAPH_HIDDEN_DIM = 16
WALLET_GRAPH_EPOCHS = 25
WALLET_GRAPH_PATIENCE = 6

SEED = 42


def metrics_to_dict(m: Metrics) -> dict:
    return {"precision": m.precision, "recall": m.recall, "auc_pr": m.auc_pr, "n_pos": m.n_pos, "n_total": m.n_total}


def main():
    torch.manual_seed(SEED)

    pdata = prepare_pp_data()
    in_dim = pdata.x.shape[1]
    print(f"\nwallet graph: {pdata.x.shape[0]} nodes, {in_dim} features, "
          f"{pdata.edge_index_undirected.shape[1]} undirected edges")

    print("\n" + "=" * 70)
    print("1/3 -- Logistic Regression baseline (no graph)")
    print("=" * 70)
    baseline_test = train_baseline(pdata)
    print(f"baseline (test): {baseline_test}")

    print("\n" + "=" * 70)
    print("2/3 -- GraphSAGE (weighted cross-entropy)")
    print("=" * 70)
    torch.manual_seed(SEED)
    sage = GraphSAGE(in_dim, hidden_dim=WALLET_GRAPH_HIDDEN_DIM)
    sage_val, sage_test, sage_hist, _ = train_gnn(sage, pdata, epochs=WALLET_GRAPH_EPOCHS, patience=WALLET_GRAPH_PATIENCE)
    print(f"GraphSAGE (val):  {sage_val}")
    print(f"GraphSAGE (test): {sage_test}")

    print("\n" + "=" * 70)
    print("3/3 -- GAT (weighted cross-entropy)")
    print("=" * 70)
    torch.manual_seed(SEED)
    gat = GAT(in_dim, hidden_dim=WALLET_GRAPH_HIDDEN_DIM, heads=4)
    gat_val, gat_test, gat_hist, _ = train_gnn(gat, pdata, epochs=WALLET_GRAPH_EPOCHS, patience=WALLET_GRAPH_PATIENCE)
    print(f"GAT (val):  {gat_val}")
    print(f"GAT (test): {gat_test}")

    print("\n" + "=" * 70)
    print("Phase 3.5 results summary (Elliptic++ wallet graph, test split t=40-49)")
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
        "dataset": "elliptic_pp_wallet_graph",
        "baseline": metrics_to_dict(baseline_test),
        "graphsage": {"val": metrics_to_dict(sage_val), "test": metrics_to_dict(sage_test)},
        "gat": {"val": metrics_to_dict(gat_val), "test": metrics_to_dict(gat_test)},
        "best_gnn": best_gnn_name,
        "gate_pass": gate_pass,
    }
    os.makedirs("models/results", exist_ok=True)
    with open("models/results/phase35_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results to models/results/phase35_results.json")

    return results


if __name__ == "__main__":
    main()
