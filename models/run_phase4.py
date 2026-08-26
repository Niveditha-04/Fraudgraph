"""
Phase 4: hybrid statistical score.

1. Benford's Law deviation score per wallet (models/benford.py), computed on
   the Elliptic++ wallet graph's genuine BTC amount fields -- NOT on base
   Elliptic, whose features are confirmed anonymized (see benford.py's
   docstring for the sourcing decision and why).
2. GNN illicit-probability score per wallet: retrains the same GraphSAGE
   config used in Phase 3.5 (hidden_dim=16, epoch-capped) to get live
   per-node probabilities -- Phase 3.5's run only persisted summary metrics,
   not raw per-node scores, so this reruns training rather than fabricating
   scores from the saved metrics.
3. Hybrid score: documented weighted average of the two, both min-max
   normalized first so neither dominates purely due to differing scales.
4. Pearson correlation between the Benford score and the GNN score is
   printed and reported honestly, whatever it turns out to be -- a low or
   near-zero correlation is a valid, reportable finding (it would suggest
   the two signals are complementary/independent, which is actually a
   reasonable argument FOR combining them), not a failure to hide.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

from data.prepare_elliptic_pp import validate_and_cache
from models.benford import compute_node_benford_scores, get_btc_amount_column_indices
from models.data_utils_pp import prepare_pp_data
from models.gnn_models import GraphSAGE
from models.hybrid_config import HYBRID_WEIGHT_BENFORD, HYBRID_WEIGHT_GNN
from models.train_gnn import train_gnn

SEED = 42
HIDDEN_DIM = 16
EPOCHS = 25
PATIENCE = 6


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    finite = x[np.isfinite(x)]
    lo, hi = finite.min(), finite.max()
    if hi == lo:
        return np.zeros_like(x)
    out = (x - lo) / (hi - lo)
    return out


def main():
    torch.manual_seed(SEED)

    raw_data = validate_and_cache()
    amount_col_idx = get_btc_amount_column_indices(raw_data.feature_names)
    print(f"Benford amount columns ({len(amount_col_idx)}): {[raw_data.feature_names[i] for i in amount_col_idx]}")

    print("\ncomputing per-node Benford MAD scores ...")
    benford_scores = compute_node_benford_scores(raw_data.x, amount_col_idx)
    n_valid_benford = np.isfinite(benford_scores).sum()
    print(f"  {n_valid_benford}/{len(benford_scores)} nodes have >=3 positive amount values (scorable)")

    print("\nretraining GraphSAGE (same config as Phase 3.5) for per-node illicit-probability scores ...")
    torch.manual_seed(SEED)
    pdata = prepare_pp_data()
    in_dim = pdata.x.shape[1]
    model = GraphSAGE(in_dim, hidden_dim=HIDDEN_DIM)
    val_m, test_m, hist, _ = train_gnn(model, pdata, epochs=EPOCHS, patience=PATIENCE)
    print(f"  test AUC-PR: {test_m.auc_pr:.4f} (sanity check vs Phase 3.5's 0.4283)")

    model.eval()
    with torch.no_grad():
        out = model(pdata.x, pdata.edge_index_undirected)
        gnn_probs = F.softmax(out, dim=1)[:, 1].numpy()  # illicit probability, ALL nodes

    test_mask_np = pdata.test_mask.numpy()
    benford_test = benford_scores[test_mask_np]
    gnn_test = gnn_probs[test_mask_np]

    both_valid = np.isfinite(benford_test) & np.isfinite(gnn_test)
    n_both = both_valid.sum()
    print(f"\ntest-set nodes with both a Benford score and a GNN score: {n_both}/{test_mask_np.sum()}")

    benford_valid = benford_test[both_valid]
    gnn_valid = gnn_test[both_valid]

    corr, p_value = pearsonr(benford_valid, gnn_valid)
    print(f"\nPearson correlation (Benford MAD vs GNN illicit-probability, test set): r={corr:.4f}, p={p_value:.4g}")

    benford_norm = min_max_normalize(benford_valid)
    gnn_norm = min_max_normalize(gnn_valid)
    hybrid = HYBRID_WEIGHT_GNN * gnn_norm + HYBRID_WEIGHT_BENFORD * benford_norm
    print(
        f"\nhybrid score = {HYBRID_WEIGHT_GNN} * gnn_score_normalized + "
        f"{HYBRID_WEIGHT_BENFORD} * benford_score_normalized"
    )
    print(f"hybrid score stats: min={hybrid.min():.4f} max={hybrid.max():.4f} mean={hybrid.mean():.4f}")

    results = {
        "n_nodes_total": int(len(benford_scores)),
        "n_nodes_benford_scorable": int(n_valid_benford),
        "n_test_nodes_both_scores": int(n_both),
        "gnn_test_auc_pr": test_m.auc_pr,
        "pearson_correlation": float(corr),
        "pearson_p_value": float(p_value),
        "hybrid_weight_gnn": HYBRID_WEIGHT_GNN,
        "hybrid_weight_benford": HYBRID_WEIGHT_BENFORD,
        "benford_score_stats": {
            "min": float(np.nanmin(benford_scores)),
            "max": float(np.nanmax(benford_scores)),
            "mean": float(np.nanmean(benford_scores)),
        },
    }
    os.makedirs("models/results", exist_ok=True)
    with open("models/results/phase4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results to models/results/phase4_results.json")

    # per-node scores for ALL nodes, persisted so Phase 6's agent can build
    # cases from real data without retraining the GNN every time it runs.
    np.savez(
        "models/results/phase4_node_scores.npz",
        gnn_probs=gnn_probs,
        benford_scores=benford_scores,
        y=pdata.y.numpy(),
        test_mask=pdata.test_mask.numpy(),
        time_step=raw_data.time_step.numpy(),
    )
    print("saved per-node scores to models/results/phase4_node_scores.npz")

    return results


if __name__ == "__main__":
    main()
