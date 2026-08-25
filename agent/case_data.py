"""
Phase 6: build investigation cases from real Phase 3.5/Phase 4 output --
not synthetic/fabricated data. Each case is a wallet node from the Elliptic++
graph's test split (t=40-49, held out from GNN training), with its real
GNN illicit-probability, Benford deviation score, hybrid score, and a
handful of real feature values summarized into text for the LLM personas.

Test-case selection (for Phase 6's gate: "at least 10 test cases, mix of
illicit and licit ground truth, and needs_human_review must actually
trigger on at least one deliberately ambiguous case") deliberately samples
across the hybrid-score range rather than randomly, so the test set
includes both confident and borderline cases -- a random sample skews
toward the licit majority and might never produce genuine model
disagreement, which would leave the human-review branch untested by
accident rather than by a deliberate ambiguous case.
"""
from dataclasses import dataclass

import numpy as np

from data.prepare_elliptic_pp import validate_and_cache

SCORES_PATH = "models/results/phase4_node_scores.npz"


@dataclass
class Case:
    case_id: int
    node_idx: int
    ground_truth: str  # "illicit" / "licit" / "unknown" -- for eval only, never shown to the LLM
    gnn_score: float
    benford_score: float
    hybrid_score: float
    evidence_summary: str


def _label_to_str(y: int) -> str:
    return {0: "licit", 1: "illicit", 2: "unknown"}[y]


def _min_max(x: np.ndarray) -> np.ndarray:
    finite = x[np.isfinite(x)]
    lo, hi = finite.min(), finite.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def build_evidence_summary(node_idx: int, feature_names: list[str], x_raw: np.ndarray, gnn_score: float,
                            benford_score: float, hybrid_score: float) -> str:
    row = x_raw[node_idx]
    feat = dict(zip(feature_names, row))
    lines = [
        f"Wallet case #{node_idx}",
        f"- GNN (GraphSAGE) illicit-probability score: {gnn_score:.3f} (0=licit-like, 1=illicit-like; "
        f"model trained on labeled Elliptic++ wallets, temporal holdout test AUC-PR 0.428)",
        f"- Benford's Law deviation score (MAD): {benford_score:.4f} (0=amounts closely follow Benford's "
        f"Law's expected first-digit distribution; higher = more deviation from natural transaction amounts)",
        f"- Combined hybrid score: {hybrid_score:.3f} (0.7*GNN + 0.3*Benford, both normalized 0-1)",
        "",
        "Wallet activity summary:",
        f"- total transactions: {feat['total_txs']:.0f} (as sender: {feat['num_txs_as_sender']:.0f}, "
        f"as receiver: {feat['num_txs_as receiver']:.0f})",
        f"- active over {feat['num_timesteps_appeared_in']:.0f} distinct time steps, "
        f"lifetime {feat['lifetime_in_blocks']:.0f} blocks",
        f"- BTC transacted: total={feat['btc_transacted_total']:.4f}, "
        f"mean={feat['btc_transacted_mean']:.4f}, median={feat['btc_transacted_median']:.4f}",
        f"- BTC sent: total={feat['btc_sent_total']:.4f}, BTC received: total={feat['btc_received_total']:.4f}",
        f"- fees paid: total={feat['fees_total']:.6f}",
    ]
    return "\n".join(lines)


def build_test_cases(n_cases: int = 10, seed: int = 42) -> list[Case]:
    scores = np.load(SCORES_PATH)
    gnn_probs = scores["gnn_probs"]
    benford_scores = scores["benford_scores"]
    y = scores["y"]
    test_mask = scores["test_mask"]

    raw_data = validate_and_cache()
    x_raw = raw_data.x.numpy()
    feature_names = raw_data.feature_names

    test_idx = np.where(test_mask)[0]
    valid = test_idx[np.isfinite(benford_scores[test_idx]) & (y[test_idx] != 2)]  # scorable, labeled only

    gnn_norm = _min_max(gnn_probs[valid])
    benford_norm = _min_max(benford_scores[valid])
    hybrid = 0.7 * gnn_norm + 0.3 * benford_norm

    rng = np.random.default_rng(seed)
    n = len(hybrid)
    y_valid = y[valid]

    # Stratify by GROUND TRUTH first, not just hybrid-score bucket: illicit is
    # a small minority even within a "high score" bucket (GraphSAGE's real
    # test precision is ~0.10-0.15, see Phase 3.5/4 -- most high-scoring
    # nodes are false positives), so naive score-based sampling drew 9
    # licit/1 illicit out of 10 in an earlier version of this function. That
    # technically satisfies "mix of illicit and licit" but makes for a weak
    # demo of the panel actually reviewing real illicit cases. Guarantee a
    # meaningful count of each ground-truth class, then vary hybrid score
    # within each stratum so both confident and borderline cases appear.
    illicit_local = np.where(y_valid == 1)[0]
    licit_local = np.where(y_valid == 0)[0]

    n_illicit_picks = max(3, n_cases // 3)
    n_licit_picks = n_cases - n_illicit_picks

    # within each stratum, spread picks across the hybrid-score range rather
    # than randomly, so some are "confident" and some are "borderline"
    def spread_pick(pool_idx: np.ndarray, k: int) -> list[int]:
        pool_sorted = pool_idx[np.argsort(hybrid[pool_idx])]
        if len(pool_sorted) <= k:
            return list(pool_sorted)
        positions = np.linspace(0, len(pool_sorted) - 1, k).round().astype(int)
        return list(pool_sorted[positions])

    picks = spread_pick(illicit_local, n_illicit_picks) + spread_pick(licit_local, n_licit_picks)
    rng.shuffle(picks)
    picks = picks[:n_cases]

    cases = []
    for i, local_idx in enumerate(picks):
        node_idx = int(valid[local_idx])
        case = Case(
            case_id=i,
            node_idx=node_idx,
            ground_truth=_label_to_str(int(y[node_idx])),
            gnn_score=float(gnn_probs[node_idx]),
            benford_score=float(benford_scores[node_idx]),
            hybrid_score=float(hybrid[local_idx]),
            evidence_summary=build_evidence_summary(
                node_idx, feature_names, x_raw,
                float(gnn_probs[node_idx]), float(benford_scores[node_idx]), float(hybrid[local_idx]),
            ),
        )
        cases.append(case)
    return cases
