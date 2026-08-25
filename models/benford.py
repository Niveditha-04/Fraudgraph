"""
Phase 4: Benford's Law deviation score.

Data source decision, flagged because it deviates from the brief's literal
instruction to use "transaction amount features": base Elliptic
(EllipticBitcoinDataset, used in Phase 3) does NOT expose real transaction
amounts. Its 165 features are confirmed anonymized by the dataset's own
paper/documentation ("the semantics of each dataset feature were not
published... all transaction features were anonymized to hide their
original values") -- verified via the original Elliptic paper's
documentation, not assumed from eyeballing the values (though the raw CSV
values, small signed floats, are also visually consistent with a PCA/
z-score transform, not raw BTC amounts). Running Benford's Law -- a test
that specifically depends on real, unscaled, scale-invariant magnitudes --
on anonymized/normalized features would produce a number that looks like an
analysis but measures nothing real.

Elliptic++'s wallet-address dataset (used in Phase 3.5) DOES expose genuine,
interpretable BTC amount fields (btc_transacted_*, btc_sent_*,
btc_received_*), so Phase 4 uses that data instead, paired with the Phase
3.5 GraphSAGE model's illicit-probability scores on the same wallet nodes.

Per-node granularity: wallets_features.csv gives only SUMMARY statistics per
wallet (total/min/max/mean/median for each of transacted/sent/received),
not the raw list of individual transaction amounts -- so a classical
per-entity Benford chi-square test (which wants many raw observations per
entity) isn't available at this granularity. Instead, each wallet's 15
btc_* summary values are treated as a small sample, and deviation from
Benford's expected first-digit distribution is measured via Mean Absolute
Deviation (MAD) between observed and expected first-digit proportions --
Nigrini's standard summary statistic for Benford analysis, which remains
well-defined (if noisier) at small sample sizes. This is a real,
disclosed limitation (n=15 per wallet, not hundreds of raw transactions),
not something to imply is a full-fidelity per-transaction Benford test.
"""
import numpy as np
import torch

BENFORD_EXPECTED = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])  # P(first digit = d), d=1..9

# indices into the 55-column feature vector built by data/prepare_elliptic_pp.py
# (order matches wallets_features.csv's column order, minus 'address'/'Time step')
BTC_AMOUNT_FEATURE_NAMES = [
    "btc_transacted_total", "btc_transacted_min", "btc_transacted_max",
    "btc_transacted_mean", "btc_transacted_median",
    "btc_sent_total", "btc_sent_min", "btc_sent_max", "btc_sent_mean", "btc_sent_median",
    "btc_received_total", "btc_received_min", "btc_received_max",
    "btc_received_mean", "btc_received_median",
]


def get_btc_amount_column_indices(all_feature_names: list[str]) -> list[int]:
    idx = [all_feature_names.index(name) for name in BTC_AMOUNT_FEATURE_NAMES]
    assert len(idx) == len(BTC_AMOUNT_FEATURE_NAMES)
    return idx


def first_significant_digit(values: np.ndarray) -> np.ndarray:
    """First significant digit of each positive value, e.g. 0.0034 -> 3, 152.7 -> 1."""
    values = values[values > 0]
    if len(values) == 0:
        return np.array([], dtype=int)
    exponents = np.floor(np.log10(values))
    mantissas = values / (10 ** exponents)
    digits = np.floor(mantissas).astype(int)
    digits = np.clip(digits, 1, 9)  # guard float edge cases (e.g. 9.9999999 -> 10)
    return digits


def benford_mad(values: np.ndarray) -> float | None:
    """Mean Absolute Deviation between observed and Benford-expected first-digit
    proportions (Nigrini's MAD statistic). Returns None if fewer than 3 positive
    values are available (not enough to say anything)."""
    digits = first_significant_digit(np.asarray(values, dtype=float))
    if len(digits) < 3:
        return None
    observed = np.array([(digits == d).mean() for d in range(1, 10)])
    return float(np.mean(np.abs(observed - BENFORD_EXPECTED)))


def compute_node_benford_scores(x_raw: torch.Tensor, amount_col_idx: list[int]) -> np.ndarray:
    """Per-node Benford MAD score. NaN for nodes with fewer than 3 positive amount values."""
    amounts = x_raw[:, amount_col_idx].numpy()
    scores = np.full(amounts.shape[0], np.nan)
    for i in range(amounts.shape[0]):
        mad = benford_mad(amounts[i])
        if mad is not None:
            scores[i] = mad
    return scores
