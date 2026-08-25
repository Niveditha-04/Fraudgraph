"""
Phase 1 gate: verify the Elliptic Bitcoin dataset loads correctly via
torch_geometric and matches the published counts from the project brief.

Label encoding (PyG's EllipticBitcoinDataset), confirmed by count cross-reference
against the brief's published figures AND against the library source
(torch_geometric/datasets/elliptic.py, `mapping = {'unknown': 2, '1': 1, '2': 0}`):
    0 = licit    (~42,019 nodes)
    1 = illicit  (~4,545 nodes)
    2 = unknown  (~157,205 nodes, the unlabeled ~77%)
This differs from the raw CSV's own class encoding (raw '1'=illicit, raw
'2'=licit, "unknown"=unlabeled) -- the two encodings are NOT the same.

IMPORTANT: EllipticBitcoinDataset.process() builds `data.x` from
`feat_df.loc[:, 2:]` -- i.e. it drops columns 0 (txId) and 1 (time_step)
entirely. The time step is NOT column 0 of data.x (a wrong assumption this
script initially made and then corrected after reading the library source).
The time step must be read from the raw features CSV directly. Row order in
the raw features CSV is confirmed (empirically, below) to align 1:1 with
`data.x`/`data.y` row order, since `process()` never reorders feat_df rows
and class_df is row-aligned to feat_df by txId.

Also note: PyG's built-in `data.train_mask`/`data.test_mask` only implement a
2-way split (train: time_step<35, test: time_step>=35, both excluding
unlabeled y==2 nodes) -- NOT the brief's required 3-way split
(train 1-34 / val 35-39 / test 40-49). Custom masks are built here instead.
"""
import pandas as pd
import torch
from torch_geometric.datasets import EllipticBitcoinDataset

EXPECTED_NODES = 203_769
EXPECTED_EDGES = 234_355
EXPECTED_TIME_STEPS = 49
EXPECTED_ILLICIT = 4_545
EXPECTED_LICIT = 42_019
TOLERANCE = 50


def load_time_steps(raw_dir: str, data) -> torch.Tensor:
    feat_df = pd.read_csv(f"{raw_dir}/elliptic_txs_features.csv", header=None)
    class_df = pd.read_csv(f"{raw_dir}/elliptic_txs_classes.csv")

    # Row-alignment is not assumed -- verify it before trusting time_step order.
    aligned = (feat_df[0].values == class_df["txId"].values).all()
    if not aligned:
        raise AssertionError(
            "raw features CSV and classes CSV are not row-aligned by txId; "
            "cannot safely attach time_step to data.x/data.y by position."
        )

    time_step = torch.from_numpy(feat_df[1].values)

    # Cross-check against data.y using the known class-count mapping, since
    # data.y is built from class_df in the same (verified-aligned) row order.
    mapping = {"unknown": 2, "1": 1, "2": 0}
    y_from_raw = torch.from_numpy(class_df["class"].map(mapping).values)
    assert torch.equal(y_from_raw, data.y), (
        "recomputed labels from raw CSV do not match data.y -- row alignment "
        "or class mapping assumption is wrong, do not trust time_step below."
    )
    return time_step


def build_temporal_masks(time_step: torch.Tensor, y: torch.Tensor):
    # Brief's required split: train 1-34, val 35-39, test 40-49.
    # Labeled nodes only (y != 2) -- unlabeled nodes can't be used for
    # supervised training/eval regardless of split.
    labeled = y != 2
    train_mask = (time_step >= 1) & (time_step <= 34) & labeled
    val_mask = (time_step >= 35) & (time_step <= 39) & labeled
    test_mask = (time_step >= 40) & (time_step <= 49) & labeled
    return train_mask, val_mask, test_mask


def validate():
    dataset = EllipticBitcoinDataset(root="./data/elliptic")
    data = dataset[0]

    n_nodes = data.num_nodes
    n_edges = data.edge_index.shape[1]

    vals, counts = torch.unique(data.y, return_counts=True)
    label_counts = dict(zip(vals.tolist(), counts.tolist()))
    n_licit = label_counts.get(0, 0)
    n_illicit = label_counts.get(1, 0)
    n_unknown = label_counts.get(2, 0)

    time_step = load_time_steps(f"{dataset.raw_dir}", data)
    n_time_steps = int(time_step.unique().numel())

    checks = {
        "node_count": (n_nodes, EXPECTED_NODES, n_nodes == EXPECTED_NODES),
        "edge_count": (n_edges, EXPECTED_EDGES, n_edges == EXPECTED_EDGES),
        "time_steps": (
            n_time_steps,
            EXPECTED_TIME_STEPS,
            n_time_steps == EXPECTED_TIME_STEPS,
        ),
        "illicit_count": (
            n_illicit,
            EXPECTED_ILLICIT,
            abs(n_illicit - EXPECTED_ILLICIT) <= TOLERANCE,
        ),
        "licit_count": (
            n_licit,
            EXPECTED_LICIT,
            abs(n_licit - EXPECTED_LICIT) <= TOLERANCE,
        ),
    }

    print(f"{'check':<15} {'actual':>10} {'expected':>10}   pass")
    all_pass = True
    for name, (actual, expected, ok) in checks.items():
        print(f"{name:<15} {actual:>10} {expected:>10}   {ok}")
        all_pass = all_pass and ok

    print(f"\nunknown (unlabeled) node count: {n_unknown} ({n_unknown/n_nodes:.1%})")

    train_mask, val_mask, test_mask = build_temporal_masks(time_step, data.y)
    print(f"\ntemporal split (labeled nodes only):")
    print(f"  train (t 1-34):  {int(train_mask.sum())}")
    print(f"  val   (t 35-39): {int(val_mask.sum())}")
    print(f"  test  (t 40-49): {int(test_mask.sum())}")
    overlap = (train_mask & val_mask).any() or (val_mask & test_mask).any() or (train_mask & test_mask).any()
    print(f"  split masks mutually exclusive: {not overlap}")
    all_pass = all_pass and (not overlap)

    print(f"\nGATE {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        raise SystemExit(1)

    data.time_step = time_step
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    return data


if __name__ == "__main__":
    validate()
