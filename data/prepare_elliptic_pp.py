"""
Phase 3.5 (optional stretch): build a PyG graph from the Elliptic++
wallet-address extension.

Design decisions, documented because they are not neutral defaults:

1. Scope: a STANDALONE wallet-address graph (nodes = unique addresses from
   wallets_classes.csv, edges = AddrAddr_edgelist.csv), not a merged
   heterogeneous tx+wallet graph. The brief's Phase 3.5 asks to "re-run the
   same GraphSAGE/GAT training and evaluation from Phase 3 on the extended
   wallet-address graph" and "report both result sets side by side" -- i.e.
   a parallel, comparable task, not a fused hetero-graph requiring a
   different GNN architecture. AddrTx/TxAddr edgelists (linking wallets to
   base-Elliptic transactions) are downloaded but not used here; a merged
   hetero graph is a natural follow-up but is materially more complex
   (needs HeteroConv, ~1.03M total nodes / ~4.4M total edges) and isn't what
   the brief's wording asks for.

2. Node granularity: ONE node per unique wallet address (822,942 nodes,
   matching wallets_classes.csv exactly and the brief's "822,000 wallet
   address nodes" figure), not one node per (address, time_step) row.
   wallets_features.csv has 1,268,260 rows because a wallet can have a
   feature snapshot recorded at each time step it was active in (verified:
   822,942 unique addresses across those rows, matching wallets_classes.csv
   exactly) -- this row count matches the brief's "1.27 million temporal
   interactions" figure, clarifying that figure refers to these per-wallet
   temporal feature snapshots, not to edge count. For each address, this
   loader keeps the snapshot from its LAST (most recent) active time step,
   since the feature columns (total_txs, lifetime_in_blocks,
   num_timesteps_appeared_in, etc.) read as cumulative/lifetime statistics
   that are most complete at a wallet's final observed snapshot.

3. Each node's assigned time step (for the train/val/test split) is that
   same last-active time step. Verified the "Time step" column range is
   1-49, the same numbering as base Elliptic, so the brief's split
   boundaries (train 1-34 / val 35-39 / test 40-49) apply unchanged.

4. Label encoding: wallets_classes.csv uses class values confirmed against
   the official EllipticPlusPlus README (not assumed): 1=illicit (14,266),
   2=licit (251,088), 3=unknown (557,588) -- counts cross-checked exactly
   against the README's published table. Remapped here to the same
   convention used throughout this project (0=licit, 1=illicit, 2=unknown,
   matching base EllipticBitcoinDataset) so downstream code (metrics,
   training harness) doesn't need a parallel code path.

5. Feature columns: all 55 numeric columns except 'address' and 'Time step'
   (wallets_features.csv has 57 total columns; a public search summary
   claimed "56 features" but that undercounts by one relative to the actual
   file -- verified directly against the downloaded CSV's header, not
   trusted from that secondary source). 'Time step' itself is excluded from
   the feature matrix, mirroring the base Elliptic loader's exclusion of its
   own time-step column -- test-set time steps (40-49) never appear in
   train (1-34), so keeping it as a raw feature would let the model see
   out-of-training-range values at test time for no modeling benefit.
"""
import time

import pandas as pd
import torch
from torch_geometric.data import Data

RAW_DIR = "data/elliptic_pp/Elliptic++ Dataset"
CACHE_PATH = "data/elliptic_pp/processed_wallet_graph.pt"

EXPECTED_ADDRESSES = 822_942
EXPECTED_ILLICIT = 14_266
EXPECTED_LICIT = 251_088
EXPECTED_UNKNOWN = 557_588
EXPECTED_FEATURE_COLS = 55


def build_graph() -> Data:
    t0 = time.time()
    print("loading wallets_classes.csv ...")
    classes_df = pd.read_csv(f"{RAW_DIR}/wallets_classes.csv")
    assert len(classes_df) == EXPECTED_ADDRESSES, (
        f"expected {EXPECTED_ADDRESSES} wallet addresses, got {len(classes_df)}"
    )
    class_counts = classes_df["class"].value_counts().to_dict()
    assert class_counts.get(1, 0) == EXPECTED_ILLICIT
    assert class_counts.get(2, 0) == EXPECTED_LICIT
    assert class_counts.get(3, 0) == EXPECTED_UNKNOWN
    print(f"  {len(classes_df)} addresses, class counts confirmed against README: {class_counts}")

    # remap: raw 1=illicit,2=licit,3=unknown -> project convention 0=licit,1=illicit,2=unknown
    label_map = {1: 1, 2: 0, 3: 2}
    classes_df["y"] = classes_df["class"].map(label_map)

    print("loading wallets_features.csv (606MB, this takes a bit) ...")
    feat_df = pd.read_csv(f"{RAW_DIR}/wallets_features.csv")
    n_feature_cols = feat_df.shape[1] - 2  # minus 'address' and 'Time step'
    assert n_feature_cols == EXPECTED_FEATURE_COLS, (
        f"expected {EXPECTED_FEATURE_COLS} feature columns, got {n_feature_cols}"
    )

    n_unique_in_features = feat_df["address"].nunique()
    assert n_unique_in_features == EXPECTED_ADDRESSES, (
        f"unique addresses in features file ({n_unique_in_features}) != "
        f"wallets_classes.csv ({EXPECTED_ADDRESSES}) -- id space mismatch"
    )
    print(f"  {len(feat_df)} rows, {n_unique_in_features} unique addresses, {n_feature_cols} feature cols")

    # keep each address's LAST (most recent) time-step snapshot
    feat_df = feat_df.sort_values("Time step").drop_duplicates(subset="address", keep="last")
    assert len(feat_df) == EXPECTED_ADDRESSES

    # canonical node ordering: the classes file (one row per address)
    node_order = classes_df[["address", "y"]].copy()
    merged = node_order.merge(feat_df, on="address", how="left")
    assert len(merged) == EXPECTED_ADDRESSES
    assert merged["Time step"].isna().sum() == 0, "some addresses have no feature snapshot"

    addr_to_idx = {addr: i for i, addr in enumerate(merged["address"].values)}

    feature_cols = [c for c in merged.columns if c not in ("address", "y", "Time step", "class")]
    assert len(feature_cols) == EXPECTED_FEATURE_COLS
    x = torch.tensor(merged[feature_cols].values, dtype=torch.float)
    y = torch.tensor(merged["y"].values, dtype=torch.long)
    time_step = torch.tensor(merged["Time step"].values, dtype=torch.long)

    print("loading AddrAddr_edgelist.csv ...")
    edge_df = pd.read_csv(f"{RAW_DIR}/AddrAddr_edgelist.csv")
    n_edges_raw = len(edge_df)
    src = edge_df["input_address"].map(addr_to_idx)
    dst = edge_df["output_address"].map(addr_to_idx)
    unmapped = src.isna().sum() + dst.isna().sum()
    print(f"  {n_edges_raw} raw edges, {unmapped} endpoints failed to map to a known address")
    valid = src.notna() & dst.notna()
    edge_index = torch.tensor(
        [src[valid].astype(int).values, dst[valid].astype(int).values], dtype=torch.long
    )
    print(f"  {edge_index.shape[1]} edges kept after mapping ({n_edges_raw - edge_index.shape[1]} dropped)")

    data = Data(x=x, edge_index=edge_index, y=y)
    data.time_step = time_step
    data.feature_names = feature_cols

    print(f"\nbuilt graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges, {x.shape[1]} features")
    print(f"took {time.time() - t0:.1f}s")
    return data


def validate_and_cache(force: bool = False) -> Data:
    import os

    if not force and os.path.exists(CACHE_PATH):
        print(f"loading cached graph from {CACHE_PATH}")
        data = torch.load(CACHE_PATH, weights_only=False)
    else:
        data = build_graph()
        torch.save(data, CACHE_PATH)
        print(f"cached to {CACHE_PATH}")

    n_nodes = data.num_nodes
    n_edges = data.edge_index.shape[1]
    vals, counts = torch.unique(data.y, return_counts=True)
    label_counts = dict(zip(vals.tolist(), counts.tolist()))

    checks = {
        "node_count": (n_nodes, EXPECTED_ADDRESSES, n_nodes == EXPECTED_ADDRESSES),
        "illicit_count": (label_counts.get(1, 0), EXPECTED_ILLICIT, label_counts.get(1, 0) == EXPECTED_ILLICIT),
        "licit_count": (label_counts.get(0, 0), EXPECTED_LICIT, label_counts.get(0, 0) == EXPECTED_LICIT),
        "unknown_count": (label_counts.get(2, 0), EXPECTED_UNKNOWN, label_counts.get(2, 0) == EXPECTED_UNKNOWN),
    }
    print(f"\n{'check':<15} {'actual':>10} {'expected':>10}   pass")
    all_pass = True
    for name, (actual, expected, ok) in checks.items():
        print(f"{name:<15} {actual:>10} {expected:>10}   {ok}")
        all_pass = all_pass and ok
    print(f"\nGATE {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        raise SystemExit(1)

    labeled = data.y != 2
    train_mask = (data.time_step >= 1) & (data.time_step <= 34) & labeled
    val_mask = (data.time_step >= 35) & (data.time_step <= 39) & labeled
    test_mask = (data.time_step >= 40) & (data.time_step <= 49) & labeled
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    print(
        f"\ntemporal split: train={int(train_mask.sum())} "
        f"val={int(val_mask.sum())} test={int(test_mask.sum())}"
    )
    return data


if __name__ == "__main__":
    validate_and_cache()
