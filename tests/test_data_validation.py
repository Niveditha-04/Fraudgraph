"""
Integration test: re-runs the Phase 1 data-validation gate (data/validate_phase1.py).
Downloads the Elliptic Bitcoin dataset if not already cached (~a few hundred MB,
public, no auth needed) -- this is the one test that hits the network, kept
because the whole project's Phase 3+ results are only meaningful if these
counts still match the published dataset.
"""
from data.validate_phase1 import validate


def test_elliptic_dataset_matches_published_counts():
    data = validate()  # raises SystemExit(1) internally if any check fails
    assert data.num_nodes == 203_769
    assert data.edge_index.shape[1] == 234_355
    assert int(data.train_mask.sum()) + int(data.val_mask.sum()) + int(data.test_mask.sum()) <= data.num_nodes


def test_temporal_split_masks_are_mutually_exclusive():
    data = validate()
    overlap_train_val = (data.train_mask & data.val_mask).any()
    overlap_val_test = (data.val_mask & data.test_mask).any()
    overlap_train_test = (data.train_mask & data.test_mask).any()
    assert not overlap_train_val
    assert not overlap_val_test
    assert not overlap_train_test
