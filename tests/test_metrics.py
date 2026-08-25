"""Unit tests for models/metrics.py -- the metric computation the whole
project's honest-reporting claims (precision/recall/AUC-PR, never accuracy)
depend on, so it's worth pinning down with known synthetic inputs."""
import numpy as np

from models.metrics import compute_metrics


def test_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    m = compute_metrics(y_true, y_score)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.auc_pr == 1.0


def test_worst_case_inverted_scores():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])  # scores exactly backwards
    m = compute_metrics(y_true, y_score)
    assert m.precision == 0.0
    assert m.recall == 0.0


def test_n_pos_and_n_total_are_correct():
    y_true = np.array([0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.9, 0.1, 0.9])
    m = compute_metrics(y_true, y_score)
    assert m.n_pos == 3
    assert m.n_total == 5


def test_threshold_changes_precision_recall_but_not_auc_pr():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.3, 0.4, 0.5, 0.6, 0.9])
    low_thresh = compute_metrics(y_true, y_score, threshold=0.1)
    high_thresh = compute_metrics(y_true, y_score, threshold=0.95)
    assert low_thresh.auc_pr == high_thresh.auc_pr  # AUC-PR is threshold-independent
    assert low_thresh.recall >= high_thresh.recall
