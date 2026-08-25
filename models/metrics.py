"""
Metrics for the illicit (positive, label=1) class.

Per the project brief: accuracy is explicitly excluded as a reported metric
-- with illicit at ~2-10% depending on denominator, a model predicting "licit"
for everything would score >90% accuracy while catching zero fraud. Only
precision, recall, and AUC-PR (average precision) are computed.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score


@dataclass
class Metrics:
    precision: float
    recall: float
    auc_pr: float
    n_pos: int
    n_total: int

    def __str__(self) -> str:
        return (
            f"precision={self.precision:.4f}  recall={self.recall:.4f}  "
            f"auc_pr={self.auc_pr:.4f}  (n_pos={self.n_pos}/{self.n_total})"
        )


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Metrics:
    """
    y_true: 0/1 array, 1 = illicit
    y_score: predicted probability of illicit, in [0, 1]
    """
    y_pred = (y_score >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    auc_pr = average_precision_score(y_true, y_score)
    return Metrics(
        precision=float(precision),
        recall=float(recall),
        auc_pr=float(auc_pr),
        n_pos=int(y_true.sum()),
        n_total=int(len(y_true)),
    )
