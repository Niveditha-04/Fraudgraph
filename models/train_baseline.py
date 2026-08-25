"""
Phase 3 step 2: Logistic Regression baseline on node features alone, no graph.
This is the bar the GNNs must clear (Phase 3 gate) to prove graph structure
adds value.
"""
from sklearn.linear_model import LogisticRegression

from models.data_utils import PreparedData, prepare_data
from models.metrics import Metrics, compute_metrics


def train_baseline(pdata: PreparedData) -> Metrics:
    x = pdata.x.numpy()
    y = pdata.y.numpy()

    x_train, y_train = x[pdata.train_mask.numpy()], y[pdata.train_mask.numpy()]
    x_test, y_test = x[pdata.test_mask.numpy()], y[pdata.test_mask.numpy()]

    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    clf.fit(x_train, y_train)

    illicit_col = list(clf.classes_).index(1)
    y_score = clf.predict_proba(x_test)[:, illicit_col]

    metrics = compute_metrics(y_test, y_score)
    return metrics


if __name__ == "__main__":
    pdata = prepare_data()
    metrics = train_baseline(pdata)
    print(f"\nLogistic Regression baseline (test set): {metrics}")
