"""
Generates the two chart images embedded in README.md / VALIDATION_REPORT.md
from the committed result files -- reproducible from data on disk, not
hand-drawn or pasted from an external tool.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "assets"


def plot_model_comparison():
    with open("models/results/phase3_results.json") as f:
        r = json.load(f)

    models = ["Logistic Regression\n(baseline)", "GraphSAGE", "GAT"]
    metrics = ["precision", "recall", "auc_pr"]
    metric_labels = ["Precision", "Recall", "AUC-PR"]
    data = {
        "Logistic Regression\n(baseline)": r["baseline"],
        "GraphSAGE": r["graphsage"]["test"],
        "GAT": r["gat"]["test"],
    }

    x = range(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4c78a8", "#e45756", "#54a24b"]
    for i, model in enumerate(models):
        values = [data[model][m] for m in metrics]
        positions = [xi + (i - 1) * width for xi in x]
        ax.bar(positions, values, width, label=model.replace("\n", " "), color=colors[i])

    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on base Elliptic (temporal test split)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "model_comparison.png")
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")


def plot_hybrid_scores():
    with open("agent/results/phase6_results.json") as f:
        cases = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 5))
    for c in cases:
        color = "#e45756" if c["ground_truth"] == "illicit" else "#4c78a8"
        ax.scatter(c["case_id"], c["hybrid_score"], color=color, s=90, zorder=3)

    ax.set_xlabel("Case number")
    ax.set_ylabel("Hybrid risk score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(10))
    ax.set_title("Hybrid score vs. actual outcome, all 10 test cases")
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c78a8", markersize=10, label="Ground truth: licit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e45756", markersize=10, label="Ground truth: illicit"),
    ]
    ax.legend(handles=legend_elements)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "hybrid_score_by_case.png")
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_model_comparison()
    plot_hybrid_scores()
