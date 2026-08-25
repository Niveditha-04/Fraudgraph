"""
Phase 2: graph construction + EDA.

Gate: printed class imbalance must show illicit ~2% of labeled nodes.
NOTE: this figure in the brief is imprecise -- ~2% is illicit's share of ALL
203,769 nodes. Within LABELED nodes only (the ~46,564 that aren't 'unknown'),
illicit is ~9.76%, not ~2%. Both numbers are reported below; the labeled-only
ratio (9.76% illicit / 90.24% licit) is the one Phase 3's weighted
cross-entropy loss must be calibrated against, since unlabeled nodes never
enter the loss.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data.validate_phase1 import validate


def report_class_distribution(data):
    n = data.num_nodes
    vals, counts = torch.unique(data.y, return_counts=True)
    label_counts = dict(zip(vals.tolist(), counts.tolist()))

    print("=== class distribution (all nodes) ===")
    for label, name in [(0, "licit"), (1, "illicit"), (2, "unknown")]:
        c = label_counts.get(label, 0)
        print(f"{name:10s}: {c:7d} ({c / n:5.1%})")

    labeled = data.y != 2
    n_labeled = int(labeled.sum())
    n_illicit = int((data.y == 1).sum())
    n_licit = int((data.y == 0).sum())
    illicit_frac = n_illicit / n_labeled

    print("\n=== class distribution (labeled nodes only) ===")
    print(f"labeled total: {n_labeled}")
    print(f"illicit: {n_illicit} ({illicit_frac:.2%})")
    print(f"licit:   {n_licit} ({n_licit / n_labeled:.2%})")

    print(
        f"\nGATE check: illicit is {n_illicit / n:.1%} of ALL nodes "
        f"(brief's '~2%' figure, confirmed) and {illicit_frac:.1%} of LABELED "
        f"nodes (the figure that actually matters for Phase 3 loss weighting)."
    )
    return illicit_frac


def plot_illicit_per_timestep(data, out_path="data/eda_outputs/illicit_per_timestep.png"):
    time_step = data.time_step
    y = data.y
    steps = list(range(1, 50))

    illicit_counts, licit_counts, unknown_counts = [], [], []
    for t in steps:
        yt = y[time_step == t]
        illicit_counts.append(int((yt == 1).sum()))
        licit_counts.append(int((yt == 0).sum()))
        unknown_counts.append(int((yt == 2).sum()))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].bar(steps, illicit_counts, color="crimson")
    axes[0].set_ylabel("illicit tx count")
    axes[0].set_title("Illicit transaction count per time step (Elliptic Bitcoin dataset)")
    axes[0].axvline(34.5, color="black", linestyle="--", linewidth=1, label="train/val boundary")
    axes[0].axvline(39.5, color="gray", linestyle="--", linewidth=1, label="val/test boundary")
    axes[0].legend(fontsize=8)

    axes[1].bar(steps, licit_counts, color="seagreen", label="licit")
    axes[1].bar(steps, unknown_counts, bottom=licit_counts, color="lightgray", label="unknown")
    axes[1].set_ylabel("tx count")
    axes[1].set_xlabel("time step")
    axes[1].set_title("Licit vs unknown transaction count per time step")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130)
    print(f"\nsaved plot to {out_path}")
    print(f"illicit count range across time steps: {min(illicit_counts)} - {max(illicit_counts)}")
    zero_steps = [t for t, c in zip(steps, illicit_counts) if c == 0]
    print(f"time steps with zero illicit labeled nodes: {zero_steps}")


if __name__ == "__main__":
    data = validate()
    print()
    illicit_frac = report_class_distribution(data)
    plot_illicit_per_timestep(data)
