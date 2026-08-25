"""
Phase 3 steps 3-4: shared training/eval harness for GraphSAGE and GAT.

Full-batch training (whole graph each step) -- 203,769 nodes x up-to-128 hidden
dims fits comfortably in memory, no need for neighbor sampling at this scale.
Early stopping on validation AUC-PR (not val loss, and not accuracy), since
AUC-PR on the illicit class is the metric that actually matters here and the
brief's Phase 3 gate is defined in terms of it.
"""
import copy
import time

import torch
import torch.nn.functional as F

from models.data_utils import PreparedData
from models.metrics import Metrics, compute_metrics


def train_gnn(
    model: torch.nn.Module,
    pdata: PreparedData,
    epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 30,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[Metrics, Metrics, dict]:
    model = model.to(device)
    x = pdata.x.to(device)
    edge_index = pdata.edge_index_undirected.to(device)
    y = pdata.y.to(device)
    train_mask = pdata.train_mask.to(device)
    val_mask = pdata.val_mask.to(device)
    test_mask = pdata.test_mask.to(device)
    class_weights = pdata.class_weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc_pr = -1.0
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    history = {"train_loss": [], "val_auc_pr": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            val_probs = F.softmax(out[val_mask], dim=1)[:, 1].cpu().numpy()
            val_y = y[val_mask].cpu().numpy()
            val_metrics = compute_metrics(val_y, val_probs)

        history["train_loss"].append(loss.item())
        history["val_auc_pr"].append(val_metrics.auc_pr)

        if val_metrics.auc_pr > best_val_auc_pr:
            best_val_auc_pr = val_metrics.auc_pr
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(
                f"  epoch {epoch:4d}  train_loss={loss.item():.4f}  "
                f"val_auc_pr={val_metrics.auc_pr:.4f}  (best={best_val_auc_pr:.4f} @ {best_epoch})"
            )

        if epochs_since_improve >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    elapsed = time.time() - t0
    if verbose:
        print(f"  training took {elapsed:.1f}s, best epoch {best_epoch}, best val AUC-PR {best_val_auc_pr:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        val_probs = F.softmax(out[val_mask], dim=1)[:, 1].cpu().numpy()
        test_probs = F.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()

    val_metrics = compute_metrics(y[val_mask].cpu().numpy(), val_probs)
    test_metrics = compute_metrics(y[test_mask].cpu().numpy(), test_probs)

    return val_metrics, test_metrics, history
