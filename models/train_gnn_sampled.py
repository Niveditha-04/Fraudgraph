"""
Phase 3.5: neighbor-sampled mini-batch training/eval harness.

Full-batch training (models/train_gnn.py) keeps the whole graph's node
features and every layer's activations in memory at once. That's fine at
base Elliptic's scale (203,769 nodes) but breaks down on the Elliptic++
wallet graph (822,942 nodes, 5.5M undirected edges): a real attempt at
full-batch training here was observed thrashing -- CPU utilization dropped
to ~18% of wall-clock time (most of the "work" was actually the OS paging
memory in and out, not the model computing), and killing it freed ~5.6GB of
memory that had been pinned by that one process. This is the standard point
in real GNN pipelines where full-batch training is replaced by neighbor
sampling (PyG's NeighborLoader) -- exactly what production systems do at
this scale, not an ad hoc workaround.

Mirrors train_gnn.py's methodology so the two are a fair comparison: same
weighted cross-entropy loss, same early stopping on validation AUC-PR (not
loss, not accuracy), same model classes (GraphSAGE/GAT take plain
(x, edge_index) and work unchanged on a NeighborLoader mini-batch's local
subgraph).
"""
import copy
import time

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from models.data_utils import PreparedData
from models.metrics import Metrics, compute_metrics

NUM_NEIGHBORS = [15, 10]  # fanout per layer, matches the original GraphSAGE paper's defaults


def _build_loader(pdata: PreparedData, mask: torch.Tensor, batch_size: int, shuffle: bool) -> NeighborLoader:
    data = Data(x=pdata.x, edge_index=pdata.edge_index_undirected, y=pdata.y)
    return NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=mask,
        batch_size=batch_size,
        shuffle=shuffle,
    )


@torch.no_grad()
def _evaluate(model: torch.nn.Module, loader: NeighborLoader, device: str) -> Metrics:
    model.eval()
    all_probs, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        seed_out = out[: batch.batch_size]
        seed_y = batch.y[: batch.batch_size]
        probs = F.softmax(seed_out, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_y.append(seed_y.cpu().numpy())
    import numpy as np

    return compute_metrics(np.concatenate(all_y), np.concatenate(all_probs))


def train_gnn_sampled(
    model: torch.nn.Module,
    pdata: PreparedData,
    epochs: int = 30,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 5,
    train_batch_size: int = 1024,
    eval_batch_size: int = 4096,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[Metrics, Metrics, dict]:
    model = model.to(device)
    class_weights = pdata.class_weights.to(device)

    train_loader = _build_loader(pdata, pdata.train_mask, train_batch_size, shuffle=True)
    val_loader = _build_loader(pdata, pdata.val_mask, eval_batch_size, shuffle=False)
    test_loader = _build_loader(pdata, pdata.test_mask, eval_batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc_pr = -1.0
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    history = {"train_loss": [], "val_auc_pr": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            seed_out = out[: batch.batch_size]
            seed_y = batch.y[: batch.batch_size]
            loss = F.cross_entropy(seed_out, seed_y, weight=class_weights)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        val_metrics = _evaluate(model, val_loader, device)

        history["train_loss"].append(avg_train_loss)
        history["val_auc_pr"].append(val_metrics.auc_pr)

        if val_metrics.auc_pr > best_val_auc_pr:
            best_val_auc_pr = val_metrics.auc_pr
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if verbose:
            print(
                f"  epoch {epoch:4d}  train_loss={avg_train_loss:.4f}  "
                f"val_auc_pr={val_metrics.auc_pr:.4f}  (best={best_val_auc_pr:.4f} @ {best_epoch})  "
                f"[{time.time() - t0:.0f}s elapsed]"
            )

        if epochs_since_improve >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    elapsed = time.time() - t0
    if verbose:
        print(f"  training took {elapsed:.1f}s, best epoch {best_epoch}, best val AUC-PR {best_val_auc_pr:.4f}")

    model.load_state_dict(best_state)
    val_metrics = _evaluate(model, val_loader, device)
    test_metrics = _evaluate(model, test_loader, device)

    return val_metrics, test_metrics, history
