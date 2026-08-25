"""
Shared data prep for Phase 3 model training.

Design decisions made here, documented because they are not neutral defaults:

1. Feature standardization is fit on TRAIN nodes only (t 1-34), then applied
   to val/test. Fitting on the full dataset would leak future feature
   distribution information into the "past" the model trains on -- the same
   temporal-integrity concern that motivates the time-based split at all.

2. Edges are converted to undirected for the GNNs (both directions added).
   The raw Elliptic graph is directed by payment flow (src pays dst). Under
   PyG's message-passing convention, a directed edge (src, dst) only lets
   dst aggregate a message FROM src -- so a node that mostly sends money
   (frequently a src, rarely a dst) would get almost no aggregated
   neighborhood signal. Classic laundering "layering" patterns (fan-in then
   fan-out through an intermediary) require seeing both incoming and
   outgoing neighbors to detect. This matches how the original Elliptic GCN
   paper (Weber et al. 2019) treats the graph. The Logistic Regression
   baseline does not use edges at all, so this choice cannot bias the
   GNN-vs-baseline comparison in the GNN's favor by construction -- it's a
   modeling choice for the GNNs' own inputs, not the eval methodology.

3. Class weights for the loss are computed from the TRAIN split's labeled
   distribution specifically, not the labeled-nodes-overall distribution
   (9.76% illicit) and not the all-nodes distribution (~2.2% illicit) --
   because the model only ever sees train-set label frequency during
   optimization.
"""
from dataclasses import dataclass

import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_undirected

from data.validate_phase1 import validate


@dataclass
class PreparedData:
    x: torch.Tensor
    x_raw: torch.Tensor
    y: torch.Tensor
    edge_index_directed: torch.Tensor
    edge_index_undirected: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    class_weights: torch.Tensor  # [w_licit, w_illicit], indexed by class id


def prepare_data() -> PreparedData:
    data = validate()  # runs the Phase 1 gate as a precondition, not just a helper call

    x_raw = data.x.clone()
    scaler = StandardScaler()
    scaler.fit(x_raw[data.train_mask].numpy())
    x = torch.from_numpy(scaler.transform(x_raw.numpy())).float()

    edge_index_undirected = to_undirected(data.edge_index)

    y_train = data.y[data.train_mask]
    n_train = y_train.numel()
    n_train_licit = int((y_train == 0).sum())
    n_train_illicit = int((y_train == 1).sum())
    # standard inverse-frequency "balanced" weighting: w_c = n / (n_classes * n_c)
    w_licit = n_train / (2 * n_train_licit)
    w_illicit = n_train / (2 * n_train_illicit)
    class_weights = torch.tensor([w_licit, w_illicit], dtype=torch.float)

    print(
        f"train class weights -> licit: {w_licit:.3f}, illicit: {w_illicit:.3f} "
        f"(train set: {n_train_licit} licit / {n_train_illicit} illicit, "
        f"{n_train_illicit / n_train:.2%} illicit)"
    )

    return PreparedData(
        x=x,
        x_raw=x_raw,
        y=data.y,
        edge_index_directed=data.edge_index,
        edge_index_undirected=edge_index_undirected,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        class_weights=class_weights,
    )
