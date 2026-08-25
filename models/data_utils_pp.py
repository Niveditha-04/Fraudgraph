"""
Phase 3.5: prep for the Elliptic++ wallet-address graph, mirroring
models/data_utils.py's design decisions (train-only scaler fit, undirected
edges for the GNNs, train-only class weights) so the two phases are a fair
apples-to-apples comparison, not confounded by different preprocessing.
"""
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_undirected

from data.prepare_elliptic_pp import validate_and_cache
from models.data_utils import PreparedData


def prepare_pp_data() -> PreparedData:
    data = validate_and_cache()

    x_raw = data.x.clone()
    scaler = StandardScaler()
    scaler.fit(x_raw[data.train_mask].numpy())
    x = torch.from_numpy(scaler.transform(x_raw.numpy())).float()

    edge_index_undirected = to_undirected(data.edge_index)

    y_train = data.y[data.train_mask]
    n_train = y_train.numel()
    n_train_licit = int((y_train == 0).sum())
    n_train_illicit = int((y_train == 1).sum())
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
