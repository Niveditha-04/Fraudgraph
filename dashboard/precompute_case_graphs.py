"""
Phase 7: pre-render a small Pyvis subgraph HTML for each of the 10 cached
Phase 6 demo cases, showing the flagged wallet and its immediate neighbors
(colored by ground-truth label where known) -- this is what makes the
"click a node, see the subgraph" dashboard requirement real rather than a
placeholder. Capped neighbor count so a rare high-degree wallet doesn't
produce an unreadable hairball.
"""
import json
import os

from pyvis.network import Network

from data.prepare_elliptic_pp import validate_and_cache

MAX_NEIGHBORS = 25
LABEL_COLOR = {0: "#4caf50", 1: "#e53935", 2: "#9e9e9e"}  # licit=green, illicit=red, unknown=gray
LABEL_NAME = {0: "licit", 1: "illicit", 2: "unknown"}

OUT_DIR = "dashboard/static/graphs"


def build_case_graph_html(node_idx: int, edge_index, y, case_id: int) -> str:
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    mask = (src == node_idx) | (dst == node_idx)
    neighbor_edges = list(zip(src[mask], dst[mask]))
    neighbors = set()
    for a, b in neighbor_edges:
        neighbors.add(int(a))
        neighbors.add(int(b))
    neighbors.discard(node_idx)
    neighbors = list(neighbors)[:MAX_NEIGHBORS]

    net = Network(
        height="420px", width="100%", bgcolor="#0d1117", font_color="#e6edf3", directed=False,
        cdn_resources="in_line",  # single self-contained HTML file, no separate lib/ folder to deploy
    )
    net.add_node(
        node_idx, label=f"CASE #{node_idx}", color="#ffb300", size=32,
        title=f"Flagged wallet (case {case_id}) -- ground truth: {LABEL_NAME[int(y[node_idx])]}",
        borderWidth=3,
    )
    for n in neighbors:
        net.add_node(
            n, label=str(n), color=LABEL_COLOR[int(y[n])], size=14,
            title=f"wallet {n} -- label: {LABEL_NAME[int(y[n])]}",
        )
    for a, b in neighbor_edges:
        a, b = int(a), int(b)
        if a in neighbors or a == node_idx:
            if b in neighbors or b == node_idx:
                net.add_edge(a, b, color="#484f58")

    net.set_options('{"physics": {"solver": "forceAtlas2Based", "stabilization": {"iterations": 100}}}')
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"case_{case_id}.html")
    net.write_html(out_path, notebook=False, open_browser=False)
    return out_path


def main():
    raw_data = validate_and_cache()
    results = json.load(open("agent/results/phase6_results.json"))

    for r in results:
        path = build_case_graph_html(r["node_idx"], raw_data.edge_index, raw_data.y, r["case_id"])
        print(f"case {r['case_id']} (node {r['node_idx']}): saved {path}")


if __name__ == "__main__":
    main()
