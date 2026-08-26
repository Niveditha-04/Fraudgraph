"""
Fails the build if VALIDATION_REPORT.md's stated hybrid weights or Phase 6
case table drift from the actual source files -- this is regression
coverage for a real failure mode: a code/data change (e.g. the hybrid
formula change in models/hybrid_config.py) ran through the dashboard and
results file, but the prose in VALIDATION_REPORT.md wasn't regenerated to
match. The report's entire premise is "these numbers are read from files,
not asserted from memory" -- a document that claims that and is wrong is a
worse failure than a bug in the code, since it's the one place a reader
has no way to independently check without doing exactly this comparison
by hand.

Reads the report's stated values directly out of its prose (regex against
the literal `HYBRID_WEIGHT_GNN=X`/`HYBRID_WEIGHT_BENFORD=X` text and the
Phase 6 markdown table), not the other way around -- this test does not
tell the report what to say, it checks what the report already says
against ground truth.
"""
import json
import re

from models.hybrid_config import HYBRID_WEIGHT_BENFORD, HYBRID_WEIGHT_GNN

REPORT_PATH = "VALIDATION_REPORT.md"
PHASE6_RESULTS_PATH = "agent/results/phase6_results.json"


def _read_report() -> str:
    with open(REPORT_PATH) as f:
        return f.read()


def test_report_states_the_current_hybrid_weights():
    report = _read_report()
    match = re.search(r"HYBRID_WEIGHT_GNN=([\d.]+).*?HYBRID_WEIGHT_BENFORD=([\d.]+)", report, re.DOTALL)
    assert match, (
        "VALIDATION_REPORT.md no longer states the hybrid weights in the "
        "expected `HYBRID_WEIGHT_GNN=X`, `HYBRID_WEIGHT_BENFORD=X` format -- "
        "either the wording changed (update this test's regex to match) or "
        "the weights were never (re-)documented after a formula change."
    )
    stated_gnn, stated_benford = float(match.group(1)), float(match.group(2))
    assert stated_gnn == HYBRID_WEIGHT_GNN, (
        f"VALIDATION_REPORT.md states HYBRID_WEIGHT_GNN={stated_gnn}, but "
        f"models/hybrid_config.py actually sets it to {HYBRID_WEIGHT_GNN} -- "
        "the report is stale relative to the active formula."
    )
    assert stated_benford == HYBRID_WEIGHT_BENFORD, (
        f"VALIDATION_REPORT.md states HYBRID_WEIGHT_BENFORD={stated_benford}, but "
        f"models/hybrid_config.py actually sets it to {HYBRID_WEIGHT_BENFORD} -- "
        "the report is stale relative to the active formula."
    )


def test_report_phase6_table_matches_committed_results():
    report = _read_report()
    with open(PHASE6_RESULTS_PATH) as f:
        results = {r["case_id"]: r for r in json.load(f)}

    # Phase 6 table rows look like: | 0 | 412186 | licit | 0.208 | ... |
    row_pattern = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(licit|illicit)\s*\|\s*([\d.]+)\s*\|", re.MULTILINE)
    rows = row_pattern.findall(report)
    assert rows, "No Phase 6 case table rows found in VALIDATION_REPORT.md -- table format may have changed."

    checked = set()
    for case_id_str, node_str, gt, hybrid_str in rows:
        case_id = int(case_id_str)
        if case_id not in results:
            continue  # not every numbered table in the doc is the Phase 6 table
        stated_node, stated_gt, stated_hybrid = int(node_str), gt, float(hybrid_str)
        actual = results[case_id]
        if stated_node != actual["node_idx"] or stated_gt != actual["ground_truth"]:
            continue  # a different table that happens to start with a matching case id

        checked.add(case_id)
        assert abs(stated_hybrid - actual["hybrid_score"]) < 0.01, (
            f"VALIDATION_REPORT.md's Phase 6 table states case {case_id} has hybrid score "
            f"{stated_hybrid}, but {PHASE6_RESULTS_PATH} has {actual['hybrid_score']:.4f} for "
            f"node {actual['node_idx']} -- the report's Phase 6 table is stale."
        )

    assert checked == set(results.keys()), (
        f"VALIDATION_REPORT.md's Phase 6 table doesn't account for all cases in "
        f"{PHASE6_RESULTS_PATH}. Found rows for cases {sorted(checked)}, expected "
        f"{sorted(results.keys())} -- the report's Phase 6 table is stale or incomplete."
    )
