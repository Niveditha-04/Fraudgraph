"""
Phase 7 (static deployment variant): export the 10 cached Phase 6 cases as
static JSON files, so the dashboard can run as a pure static site (no
backend) -- required for Hugging Face's free Space tier, which only hosts
static Spaces for free (Docker/Gradio require a PRO subscription, discovered
when the initial Docker Space creation returned HTTP 402).

The local FastAPI backend (api/main.py) still works standalone for anyone
running this repo themselves; this script produces an equivalent, static
snapshot of the same 10 cases for the deployed demo.
"""
import json
import os

RESULTS_PATH = "agent/results/phase6_results.json"
OUT_DIR = "dashboard/static/data"


def main():
    with open(RESULTS_PATH) as f:
        cases = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    summary = [
        {
            "case_id": c["case_id"],
            "node_idx": c["node_idx"],
            "ground_truth": c["ground_truth"],
            "hybrid_score": c["hybrid_score"],
            "status": c["status"],
            "consensus_verdict": c["consensus_verdict"],
            "triggered_human_review": c["triggered_human_review"],
        }
        for c in cases
    ]
    with open(os.path.join(OUT_DIR, "cases.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for c in cases:
        with open(os.path.join(OUT_DIR, f"case_{c['case_id']}.json"), "w") as f:
            json.dump(c, f, indent=2)

    print(f"exported {len(cases)} cases to {OUT_DIR}/ (cases.json + case_N.json)")


if __name__ == "__main__":
    main()
