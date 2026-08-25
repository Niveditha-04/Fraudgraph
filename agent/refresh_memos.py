"""
One-off fix: regenerate only the memo text for the 10 cached Phase 6 cases
using the corrected MEMO_MODEL (claude-sonnet-5, not the older
claude-sonnet-4-5-20250929 snapshot the first runs used). Reuses the
existing persona verdicts/consensus/typology chunks -- those are model-version
independent and already correct -- rather than re-running the whole panel
and paying for 30 more persona calls that would produce the same result.
"""
import json

from dotenv import load_dotenv

load_dotenv()

from agent.graph import draft_memo_node, rag_lookup_node

RESULTS_PATH = "agent/results/phase6_results.json"


def main():
    with open(RESULTS_PATH) as f:
        cases = json.load(f)

    for c in cases:
        state = {
            "case_id": c["case_id"],
            "evidence_summary": c["evidence_summary"],
            "persona_results": c["persona_results"],
            "consensus_verdict": c["consensus_verdict"],
            "human_decision": c["consensus_verdict"] if c["triggered_human_review"] else None,
        }
        # typology_chunks wasn't persisted in the original results file --
        # regenerate it (cheap: local embedding + vector search, no LLM
        # call) rather than treating this as a full agent re-run.
        rag_result = rag_lookup_node(state)
        state.update(rag_result)

        result = draft_memo_node(state)
        c["memo"] = result["memo"]
        print(f"case {c['case_id']}: memo refreshed ({len(result['memo'])} chars)")

    with open(RESULTS_PATH, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"\nsaved updated memos to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
