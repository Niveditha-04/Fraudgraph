"""
Phase 6 gate: run the full investigation graph on >=10 test cases (mix of
illicit/licit ground truth) and confirm the needs_human_review branch
actually triggers on at least one case -- not just exist unused in the code.

Human review simulation: for cases that hit `interrupt()`, this script plays
the role of the human reviewer with a simple, documented policy (majority
verdict among the panel if there is a 2-1 split, else "uncertain" is passed
through as the resolved verdict) so the full resume flow is exercised
end-to-end, not just the pause. A real deployment would surface the
interrupt payload to an actual human via the API layer (Phase 7) instead.
"""
import json
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command

from agent.case_data import build_test_cases
from agent.graph import build_graph


def simulate_human_decision(interrupt_value: dict) -> str:
    verdicts = [r["verdict"] for r in interrupt_value["persona_results"]]
    counts = {v: verdicts.count(v) for v in set(verdicts)}
    majority_verdict, majority_count = max(counts.items(), key=lambda kv: kv[1])
    if majority_count >= 2 and majority_verdict != "uncertain":
        return majority_verdict
    return "uncertain"


def run_case(app, case) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {
        "case_id": case.case_id,
        "evidence_summary": case.evidence_summary,
        "hybrid_score": case.hybrid_score,
    }
    result = app.invoke(initial_state, config=config)

    interrupted = "__interrupt__" in result
    if interrupted:
        interrupt_obj = result["__interrupt__"][0]
        human_decision = simulate_human_decision(interrupt_obj.value)
        result = app.invoke(Command(resume=human_decision), config=config)

    return {
        "case_id": case.case_id,
        "node_idx": case.node_idx,
        "ground_truth": case.ground_truth,
        "hybrid_score": case.hybrid_score,
        "gnn_score": case.gnn_score,
        "benford_score": case.benford_score,
        "evidence_summary": case.evidence_summary,
        "persona_results": result["persona_results"],
        "status": result["status"],
        "consensus_verdict": result["consensus_verdict"],
        "triggered_human_review": interrupted,
        "memo": result["memo"],
    }


def main(n_cases: int = 10):
    print(f"building {n_cases} test cases from real Phase 3.5/4 data ...")
    cases = build_test_cases(n_cases=n_cases)
    for c in cases:
        print(f"  case {c.case_id}: node={c.node_idx} ground_truth={c.ground_truth} hybrid_score={c.hybrid_score:.3f}")

    print("\nbuilding LangGraph investigation agent ...")
    app = build_graph()

    results = []
    for case in cases:
        print(f"\n{'=' * 70}\nrunning case {case.case_id} (ground truth: {case.ground_truth}) ...")
        r = run_case(app, case)
        print(f"  persona verdicts: {[p['verdict'] for p in r['persona_results']]}")
        print(f"  status: {r['status']}  triggered_human_review: {r['triggered_human_review']}")
        print(f"  final consensus verdict: {r['consensus_verdict']}")
        results.append(r)

    n_human_review = sum(r["triggered_human_review"] for r in results)
    n_illicit_gt = sum(r["ground_truth"] == "illicit" for r in results)
    n_licit_gt = sum(r["ground_truth"] == "licit" for r in results)

    print(f"\n{'=' * 70}\nPhase 6 gate summary\n{'=' * 70}")
    print(f"total cases run: {len(results)}")
    print(f"ground truth mix: {n_illicit_gt} illicit, {n_licit_gt} licit")
    print(f"cases that triggered needs_human_review: {n_human_review}")
    gate_pass = len(results) >= 10 and n_human_review >= 1 and n_illicit_gt >= 1 and n_licit_gt >= 1
    print(f"GATE (>=10 cases run, mix of illicit/licit ground truth, human_review triggered >=1x): "
          f"{'PASS' if gate_pass else 'FAIL'}")

    os.makedirs("agent/results", exist_ok=True)
    with open("agent/results/phase6_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results to agent/results/phase6_results.json")

    if not gate_pass:
        raise SystemExit(1)
    return results


if __name__ == "__main__":
    main()
