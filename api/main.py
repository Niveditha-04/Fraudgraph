"""
Phase 7: thin FastAPI backend serving the 10 cached Phase 6 investigation
cases -- case lookup, hybrid score, persona verdicts, and the final memo.

Deliberately serves precomputed/cached results (agent/results/phase6_results.json)
rather than calling the LLM live on every request: per the brief, the
deployed demo must work "even without live API credits," and re-running a
persona panel + memo draft on every page load would also be a real latency/
cost problem for a public demo. A "live investigate" endpoint could call the
real LangGraph agent (agent/graph.py) for a NEW case in a follow-up
iteration, but the cached-case browsing experience is the actual Phase 7
deliverable.
"""
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

RESULTS_PATH = "agent/results/phase6_results.json"
GRAPHS_DIR = "dashboard/static/graphs"

app = FastAPI(title="FraudGraph API", description="GNN + agentic AML investigation demo")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _load_cases() -> list[dict]:
    if not os.path.exists(RESULTS_PATH):
        return []
    with open(RESULTS_PATH) as f:
        return json.load(f)


@app.get("/api/cases")
def list_cases():
    cases = _load_cases()
    return [
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


@app.get("/api/cases/{case_id}")
def get_case(case_id: int):
    cases = _load_cases()
    for c in cases:
        if c["case_id"] == case_id:
            return c
    raise HTTPException(status_code=404, detail=f"case {case_id} not found")


@app.get("/api/health")
def health():
    return {"status": "ok", "cases_loaded": len(_load_cases())}


if os.path.isdir(GRAPHS_DIR):
    app.mount("/graphs", StaticFiles(directory=GRAPHS_DIR), name="graphs")

if os.path.isdir("dashboard/static"):
    app.mount("/", StaticFiles(directory="dashboard/static", html=True), name="dashboard")
