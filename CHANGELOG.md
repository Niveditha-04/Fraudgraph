# Changelog

## 2026-08-26 (2)

- Enabled LangSmith tracing (`LANGCHAIN_TRACING_V2`, project `Fraudgraph`). Replaced the estimated LLM token/cost figure in VALIDATION_REPORT.md with measured numbers from a traced Phase 6 run.
- LangSmith tracing initially failed with 403 Forbidden on every request — the API key was a Service Key (`lsv2_sk_...`), which requires an `X-Tenant-Id`/workspace-ID header that isn't needed for Personal Access Tokens. Fixed by setting `LANGCHAIN_WORKSPACE_ID`.
- Found via LangSmith trace data that the memo-drafting call's `max_tokens=800` was too low — every call hit the cap and every saved memo was truncated mid-sentence. Raised to 1500, then to 2500 for two memos still truncated at 1500. All 10 memos now end on a complete sentence.
- Attempted to add Ragas for automated RAG evaluation. Both ragas 0.4.3 and 0.3.9 fail to import in this environment — ragas unconditionally imports `langchain_community.chat_models.vertexai`, which doesn't exist in the current `langchain-community` (upstream bug, ragas issues #2745/#2753). Not integrated; Phase 5's RAG check remains 5 manually-read queries.

## 2026-08-26

- Added `models/evaluate_hybrid_score.py`. The hybrid score (0.7×GNN + 0.3×Benford) had only ever been assessed through its two components separately; direct evaluation shows it underperforms the GNN alone (AUC-PR 0.232 vs 0.428). Documented in README and VALIDATION_REPORT.
- Relabeled the deployed demo link from "Live demo" to "Explore 10 investigated cases" — it serves pre-computed cases, not a live scoring endpoint.
- Added a "not viable at this precision" note to GAT's row in results tables.
- Documented the prompt-injection/adversarial-input gap as an unaddressed limitation.
- Added `DATASET.md`.

## 2026-08-25

- Fixed Phase 6's memo-drafting model: `claude-sonnet-4-5-20250929` → `claude-sonnet-5`. All 10 committed memos regenerated (`agent/refresh_memos.py`).
- Removed the `temperature` parameter from the memo-drafting call — `claude-sonnet-5` rejects it.
- Initial build: Phases 0-8 complete, plus the optional Phase 3.5 (Elliptic++) scale extension. Deployed to Hugging Face Spaces and pushed to GitHub.
