# Changelog

## 2026-08-26 (5)

- Changed the hybrid score formula: `models/hybrid_config.py` (new, single source of truth, replacing three separately-hardcoded copies) sets `HYBRID_WEIGHT_GNN=1.0`, `HYBRID_WEIGHT_BENFORD=0.0`. The hybrid score is now the GNN score alone, since Benford's Law deviation was confirmed non-predictive of the ground-truth label on its own (standalone AUC-PR below the base rate). `models/evaluate_hybrid_score.py` still hardcodes the original 0.7/0.3 weighting deliberately, to document what that weighting did.
- Re-ran Phase 4 and the full Phase 6 pipeline against the corrected formula. Case selection changed (9 of 10 wallets are different from the previous set; only node 286169 is shared) since it's stratified by hybrid score. Gate still passes: 6/10 human review, 3 illicit/7 licit ground truth.
- The new case 2 (wallet #565660) replaces the old case 2 as the named "confident miss" example: a confirmed-illicit wallet with a one-hop pass-through pattern (received ~0.0234 BTC, sent ~0.0234 BTC back out), GNN/hybrid score exactly 0.000, panel unanimous at 0.92-0.95 confidence it's licit.
- Regenerated all dashboard subgraphs, static case data, and both report charts; redeployed to the live HF Space.
- Updated the measured LLM cost total: 102 traced calls, $0.7948 (was $0.1961 after the first traced run).

## 2026-08-26 (4)

- Added `models/threshold_analysis.py`. GAT's default-threshold precision (0.115) was largely a threshold artifact: at its own F1-optimal threshold (selected on the validation set), precision reaches 0.552 and F1 more than doubles (0.199→0.402). Corrected the "GAT not viable" framing in README/VALIDATION_REPORT to reflect this — GraphSAGE still wins on AUC-PR and F1 at every threshold, but the gap is smaller than the default-threshold numbers implied.
- Added base-rate context to the existing Benford standalone validation (`models/evaluate_hybrid_score.py`, run in an earlier session): illicit is 4.49% of the test set; Benford's standalone AUC-PR (0.038) is below that base rate, meaning it predicts the label at or below chance level on its own. This analysis already existed before this entry — only the base-rate framing is new.
- Added `LICENSE` (MIT).
- Added a note to the architecture diagram's investigation step: personas currently receive aggregate wallet stats, not subgraph structure.

## 2026-08-26 (3)

- Added `models/generate_report_charts.py`, generating `assets/model_comparison.png` and `assets/hybrid_score_by_case.png` from the committed result files. Added `assets/dashboard_screenshot.png` (headless Chrome capture of the live deployment). All three embedded in README.md and VALIDATION_REPORT.md.
- Replaced the "distribution drift" hedge for the Phase 3 val/test AUC-PR gap with the specific cause: per-timestep illicit counts in the test window range from 2 (t46) to 239 (t42), a ~120x swing, so aggregate AUC-PR mixes near-empty and dense timesteps.
- Added a named example to Phase 6's results: case 2 (wallet #572167), a confirmed-illicit wallet the panel unanimously classified as licit at 0.88-0.92 confidence — a concrete instance of the GNN's 0.428 AUC-PR producing a false negative.
- Quantified the "unknown outnumbers licit" claim in Phase 2: ratio ranges 1.4x-8.0x across all 49 time steps, averaging 4.3x.

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
