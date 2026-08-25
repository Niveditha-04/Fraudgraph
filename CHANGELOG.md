# Changelog

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
