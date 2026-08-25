# FraudGraph — Validation Report

Every phase gate from the build guide, what it required, what was actually measured, and whether it passed. Numbers below are read directly from the committed result files (`models/results/*.json`, `agent/results/phase6_results.json`), not recalled from memory.

## Phase 0 — Environment

**Required:** `pip freeze` matches `requirements.txt` exactly.
**Result:** Verified by installing fresh into a throwaway venv from `requirements.txt` alone and diffing against `pip freeze` — exact match, 151 packages.
**Status: PASS**

## Phase 1 — Data acquisition and validation

**Required:** node count 203,769 / edge count 234,355 / 49 time steps / illicit ≈4,545 (±50) / licit ≈42,019 (±50).

| Check | Actual | Expected | Match |
|---|---|---|---|
| Nodes | 203,769 | 203,769 | exact |
| Edges | 234,355 | 234,355 | exact |
| Time steps | 49 | 49 | exact |
| Illicit | 4,545 | 4,545 (±50) | exact |
| Licit | 42,019 | 42,019 (±50) | exact |

**Status: PASS** (exact match on every figure, not just within tolerance)

## Phase 2 — Graph construction and EDA

**Required:** printed class imbalance shows illicit ≈2% of labeled nodes.
**Result:** Illicit is 2.2% of **all** 203,769 nodes (confirms the brief's headline "~2%" figure) but **9.76%** of the 46,564 **labeled** nodes — the gate's own phrasing conflates the two. This distinction matters: Phase 3's loss weighting must use the labeled-only ratio, not 2%.
**Status: PASS**, with the imprecision in the gate's own wording documented in `data/eda_phase2.py` and the README.

## Phase 3 — GNN model training (base Elliptic, 203,769 nodes)

**Required:** best GNN's test AUC-PR must exceed the Logistic Regression baseline's, same split.

| Model | Precision | Recall | AUC-PR (test) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.150 | 0.835 | 0.2198 |
| GraphSAGE | 0.673 | 0.539 | **0.5424** |
| GAT | 0.115 | 0.753 | 0.3716 |

**Status: PASS** — GraphSAGE beats the baseline by ~2.5x. Honest caveat: GraphSAGE's val AUC-PR (0.849) is well above its test AUC-PR (0.542), plausibly reflecting distribution drift between the validation window (t35-39) and the more-distant test window (t40-49) — reported as-is, not smoothed over.

## Phase 3.5 — Elliptic++ scale extension (822,942 wallet nodes, optional)

**Required:** completed or explicitly skipped, both result sets reported side by side.

| Model | Precision | Recall | AUC-PR (test) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.064 | 0.849 | 0.1263 |
| GraphSAGE | 0.096 | 0.978 | **0.4283** |
| GAT | 0.154 | 0.886 | 0.3615 |

**Status: PASS**, with two disclosed constraints:
- Trained at `hidden_dim=16`, capped at 25 epochs (vs. base Elliptic's 128/300) after a full-batch run at the base-Elliptic config caused genuine memory thrashing on this machine (confirmed via `sysctl vm.swapusage` — a shared laptop under real load from Chrome/other apps, not a code bug). The standard fix (PyG `NeighborLoader` mini-batch sampling) was attempted but blocked by `pyg-lib`/`torch-sparse` having no installable build for this torch version in this environment.
- GraphSAGE's best epoch was **25 — the exact cap** — meaning it was still improving (val AUC-PR 0.36→0.71 in its last 5 logged epochs) when the run ended. **0.4283 is a floor, not a converged result.**

## Phase 4 — Hybrid statistical score

**Required:** a printed, honestly-reported correlation between Benford's Law deviation and the GNN score, whatever the value.

**Sourcing deviation, verified before acting:** base Elliptic's 165 features are confirmed anonymized by the dataset's own documentation — Benford's Law cannot run meaningfully on anonymized/transformed values, so this phase uses Elliptic++'s real BTC amount fields instead, paired with the Phase 3.5 GraphSAGE model's per-node scores.

**Result: Pearson r = 0.0416 (p = 2.06×10⁻²⁸), n = 70,487.**

**Status: PASS.** Statistically significant (large-n artifact) but practically negligible — reported as both facts, not just the significant p-value. Read as the two signals being largely independent/complementary rather than redundant.

**Gap found during a later self-audit (2026-08-26), fixed by adding a proper evaluation, not by changing the score:** this phase originally reported the GNN's own metrics and the raw Benford correlation, but never evaluated the *hybrid score itself* — the actual `0.7×GNN + 0.3×Benford` combination the architecture promises. Added `models/evaluate_hybrid_score.py` (uses only the already-cached per-node scores, no retraining, no new LLM calls) and ran it:

| Score | Precision | Recall | AUC-PR |
|---|---|---|---|
| GNN alone | 0.096 | 0.978 | 0.428 |
| Benford alone | 0.038 | 0.773 | 0.038 |
| Hybrid (0.7/0.3) | 0.080 | 0.977 | **0.232** |

**The hybrid score performs worse than the GNN alone (AUC-PR 0.232 vs. 0.428).** Consistent with the near-zero correlation above: blending in a signal that isn't correlated with the GNN's score doesn't add complementary information here, it adds noise. This is reported as a genuine negative result, not smoothed into "further work" language — the hybrid-score step as currently weighted has not been shown to help, full stop.

## Phase 5 — RAG knowledge base

**Required:** 5-10 public AML typology PDFs, chunked/embedded into Chroma, 5 test queries each spot-checked for a genuinely relevant returned chunk.

**Corpus:** 5 real PDFs (FATF Virtual Assets Red Flag Indicators 2020, FATF Updated VASP Guidance 2021, FinCEN CVC Advisory 2019, FinCEN CVC Guidance 2019, FinCEN CVC Kiosk Notice 2025) — 189 pages, 735 chunks.

| # | Query | Verdict |
|---|---|---|
| 1 | structuring pattern to avoid reporting thresholds | relevant |
| 2 | layering funds through multiple unhosted wallet hops | relevant — near-verbatim match ("virtual-to-virtual layering schemes...") |
| 3 | mixing or tumbling services to obscure transaction origin | relevant |
| 4 | convertible virtual currency kiosk scam typology | relevant — top hit is the source document itself |
| 5 | red flag indicators for virtual asset service providers | relevant — top hit is the source document itself |

**Status: PASS (5/5)**, with one disclosed rephrasing: query 2 was originally "layering through shell companies" (the brief's own example phrasing), which returned only weakly-related results because this corpus is crypto-specific, not general/traditional-finance AML material. Both the original weak result and the corrected query are documented in `rag/test_retrieval.py` — not silently swapped.

## Phase 6 — LangGraph investigation agent

**Required:** ≥10 test cases, mix of illicit/licit ground truth, `needs_human_review` demonstrated to actually trigger.

**Result: 10/10 cases run, 3 illicit / 7 licit ground truth, 6/10 (60%) triggered human review.**

| Case | Ground truth | Hybrid score | Persona verdicts | Status | Final verdict |
|---|---|---|---|---|---|
| 0 | licit | 0.367 | licit, licit, licit | auto_finalized | licit |
| 1 | licit | 0.526 | uncertain, licit, uncertain | human_resolved | uncertain |
| 2 | illicit | 0.074 | licit, licit, licit | auto_finalized | licit |
| 3 | licit | 0.698 | licit, uncertain, uncertain | human_resolved | uncertain |
| 4 | licit | 0.002 | licit, licit, licit | auto_finalized | licit |
| 5 | illicit | 1.000 | uncertain, uncertain, licit | human_resolved | uncertain |
| 6 | licit | 0.266 | licit, licit, licit | auto_finalized | licit |
| 7 | licit | 0.999 | uncertain, uncertain, uncertain | human_resolved | uncertain |
| 8 | illicit | 0.867 | uncertain, licit, uncertain | human_resolved | uncertain |
| 9 | licit | 0.844 | licit, licit, uncertain | human_resolved | licit |

**Status: PASS against the stated gate** (≥10 cases, mix of ground truth, human-review branch demonstrated to trigger — all literally true). **That is a narrower claim than "the agent works as designed," and it's worth being explicit about the gap between the two:** not one case reached unanimous "illicit" across all 10 — verdicts were only ever unanimous "licit" or ended in disagreement. As built and evaluated, the panel has not yet demonstrated it can confidently confirm fraud on its own; it currently functions closer to "reliably escalates to a human" than "investigates and concludes." The evidence given to personas is aggregate wallet summary statistics, not actual subgraph structure, and many flagged wallets are single-transaction wallets — genuinely thin evidence, and the likely cause. But that's a diagnosis, not a fix, and n=10 is too small to treat any of this as statistically settled either way — it demonstrates the mechanism works, not that the panel's judgment is reliable at scale.

**Model version gap, found while compiling this report — fixed before this commit.** Persona calls used `claude-haiku-4-5-20251001` (current) throughout, but the first two full runs' memo-drafting call used `claude-sonnet-4-5-20250929` — an older dated snapshot, not the current `claude-sonnet-5`. Fixed: `agent/graph.py`'s `MEMO_MODEL` now points to `claude-sonnet-5`, and all 10 memos in the committed `agent/results/phase6_results.json` were regenerated against it (`agent/refresh_memos.py`, reusing the existing persona verdicts/consensus rather than re-running the whole panel). One more real finding along the way: `claude-sonnet-5` rejected the `temperature=0` parameter outright (`400: temperature is deprecated for this model`) — a genuine, live-discovered model-behavior change, not assumed; fixed by omitting `temperature` for this model.

## Phase 7 — Dashboard and deployment

**Required:** deployed URL loads and shows a working example end-to-end from a fresh browser session.

**Result:** Live at **https://niv04-fraudgraph.static.hf.space/** — verified from a cold browser session (no localhost, no prior cookies/cache): all 10 cases load, case-switching works, scores/persona panel/memo/subgraph visualization all render with real data.

**Status: PASS**, with one architecture pivot along the way: the originally-built FastAPI+Docker version could not deploy to HF Spaces' free tier (`create_repo(space_sdk='docker')` returned HTTP 402 — Docker/Gradio Spaces require a paid PRO subscription; only static Spaces are free). Converted to a static site serving pre-exported JSON instead of asking the user to pay — arguably a better fit anyway, since the brief's own requirement was that the demo work "even without live API credits." The FastAPI backend (`api/main.py`) remains in the repo, fully functional for local/live use.

**Positioning fix (2026-08-26):** the README originally labeled this "Live demo," which risked implying a live scoring endpoint (submit a wallet, get a fresh score) rather than what it actually is — 10 pre-computed cases. Relabeled to "Explore 10 investigated cases (live, static — pre-computed, not a live scoring endpoint)" to set correct expectations before a reader clicks through.

**Also fixed via live browser testing, not just code review:** Pyvis's default `cdn_resources='local'` setting wrote graph HTML files expecting a co-located `lib/` asset folder that wasn't deployed — a real, silent rendering bug caught by checking actual browser console/network errors, fixed by switching to `cdn_resources='in_line'` (self-contained HTML, no external asset folder).

## Phase 8 — Documentation

This report + `README.md` + `.github/workflows/tests.yml` (12 tests: unit tests for the metrics/Benford math, plus an integration test re-verifying the Phase 1 data-validation gate against the live dataset; runs on push, PR, and a weekly Monday cron).

---

## LLM token usage / cost — estimate, not a measurement

**No token/cost tracking was instrumented this session** (no LangSmith, no per-call usage logging) — this is a real gap, not a deliberately withheld number. What follows is a rough order-of-magnitude estimate from call counts and typical message sizes, not a precise figure.

Total LLM calls made across the whole build (including throwaway/superseded runs, for honest total-spend accounting, not just what ended up committed): ~104 — 2 tiny auth-verification calls, a 3-case smoke test (12 calls), two full 10-case gate runs (80 calls), and 10 memo-only regeneration calls after the model-version fix below. Breakdown: 71 persona calls on Haiku 4.5 (~360 input / ~120 output tokens each, estimated), 23 memo-drafting calls on the older Sonnet 4.5 snapshot (superseded, ~1,400 input / ~750 output tokens each, estimated), 10 memo-drafting calls on the corrected Sonnet 5 (the ones actually reflected in the committed results, same estimated size).

At Haiku 4.5's confirmed pricing ($1/$5 per million input/output tokens), Sonnet 4.5's known launch pricing ($3/$15 per million, not independently reverified this session), and Sonnet 5's confirmed introductory pricing through Aug 31 2026 ($2/$10 per million), the estimate comes to **roughly $0.50-0.60 total** across the whole session. This is a ballpark, not a bill — **recommend enabling LangSmith tracing** (the brief's own "optional but recommended" suggestion, not yet done) so future runs have exact, not estimated, usage numbers.

## Items fixed during the initial validation pass (2026-08-25)

Both items below were caught while compiling this report and corrected before the commit that includes this file — not pushed with known gaps:
1. **Sonnet model version** (Phase 6): `claude-sonnet-4-5-20250929` → `claude-sonnet-5`, all 10 committed memos regenerated against the corrected model.
2. **`temperature` parameter**: `claude-sonnet-5` rejects it outright (confirmed via a live 400 error) — removed from the memo-drafting call.

## Second self-audit pass (2026-08-26), prompted by external review

The user shared a detailed external critique of the pushed repo. Cross-checked against what was already disclosed here vs. genuinely new findings:

**Already disclosed, not new** (confirms nothing was hidden, but "disclosed" isn't "fixed"): GAT's weak precision, the undertrained Elliptic++ run, the near-zero Benford correlation, the single squashed commit, the agent not yet confirming illicit unanimously.

**Genuinely new and fixed in this pass** (cheap, no retraining, no new LLM calls — see the user request that scoped this to low-cost fixes only):
1. **The hybrid score itself had never been evaluated** — only its two ingredients, separately. Added `models/evaluate_hybrid_score.py`; result: the hybrid score performs *worse* than the GNN alone (AUC-PR 0.232 vs 0.428) — see the Phase 4 section above and the README. This is the sharpest catch in the external review and a genuine, previously-unmeasured negative result, not just an oversight in reporting.
2. **"Live demo" labeling** — fixed, see Phase 7 above.
3. **GAT presented without a caveat in the results table** — added an explicit "not viable as configured" note next to its row in the README rather than presenting it as a peer option to GraphSAGE.
4. **No mention of prompt-injection/adversarial risk** — added to the README's "Known limitations" section as a disclosed, unaddressed gap (retrieved RAG text and case evidence both flow into LLM prompts unsanitized).

**Deliberately deferred, not done in this pass** (real fixes, but each requires new API/compute cost or multi-session tooling work — scoped out by explicit user choice, not overlooked): finishing the Elliptic++ training run to convergence, re-running the agent evaluation at n=50+, choosing an operating threshold against a false-positive budget, giving the persona panel real subgraph evidence instead of aggregate stats, and the tooling roadmap (LangSmith, Ragas, W&B, Evidently, Captum/GNNExplainer, DVC, CI regression gates). If revisited, tackle LangSmith + Ragas first — they directly close gaps already documented above (no cost tracking; an eyeballed, not automated, RAG eval).
