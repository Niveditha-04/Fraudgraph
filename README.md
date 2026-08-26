# FraudGraph

[![Tests](https://github.com/Niveditha-04/fraudgraph/actions/workflows/tests.yml/badge.svg)](https://github.com/Niveditha-04/fraudgraph/actions/workflows/tests.yml)

**Explore 10 investigated cases (static, pre-computed — not a live scoring endpoint):** https://niv04-fraudgraph.static.hf.space/

A two-layer fraud detection system on the Elliptic Bitcoin transaction graph: a Graph Neural Network that spots suspicious wallet neighborhoods, and an LLM-based investigation agent that reviews each flagged case through a 3-persona panel before drafting a memo — routing to human review whenever the panel disagrees.

Gate-by-gate results: [VALIDATION_REPORT.md](VALIDATION_REPORT.md). Dataset provenance: [DATASET.md](DATASET.md).

![Dashboard screenshot](assets/dashboard_screenshot.png)

## The problem

Money laundering and coordinated fraud rings move money across *networks* of accounts specifically so no single transaction looks suspicious on its own. Row-by-row transaction monitoring can't see this — the pattern only exists at the network level. Graph-based AML analytics is established industry practice (FATF calls for it explicitly); the newer part is combining it with an LLM investigation layer on top, a combination first published (as "FLAG") at KDD 2025.

## Architecture

```mermaid
flowchart TB
    subgraph Detection["1 · Detection (established practice)"]
        A[Elliptic Bitcoin graph<br/>203K-823K nodes] --> B[GraphSAGE / GAT<br/>temporal train/val/test split]
        B --> C[Illicit-probability score]
    end
    subgraph Hybrid["2 · Hybrid score"]
        C --> D["Hybrid score = GNN score<br/><i>Benford weight = 0: confirmed non-predictive</i>"]
        E[Benford's Law deviation<br/>on real BTC amounts] --> D
    end
    subgraph Investigation["3 · Agentic investigation (research-frontier)"]
        D --> F["LangGraph: retrieve case<br/><i>current: aggregate wallet stats,<br/>not subgraph structure</i>"]
        F --> G[3-persona panel<br/>AML analyst · compliance officer · skeptic]
        G --> H{Unanimous?}
        H -->|yes| I[RAG lookup<br/>FATF/FinCEN typology docs]
        H -->|no| J[interrupt → human review]
        J --> I
        I --> K[Draft memo, citing sources]
    end
    K --> L[Dashboard]
```

The detection layer (GNN on a transaction graph) is standard industry practice. The investigation layer (LLM persona panel with human-in-the-loop routing on disagreement) is the newer, less-established half.

## Results

Metrics are precision / recall / AUC-PR, never accuracy — illicit is 2-10% of the data depending on denominator, so a model predicting "licit" for everything scores >90% accuracy while catching zero fraud. All splits are temporal (train on early time steps, test on later ones); a random split leaks future information into training.

**Base Elliptic (203,769 nodes):**

| Model | Precision | Recall | AUC-PR |
|---|---|---|---|
| Logistic Regression (no graph) | 0.150 | 0.835 | 0.220 |
| GraphSAGE | 0.673 | 0.539 | **0.542** |
| GAT | 0.115 | 0.753 | 0.372 |

![Model comparison bar chart](assets/model_comparison.png)

The precision/recall pairs above are at the default 0.5 classification threshold, and GAT's low precision there is partly a threshold artifact, not purely a model-quality gap: at each model's own F1-maximizing threshold (selected on the validation set, applied to test), GraphSAGE improves modestly (precision 0.673→0.784, F1 0.599→0.608 at threshold 0.83) while GAT improves sharply (precision 0.115→0.552, F1 0.199→0.402 at threshold 0.93). GraphSAGE still wins on AUC-PR (threshold-independent: 0.542 vs 0.372) and still has the better F1 even at GAT's best threshold, so the ranking doesn't change — but GAT is a substantially more usable model than its default-threshold numbers suggest. See `models/threshold_analysis.py`.

**Elliptic++ (822,942 wallet nodes, optional scale extension):**

| Model | Precision | Recall | AUC-PR |
|---|---|---|---|
| Logistic Regression (no graph) | 0.064 | 0.849 | 0.126 |
| GraphSAGE | 0.096 | 0.978 | **0.428** — undertrained, see Known limitations |
| GAT | 0.154 | 0.886 | 0.362 |

GraphSAGE beats the non-graph baseline by ~2.5x on both datasets — the graph structure carries real signal.

**Benford's Law correlation:** r = 0.042 (p = 2×10⁻²⁸) between Benford deviation and the GNN score, on 70,487 test nodes — statistically significant at this sample size, practically negligible.

**Hybrid score, evaluated as a combined classifier — this is what led to changing the formula:**

| Score | Precision | Recall | AUC-PR |
|---|---|---|---|
| GNN alone | 0.096 | 0.978 | **0.428** |
| Benford alone | 0.038 | 0.773 | 0.038 |
| Original hybrid (0.7×GNN + 0.3×Benford) | 0.080 | 0.977 | 0.232 |

The original 0.7/0.3 hybrid score underperformed the GNN alone. The "Benford alone" row is Benford's own precision/recall/AUC-PR against the ground-truth label, not just its correlation with the GNN score — illicit is 4.49% of this test set, and Benford's standalone AUC-PR (0.038) is below that base rate. Benford's Law deviation is a confirmed non-predictive signal here, not merely an untested one; giving it 30% weight actively dragged the hybrid score below the GNN alone. See `models/evaluate_hybrid_score.py`.

**The hybrid formula was changed as a result: `models/hybrid_config.py` now sets the weighting to 100% GNN / 0% Benford** — the "hybrid" score is currently the GNN score alone. Benford's computation and its rejected-weighting evaluation stay in the codebase as a documented negative result, not deleted. All case data, scores, and dashboard content below reflect the corrected formula.

**Investigation agent:** run on 10 test cases (3 illicit / 7 licit ground truth) — 6 of 10 triggered human review on panel disagreement. No case reached unanimous "illicit" consensus. See Known limitations.

![Hybrid score by case, colored by ground truth](assets/hybrid_score_by_case.png)

A well-calibrated score would show illicit (orange) cases clustered high and licit (blue) cases clustered low. The overlap here is real: case 2 is a confirmed-illicit wallet with a GNN/hybrid score of exactly 0.000, and the panel unanimously classified it licit at 0.92-0.95 confidence — see VALIDATION_REPORT.md for detail.

## Repository layout

```
data/     Elliptic / Elliptic++ loading, validation gates, EDA
models/   GraphSAGE, GAT, baseline, Benford's Law, hybrid score
rag/      AML typology PDFs → Chroma vector store
agent/    LangGraph investigation agent (persona panel, human review, memo)
api/      FastAPI backend (local/live use)
dashboard/ Static dashboard (deployed) + Pyvis subgraph rendering
tests/    pytest suite (runs in CI on push/PR/weekly)
```

## Running locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# Phase 1-2: data validation + EDA
python -m data.eda_phase2

# Phase 3: train GraphSAGE/GAT vs. baseline on base Elliptic
python -m models.run_phase3

# Phase 5: build the RAG vector store (needs the PDFs in rag/documents/)
python -m rag.build_vectorstore

# Phase 6: run the investigation agent (needs ANTHROPIC_API_KEY in .env;
# optionally LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
# for LangSmith tracing)
python -m agent.run_phase6

# Phase 7: local dashboard with a live FastAPI backend
uvicorn api.main:app --reload
```

## Tooling

PyTorch + PyTorch Geometric (GraphSAGE/GAT), LangChain / LangGraph (`create_agent`, `StateGraph`, `interrupt()`-based human-in-the-loop), Chroma via `langchain-chroma`, local HuggingFace embeddings for RAG, FastAPI, Pyvis.

## Known limitations

- The "hybrid" score is currently the GNN score alone (Benford weighted at 0, see Results) — there is no second predictive signal complementing the GNN right now. A genuinely predictive second signal (e.g. transaction burstiness, in/out-degree ratio, wallet age) would need to replace Benford before the hybrid architecture adds real value over the GNN by itself.
- GAT's default-threshold precision (0.115) looks unusable but is substantially a threshold artifact — at its own F1-optimal threshold it reaches 0.552 precision (see Results). GraphSAGE still outperforms it at every threshold (higher AUC-PR, better F1), but "not viable" overstates the gap.
- The Elliptic++ GraphSAGE run is undertrained: capped at 25 epochs by a memory constraint while validation AUC-PR was still climbing (0.36→0.71 in its last 5 logged epochs). Its 0.428 AUC-PR is a floor, not a converged result.
- Thresholds are F1-optimal (see Results), not chosen against a deployment false-positive budget — a real deployment picks a threshold based on how many flags a compliance team can actually review per day, which may differ from the F1-maximizing point.
- The investigation agent has not demonstrated it can confirm fraud unanimously on its own — across 10 test cases, verdicts were either unanimous "licit" or a disagreement routed to human review, never unanimous "illicit." The panel is given aggregate wallet statistics, not real subgraph/neighbor context, which is the likely cause.
- n=10 test cases is too small to draw statistical conclusions about panel reliability — it demonstrates the human-review mechanism triggers, not that the panel's judgment holds up at scale.
- No prompt-injection or adversarial-input handling. Retrieved RAG text and case evidence flow into LLM prompts unsanitized — untested against manipulated input.

## What this project is not claiming

The GNN fraud-detection method is standard industry practice, not novel — graph-based AML analytics is explicitly called for by FATF and used in real enforcement (e.g. DOJ's Operation Gold Rush). The agentic investigation layer stacked on top of it is the newer part.
