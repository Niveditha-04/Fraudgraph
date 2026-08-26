"""
Phase 6: LangGraph investigation agent.

Nodes, in order (per the project brief):
  1. retrieve         -- pull the case's evidence summary + hybrid score
  2. persona_panel     -- 3 LLM calls in parallel (AML analyst, compliance
                           officer, skeptic), each an independent structured verdict
  3. consensus_check   -- unanimous non-uncertain verdict -> proceed;
                           any disagreement -> interrupt() for human review
  4. rag_lookup        -- retrieve the matching AML typology chunk (Phase 5's
                           Chroma store) for the agreed-upon verdict
  5. draft_memo        -- final LLM call drafts the memo, citing the
                           retrieved typology source

Human-in-the-loop uses LangGraph's native `interrupt()`/`Command(resume=...)`
(langgraph.types) -- a first-class, checkpointer-backed mechanism, not a
hand-rolled retry loop or a bare "if disagree: skip" branch.
"""
import concurrent.futures
from typing import Literal, Optional, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from agent.personas import PERSONA_SYSTEM_PROMPTS, PersonaVerdict, build_persona_llm
from rag.build_vectorstore import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIR

MEMO_MODEL = "claude-sonnet-5"


class PersonaResult(TypedDict):
    persona: str
    verdict: str
    confidence: float
    reasoning: str


class InvestigationState(TypedDict, total=False):
    case_id: int
    evidence_summary: str
    hybrid_score: float
    persona_results: list[PersonaResult]
    consensus_verdict: Optional[str]  # "illicit" / "licit", set only on unanimous agreement
    status: str  # "auto_finalized" | "needs_human_review" | "human_resolved"
    human_decision: Optional[str]
    typology_chunks: list[str]
    memo: Optional[str]


def _call_persona(persona_key: str, evidence_summary: str) -> PersonaResult:
    llm = build_persona_llm(persona_key)
    system_prompt = PERSONA_SYSTEM_PROMPTS[persona_key]
    result: PersonaVerdict = llm.invoke(
        [
            ("system", system_prompt),
            ("human", f"Review this flagged case and give your verdict:\n\n{evidence_summary}"),
        ]
    )
    return PersonaResult(
        persona=persona_key, verdict=result.verdict, confidence=result.confidence, reasoning=result.reasoning
    )


def retrieve_node(state: InvestigationState) -> dict:
    # The case's evidence/hybrid score is already attached to the state when
    # the graph is invoked (agent/run_phase6.py builds it from real Phase 4
    # data) -- this node is a pass-through that exists to match the brief's
    # specified node ordering and would be where a live system fetches from
    # a database, rather than logic that needs to transform anything here.
    return {}


def persona_panel_node(state: InvestigationState) -> dict:
    personas = list(PERSONA_SYSTEM_PROMPTS.keys())
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(personas)) as executor:
        futures = {executor.submit(_call_persona, p, state["evidence_summary"]): p for p in personas}
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return {"persona_results": results}


def consensus_check_node(state: InvestigationState) -> dict:
    verdicts = {r["verdict"] for r in state["persona_results"]}
    if len(verdicts) == 1 and "uncertain" not in verdicts:
        return {"consensus_verdict": verdicts.pop(), "status": "auto_finalized"}
    return {"consensus_verdict": None, "status": "needs_human_review"}


def human_review_node(state: InvestigationState) -> dict:
    decision = interrupt(
        {
            "reason": "persona panel disagreed",
            "case_id": state["case_id"],
            "persona_results": state["persona_results"],
            "hybrid_score": state["hybrid_score"],
        }
    )
    return {"human_decision": decision, "consensus_verdict": decision, "status": "human_resolved"}


def rag_lookup_node(state: InvestigationState) -> dict:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=PERSIST_DIR)

    verdict = state["consensus_verdict"]
    query = (
        "red flag indicators of money laundering typical patterns for a flagged illicit case"
        if verdict == "illicit"
        else "legitimate high-volume wallet activity that resembles but is not money laundering"
    )
    docs = vectorstore.similarity_search(query, k=3)
    chunks = [f"[{d.metadata.get('source_document', '?')} p{d.metadata.get('page', '?')}] {d.page_content}" for d in docs]
    return {"typology_chunks": chunks}


def draft_memo_node(state: InvestigationState) -> dict:
    # claude-sonnet-5 has deprecated the `temperature` parameter entirely
    # (confirmed via a live 400 error, not assumed) -- omit it here rather
    # than pin a value the model no longer accepts.
    # max_tokens=800 was too low: LangSmith traces showed all 10 memo calls
    # hitting exactly 800 output tokens, and the saved memo text confirmed
    # every one was truncated mid-sentence. 1500 still wasn't enough for the
    # two longest memos (also confirmed truncated mid-sentence). Raised again
    # with more margin.
    llm = ChatAnthropic(model=MEMO_MODEL, max_tokens=2500)
    persona_summary = "\n".join(
        f"- {r['persona']}: {r['verdict']} (confidence {r['confidence']:.2f}) -- {r['reasoning']}"
        for r in state["persona_results"]
    )
    typology_text = "\n\n".join(state["typology_chunks"])
    human_note = f"\n\nNote: this verdict was set by human reviewer override after panel disagreement." if state.get("human_decision") else ""

    prompt = (
        f"Draft a concise investigation memo for this case.\n\n"
        f"EVIDENCE:\n{state['evidence_summary']}\n\n"
        f"PERSONA PANEL VERDICTS:\n{persona_summary}\n\n"
        f"CONSENSUS VERDICT: {state['consensus_verdict']}{human_note}\n\n"
        f"RELEVANT AML TYPOLOGY REFERENCE MATERIAL (cite by [document, page]):\n{typology_text}\n\n"
        f"Write a 3-5 paragraph memo: (1) summary of the case and verdict, (2) evidence supporting it, "
        f"(3) the specific typology pattern this matches, citing the reference material by source document, "
        f"(4) recommended next action."
    )
    resp = llm.invoke(prompt)
    return {"memo": resp.content}


def route_after_consensus(state: InvestigationState) -> Literal["human_review", "rag_lookup"]:
    return "rag_lookup" if state["status"] == "auto_finalized" else "human_review"


def build_graph():
    graph = StateGraph(InvestigationState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("persona_panel", persona_panel_node)
    graph.add_node("consensus_check", consensus_check_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("rag_lookup", rag_lookup_node)
    graph.add_node("draft_memo", draft_memo_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "persona_panel")
    graph.add_edge("persona_panel", "consensus_check")
    graph.add_conditional_edges(
        "consensus_check", route_after_consensus, {"human_review": "human_review", "rag_lookup": "rag_lookup"}
    )
    graph.add_edge("human_review", "rag_lookup")
    graph.add_edge("rag_lookup", "draft_memo")
    graph.add_edge("draft_memo", END)

    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)
