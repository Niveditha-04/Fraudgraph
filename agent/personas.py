"""
Phase 6: the three-persona panel.

Structured output (a Pydantic schema via `.with_structured_output`) is used
instead of asking the LLM to produce free-text and regex/JSON-parsing it out
-- reliable structured output is one of the things worth doing properly in
an agentic system, and it removes an entire class of "the model almost
followed the format" parsing failures.
"""
from typing import Literal

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

Verdict = Literal["illicit", "licit", "uncertain"]


class PersonaVerdict(BaseModel):
    verdict: Verdict = Field(description="This persona's independent judgment on the case")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the verdict, 0-1")
    reasoning: str = Field(description="2-4 sentence justification citing the specific evidence given")


PERSONA_SYSTEM_PROMPTS = {
    "aml_analyst": (
        "You are an AML (anti-money-laundering) analyst reviewing a flagged Bitcoin wallet case. "
        "You focus on transaction-pattern evidence: graph-neighborhood risk signals, statistical "
        "anomalies (e.g. Benford's Law deviation in transaction amounts), and known typologies "
        "(layering, structuring, mixing services, exchange-hopping). You are thorough but not "
        "alarmist -- a high model score alone, without a coherent pattern, is not enough for you "
        "to call a case illicit. Weigh the evidence given and give your independent verdict."
    ),
    "compliance_officer": (
        "You are a compliance officer at a financial institution, reviewing a flagged Bitcoin "
        "wallet case for a potential Suspicious Activity Report (SAR) filing decision. You think "
        "in terms of regulatory obligations and institutional risk: would this evidence, as "
        "documented, meet the bar to justify escalation under FinCEN/FATF guidance? You are "
        "calibrated by the cost of both false positives (wasted investigator time, annoyed "
        "customers) and false negatives (regulatory and reputational exposure). Give your "
        "independent verdict based on the evidence given."
    ),
    "skeptic": (
        "You are a skeptical reviewer whose job is to actively look for reasons the flagged case "
        "might be a FALSE POSITIVE. Automated fraud-detection systems are known to over-flag: "
        "high-volume legitimate businesses (exchanges, payment processors, miners) can look "
        "statistically unusual without being illicit. Before agreeing with an 'illicit' call, "
        "ask yourself what an innocent explanation for this evidence would be. You are not "
        "contrarian for its own sake -- if the evidence is genuinely damning, say so -- but your "
        "default posture is to demand a real pattern, not just an elevated score."
    ),
}


def build_persona_llm(persona_key: str, model: str = "claude-haiku-4-5-20251001") -> ChatAnthropic:
    llm = ChatAnthropic(model=model, max_tokens=500, temperature=0)
    return llm.with_structured_output(PersonaVerdict)
