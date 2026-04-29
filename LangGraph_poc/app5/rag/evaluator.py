"""
LLM-as-judge evaluator for RAG answers.

Scores a (question, context, answer) triple on four dimensions:

  faithfulness        – Every claim in the answer is supported by the context.
                        Hallucinated facts score low even if the answer sounds good.
  answer_relevance    – The answer actually addresses what was asked.
                        An off-topic but accurate answer still scores low.
  context_utilization – The answer makes good use of the retrieved chunks.
                        Ignoring obviously relevant context scores low.
  correctness         – (optional) Factual agreement with a reference answer.
                        Skipped (returns None) when no reference is provided.

All scores are floats in [0.0, 1.0]. The evaluator returns a dict with:
  scores      – the four dimensions above
  overall     – weighted average of the available scores
  reasoning   – one-sentence explanation per dimension
  passed      – True when overall >= threshold (default 0.7)
"""

import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    return _llm


# Weights for the overall score (correctness has higher weight when a reference exists)
_WEIGHTS_NO_REF  = {"faithfulness": 0.4, "answer_relevance": 0.35, "context_utilization": 0.25}
_WEIGHTS_WITH_REF = {"faithfulness": 0.3, "answer_relevance": 0.25, "context_utilization": 0.2, "correctness": 0.25}

_SYSTEM_PROMPT = """\
You are a strict RAG evaluation judge. You receive:
  - QUESTION: the user's question
  - CONTEXT: the retrieved document chunks used to generate the answer
  - ANSWER: the AI-generated answer
  - REFERENCE (optional): a known-correct reference answer

Score the answer on the dimensions below. Return ONLY a JSON object — no prose outside the JSON.

Dimensions:
  faithfulness        (0.0–1.0): Every factual claim in the answer is explicitly supported
                                 by the CONTEXT. Deduct for any claim not found in CONTEXT.
  answer_relevance    (0.0–1.0): The answer directly and completely addresses the QUESTION.
                                 Deduct for vagueness, off-topic content, or missing key points.
  context_utilization (0.0–1.0): The answer makes good use of the relevant parts of CONTEXT.
                                 Deduct if the answer ignores chunks that clearly contain the answer.
  correctness         (0.0–1.0): ONLY scored when REFERENCE is provided. Factual agreement with
                                 the REFERENCE answer. Omit this key entirely when no REFERENCE.

Required JSON format:
{
  "faithfulness":        <float 0.0–1.0>,
  "answer_relevance":    <float 0.0–1.0>,
  "context_utilization": <float 0.0–1.0>,
  "correctness":         <float 0.0–1.0 or omit>,
  "reasoning": {
    "faithfulness":        "<one sentence>",
    "answer_relevance":    "<one sentence>",
    "context_utilization": "<one sentence>",
    "correctness":         "<one sentence or omit>"
  }
}
"""


def evaluate(
    question: str,
    context: list[tuple[str, str]],
    answer: str,
    reference: str | None = None,
    threshold: float = 0.7,
) -> dict:
    """
    Evaluate a RAG answer.

    Args:
        question:  The original user question.
        context:   Retrieved chunks as (text, source) tuples.
        answer:    The generated answer to evaluate.
        reference: Optional known-correct reference answer for correctness scoring.
        threshold: overall score below which passed=False (default 0.7).

    Returns a dict:
        {
          "scores": {
              "faithfulness": 0.9,
              "answer_relevance": 0.8,
              "context_utilization": 0.7,
              "correctness": 0.95,   # only if reference provided
          },
          "reasoning": { ... one sentence per dimension ... },
          "overall": 0.86,
          "passed": True,
          "threshold": 0.7,
        }
    """
    context_text = "\n\n---\n\n".join(
        f"[{src}]\n{text}" for text, src in context
    )

    user_content = f"QUESTION:\n{question}\n\nCONTEXT:\n{context_text}\n\nANSWER:\n{answer}"
    if reference:
        user_content += f"\n\nREFERENCE:\n{reference}"

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    raw = _get_llm().invoke(messages).content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    parsed = json.loads(raw)

    scores = {
        "faithfulness":        float(parsed.get("faithfulness", 0)),
        "answer_relevance":    float(parsed.get("answer_relevance", 0)),
        "context_utilization": float(parsed.get("context_utilization", 0)),
    }
    if reference and "correctness" in parsed:
        scores["correctness"] = float(parsed["correctness"])

    weights = _WEIGHTS_WITH_REF if "correctness" in scores else _WEIGHTS_NO_REF
    total_w = sum(weights[k] for k in scores)
    overall = sum(scores[k] * weights[k] for k in scores) / total_w

    reasoning = parsed.get("reasoning", {})

    return {
        "scores": scores,
        "reasoning": {k: reasoning.get(k, "") for k in scores},
        "overall": round(overall, 4),
        "passed": overall >= threshold,
        "threshold": threshold,
    }
