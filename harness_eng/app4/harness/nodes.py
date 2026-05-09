"""
Project 12: Multi-Project Orchestrator — async LangGraph node functions.

Three nodes compose the graph:
  decompose       — Director splits a large goal into sub-projects
  process_project — Plans, generates, and evaluates one sub-project
  aggregate       — Synthesises all results into a final report
"""

import json
import re
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from . import config, vector_memory

_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
_ARTIFACTS_DIR.mkdir(exist_ok=True)


# ── LLM factory ────────────────────────────────────────────────────────────────

def _llm(streaming: bool = False):
    if config.PROVIDER == "openai":
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            streaming=streaming,
        )
    return ChatAnthropic(
        model=config.ANTHROPIC_MODEL,
        api_key=config.ANTHROPIC_API_KEY,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        streaming=streaming,
    )


# ── JSON helpers ───────────────────────────────────────────────────────────────

def _parse_json(raw: str, fallback: dict) -> dict:
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return fallback


# ── Node: decompose ─────────────────────────────────────────────────────────

async def decompose(state: dict) -> dict:
    """Director: break the goal into 2-4 concrete, non-overlapping sub-projects."""
    goal = state["goal"]

    past = vector_memory.search(f"decompose: {goal}", n=2)
    memory_ctx = "\n".join(f"- {r['text']}" for r in past) if past else ""

    system = (
        "You are a senior engineering director. Decompose a large engineering goal "
        "into 2-4 concrete, non-overlapping sub-projects.\n"
        "Respond with valid JSON only — no prose, no fences:\n"
        '{"projects": [{"label": "Short Name", "task": "Full task description"}]}'
    )
    user = f"Goal: {goal}"
    if memory_ctx:
        user += f"\n\nRelated prior work (from memory):\n{memory_ctx}"

    resp = await _llm().ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    data = _parse_json(resp.content, {"projects": []})
    projects = data.get("projects") or [{"label": "Full Implementation", "task": goal}]

    vector_memory.add(
        f"decompose:{goal[:60]}",
        f"Goal: {goal}\nProjects: {', '.join(p['label'] for p in projects)}",
        {"type": "decomposition"},
    )

    return {"projects": projects}


# ── Node: process_project ───────────────────────────────────────────────────

async def process_project(state: dict) -> dict:
    """Plan + Generate + Evaluate a single sub-project, with vector memory."""
    goal = state["goal"]
    project = state["project"]
    label = project["label"]
    task = project["task"]

    # Retrieve relevant semantic context before planning
    ctx_hits = vector_memory.search(task, n=3)
    ctx = "\n".join(f"- [{r['id']}] {r['text']}" for r in ctx_hits) if ctx_hits else "None yet."

    llm = _llm()

    # ── Plan ─────────────────────────────────────────────────────────────────
    plan_resp = await llm.ainvoke([
        SystemMessage(content=(
            "You are a technical planner. Given a sub-project task, produce a JSON plan.\n"
            "Respond with valid JSON only:\n"
            '{"components": ["...", "..."], "constraints": ["must ...", "must ..."]}'
        )),
        HumanMessage(content=f"Sub-project: {label}\nTask: {task}\n\nMemory context:\n{ctx}"),
    ])
    plan = _parse_json(plan_resp.content, {})
    plan.setdefault("components", ["design", "implementation", "error_handling"])
    plan.setdefault("constraints", ["must address security", "must document interfaces"])

    # ── Generate artifact ─────────────────────────────────────────────────────
    constraints_block = "\n".join(f"  - {c}" for c in plan["constraints"])
    gen_resp = await llm.ainvoke([
        SystemMessage(content=(
            "You are a senior engineer. Write a detailed technical design document in markdown. "
            "Cover every component. No stubs, no TODOs."
        )),
        HumanMessage(content=(
            f"Overall goal: {goal}\n"
            f"Sub-project: {label}\n"
            f"Task: {task}\n"
            f"Components: {', '.join(plan['components'])}\n"
            f"Constraints:\n{constraints_block}\n\n"
            f"Prior knowledge from memory:\n{ctx}\n\n"
            "Write the full technical document."
        )),
    ])
    artifact_content = gen_resp.content

    slug = label.lower().replace(" ", "_")
    (_ARTIFACTS_DIR / f"{slug}.md").write_text(artifact_content)

    # Store summary in vector memory for future projects to reuse
    vector_memory.add(
        f"artifact:{slug}",
        f"Project: {label} | Task: {task} | Summary: {artifact_content[:400]}",
        {"type": "artifact", "project": label},
    )

    # ── Evaluate (streaming LLM — tokens flow to dashboard via astream_events) ─
    eval_resp = await _llm(streaming=True).ainvoke(
        [
            SystemMessage(content=(
                "You are a strict technical evaluator. Score the artifact on:\n"
                "  completeness (1-5), correctness (1-5), quality (1-5)\n"
                "Respond with valid JSON only:\n"
                '{"completeness": <int>, "correctness": <int>, "quality": <int>, '
                '"overall": <float>, "feedback": "<one sentence>"}'
            )),
            HumanMessage(content=(
                f"Task: {task}\n"
                f"Components expected: {', '.join(plan['components'])}\n\n"
                f"Artifact:\n{artifact_content[:3000]}"
            )),
        ],
        config={"tags": ["evaluator"]},
    )
    scores = _parse_json(eval_resp.content, {})
    scores.setdefault("completeness", 3)
    scores.setdefault("correctness", 3)
    scores.setdefault("quality", 3)
    if "overall" not in scores:
        scores["overall"] = round(
            sum(scores[k] for k in ("completeness", "correctness", "quality")) / 3, 1
        )
    scores.setdefault("feedback", "")

    return {"results": [{
        "label": label,
        "task": task,
        "artifact": slug,
        "plan": plan,
        "scores": scores,
        "artifact_len": len(artifact_content),
    }]}


# ── Node: aggregate ─────────────────────────────────────────────────────────

async def aggregate(state: dict) -> dict:
    """Synthesise all sub-project results into a final orchestration report."""
    results = state["results"]
    goal = state["goal"]

    score_lines = []
    for r in sorted(results, key=lambda x: x["label"]):
        s = r["scores"]
        score_lines.append(
            f"### {r['label']}\n"
            f"- Artifact: `{r['artifact']}.md` ({r['artifact_len']} chars)\n"
            f"- Scores: completeness={s.get('completeness','?')}/5  "
            f"correctness={s.get('correctness','?')}/5  "
            f"quality={s.get('quality','?')}/5  "
            f"overall={s.get('overall','?')}/5\n"
            f"- Feedback: {s.get('feedback','')}"
        )

    resp = await _llm().ainvoke([
        SystemMessage(content=(
            "You are a senior engineering director. "
            "Write a concise integration summary showing how the sub-projects fit together. "
            "Use markdown. Cover shared interfaces, data flows, and integration risks."
        )),
        HumanMessage(content=(
            f"Goal: {goal}\n\n"
            f"Sub-project results:\n\n" + "\n\n".join(score_lines) + "\n\n"
            "Write a 200-word integration summary."
        )),
    ])
    report = resp.content

    (_ARTIFACTS_DIR / "_report.md").write_text(
        f"# Orchestrator Report\n\nGoal: {goal}\n\n"
        + "\n\n".join(score_lines)
        + f"\n\n## Integration Summary\n\n{report}"
    )

    return {"final_report": report}
