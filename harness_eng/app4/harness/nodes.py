"""
Project 12: Multi-Project Orchestrator — async LangGraph node functions.

Three nodes compose the graph:
  decompose       — Director splits a large goal into sub-projects
  process_project — Plans, generates design doc + code, and evaluates
  aggregate       — Synthesises all results into a final report

Each run is keyed by gen_id (e.g. "20260509_175030") so multiple runs
accumulate under artifacts/{gen_id}/ without overwriting each other.
"""

import json
import re
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from . import config, vector_memory

_ARTIFACTS_ROOT = Path(__file__).parent.parent / "artifacts"


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


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _artifact_dir(gen_id: str) -> Path:
    d = _ARTIFACTS_ROOT / gen_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences, keeping only the inner content."""
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip(), flags=re.MULTILINE)
    return text.rstrip("`").strip()


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
    """Plan + generate design doc + generate code + evaluate for one sub-project."""
    goal = state["goal"]
    gen_id = state["gen_id"]
    project = state["project"]
    label = project["label"]
    task = project["task"]
    slug = label.lower().replace(" ", "_")
    out_dir = _artifact_dir(gen_id)

    # Retrieve semantic context before planning
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

    constraints_block = "\n".join(f"  - {c}" for c in plan["constraints"])

    # ── Design document ───────────────────────────────────────────────────────
    design_resp = await llm.ainvoke([
        SystemMessage(content=(
            "You are a senior engineer. Write a detailed technical design document in markdown. "
            "Cover every component with Purpose, Design, and Interface sections. No stubs."
        )),
        HumanMessage(content=(
            f"Overall goal: {goal}\n"
            f"Sub-project: {label}\n"
            f"Task: {task}\n"
            f"Components: {', '.join(plan['components'])}\n"
            f"Constraints:\n{constraints_block}\n\n"
            f"Prior knowledge from memory:\n{ctx}\n\n"
            "Write the full technical design document."
        )),
    ])
    design_doc = design_resp.content
    (out_dir / f"{slug}.md").write_text(design_doc)

    # ── Code generation ───────────────────────────────────────────────────────
    code_resp = await llm.ainvoke([
        SystemMessage(content=(
            "You are a senior Python engineer. Given a technical design document, "
            "write a clean, working Python implementation. Rules:\n"
            "  - Module docstring at the top\n"
            "  - Type hints throughout (use dataclasses, typing, asyncio as needed)\n"
            "  - Real logic — no stubs, no TODO, no pass placeholders\n"
            "  - Standard library + common packages only\n"
            "  - 100-250 lines max; focus on the core; skip infra boilerplate\n"
            "Output raw Python — no markdown fences."
        )),
        HumanMessage(content=(
            f"Sub-project: {label}\n"
            f"Components to implement: {', '.join(plan['components'])}\n\n"
            f"Design document:\n{design_doc[:4000]}"
        )),
    ])
    code = _strip_code_fence(code_resp.content)
    (out_dir / f"{slug}.py").write_text(code)

    # Store design summary in vector memory for future runs to reuse
    vector_memory.add(
        f"artifact:{gen_id}:{slug}",
        f"Project: {label} | Task: {task} | Summary: {design_doc[:400]}",
        {"type": "artifact", "project": label, "gen_id": gen_id},
    )

    # ── Evaluate (streaming LLM — tokens captured by astream_events) ─────────
    eval_resp = await _llm(streaming=True).ainvoke(
        [
            SystemMessage(content=(
                "You are a strict technical evaluator. Score the design doc + code on:\n"
                "  completeness (1-5), correctness (1-5), quality (1-5)\n"
                "Respond with valid JSON only:\n"
                '{"completeness": <int>, "correctness": <int>, "quality": <int>, '
                '"overall": <float>, "feedback": "<one sentence>"}'
            )),
            HumanMessage(content=(
                f"Task: {task}\n"
                f"Components expected: {', '.join(plan['components'])}\n\n"
                f"Design doc:\n{design_doc[:2000]}\n\n"
                f"Code:\n{code[:1000]}"
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
        "slug": slug,
        "plan": plan,
        "scores": scores,
        "design_len": len(design_doc),
        "code_len": len(code),
    }]}


# ── Node: aggregate ─────────────────────────────────────────────────────────

async def aggregate(state: dict) -> dict:
    """Synthesise all sub-project results into a final orchestration report."""
    results = state["results"]
    goal = state["goal"]
    gen_id = state["gen_id"]
    out_dir = _artifact_dir(gen_id)

    score_lines = []
    for r in sorted(results, key=lambda x: x["label"]):
        s = r["scores"]
        score_lines.append(
            f"### {r['label']}\n"
            f"- Design : `{r['slug']}.md` ({r['design_len']} chars)\n"
            f"- Code   : `{r['slug']}.py`  ({r['code_len']} chars)\n"
            f"- Scores : C={s.get('completeness','?')}  Co={s.get('correctness','?')}  "
            f"Q={s.get('quality','?')}  Overall={s.get('overall','?')}/5\n"
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

    (out_dir / "_report.md").write_text(
        f"# Orchestrator Report\n\n"
        f"**Goal:** {goal}\n"
        f"**Run ID:** `{gen_id}`\n\n"
        + "\n\n".join(score_lines)
        + f"\n\n## Integration Summary\n\n{report}"
    )

    return {"final_report": report}
