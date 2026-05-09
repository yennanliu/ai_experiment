"""
app4 — Multi-Project Orchestrator with Vector Memory + Streaming Dashboard

Projects implemented:
  11. Persistent Vector Memory  (harness/vector_memory.py — ChromaDB)
  12. Multi-Project Orchestrator (harness/graph.py       — LangGraph)
  13. Streaming Evaluator Dashboard (harness/dashboard.py — Rich TUI)

Run:
    uv run python main.py
"""

import asyncio
from datetime import datetime
from langchain_core.messages import AIMessageChunk

from harness import config, vector_memory
from harness.graph import graph
from harness.dashboard import Dashboard


GOAL = (
    "Design a production-ready real-time analytics platform: "
    "data ingestion pipeline, stream processing layer, columnar storage tier, "
    "and a low-latency query API with observability."
)


async def run(dash: Dashboard, gen_id: str) -> dict:
    inputs = {
        "goal": GOAL,
        "gen_id": gen_id,
        "projects": [],
        "results": [],
        "final_report": "",
    }

    dash.set_phase("decomposing")
    final_state: dict = {}

    async for event in graph.astream_events(inputs, version="v2"):
        etype = event["event"]
        name = event.get("name", "")
        node = event.get("metadata", {}).get("langgraph_node", "")
        data = event.get("data", {})

        # ── Project discovery ─────────────────────────────────────────────
        # Filter by name==node to skip _fan_out's on_chain_end (same node
        # metadata but output is a list of Send objects, not a state dict).
        if etype == "on_chain_end" and name == "decompose" and node == "decompose":
            output = data.get("output", {})
            projects = output.get("projects", []) if isinstance(output, dict) else []
            dash.set_projects(projects)
            dash.set_phase("generating")

        # ── Per-project lifecycle ─────────────────────────────────────────
        elif etype == "on_chain_start" and name == "process_project" and node == "process_project":
            project = data.get("input", {}).get("project", {})
            label = project.get("label", "?")
            if label not in dash._projects:
                dash.set_projects([project])
            dash.set_status(label, "running")

        elif etype == "on_chain_end" and name == "process_project" and node == "process_project":
            output = data.get("output", {})
            results = output.get("results", []) if isinstance(output, dict) else []
            for r in results:
                dash.set_status(r["label"], "done")
                dash.set_scores(r["label"], r.get("scores", {}))

        # ── Aggregation ───────────────────────────────────────────────────
        elif etype == "on_chain_start" and name == "aggregate" and node == "aggregate":
            dash.set_phase("aggregating")

        elif etype == "on_chain_end" and name == "aggregate" and node == "aggregate":
            final_state = data.get("output", {})
            dash.set_phase("done")

        # ── Live token streaming ──────────────────────────────────────────
        elif etype == "on_llm_stream":
            chunk = data.get("chunk")
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                content = chunk.content
                if isinstance(content, str):
                    dash.push_token(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            dash.push_token(block.get("text", ""))

    return final_state


def _print_summary(state: dict, dash: Dashboard, gen_id: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  Run ID    : {gen_id}")
    print(f"  Provider  : {config.PROVIDER} / {config.active_model()}")
    print(f"  Memory    : {vector_memory.count()} documents in ChromaDB")
    print(f"{'─' * 64}")

    for label, s in sorted(dash._scores.items()):
        print(
            f"  {label:<30}  "
            f"C={s.get('completeness','?')}  Co={s.get('correctness','?')}  "
            f"Q={s.get('quality','?')}  Overall={s.get('overall','?')}/5"
        )

    print(f"{'─' * 64}")
    report = state.get("final_report", "")
    if report:
        print("  Integration Summary (excerpt):")
        print("  " + report[:400].replace("\n", "\n  "))
    print(f"{'─' * 64}")
    print(f"  Artifacts → artifacts/{gen_id}/")
    print(f"  Memory    → memory/chroma/")
    print(f"{'═' * 64}\n")


async def main() -> None:
    gen_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'═' * 64}")
    print("  app4 — Multi-Project Orchestrator")
    print(f"  Run ID   : {gen_id}")
    print(f"  Provider : {config.PROVIDER} / {config.active_model()}")
    print(f"  Memory   : {vector_memory.count()} documents already in ChromaDB")
    print(f"{'═' * 64}\n")

    dash = Dashboard(goal=GOAL, gen_id=gen_id)
    final = await dash.run(run(dash, gen_id))
    _print_summary(final, dash, gen_id)


if __name__ == "__main__":
    asyncio.run(main())
