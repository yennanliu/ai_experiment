"""
Parallel Agent Fan-out Demo  (app3)
====================================
Harness pattern: run one Generator agent per plan component concurrently
using a ThreadPoolExecutor, then merge results into a consolidated artifact.

Compare with app2's sequential generator loop to see the difference.
"""

import time

from harness import config, evaluator, fan_out, generator, memory, merger, planner

# ── Task definitions ───────────────────────────────────────────────────────────

TASKS = [
    {
        "label": "Distributed Cache Layer",
        "task": (
            "Design a distributed cache layer for a high-traffic e-commerce platform. "
            "Cover: eviction policy, replication strategy, client interface, and observability."
        ),
    },
]


# ── Fan-out runner ─────────────────────────────────────────────────────────────

def run_fan_out(task_cfg: dict) -> None:
    task = task_cfg["task"]

    print("\n  [Planner] Decomposing task...")
    plan = planner.plan(task)
    n = len(plan["components"])
    print(f"  Components ({n}): {', '.join(plan['components'])}")
    print(f"  Artifacts  ({n}): {', '.join(plan['artifacts'])}")
    print(f"  Workers       : up to {config.MAX_WORKERS} parallel threads")

    mem = memory.load()

    print(f"\n  [Fan-out] Launching {n} generator agents in parallel...")
    t0 = time.perf_counter()
    results, mem = fan_out.run(
        task=task,
        plan=plan,
        generate_fn=generator.run,
        shared_mem=mem,
    )
    total = time.perf_counter() - t0

    print(f"\n  [Fan-out] All agents finished in {total:.1f}s total")
    print(f"  {'Component':<28} {'Status':<8} {'Time':>6}  Artifact")
    print(f"  {'─' * 28} {'─' * 8} {'─' * 6}  {'─' * 20}")
    for r in results:
        status = "OK" if r.ok else "FAILED"
        print(f"  {r.component:<28} {status:<8} {r.elapsed:>5.1f}s  {r.artifact}")
        if not r.ok:
            print(f"    Error: {r.error}")

    ok_count = sum(1 for r in results if r.ok)
    fail_count = len(results) - ok_count
    if fail_count:
        print(f"\n  ⚠  {fail_count} component(s) failed — merger will note the gaps.")

    memory.save(mem)

    print("\n  [Merger] Synthesising artifacts into integration summary...")
    summary = merger.merge(task, results)
    preview = summary[:300].replace("\n", " ")
    print(f"  Preview: {preview}...")
    print("  Written: artifacts/_merged.md")

    print("\n  [Evaluator] Scoring all artifacts (including merged summary)...")
    scores = evaluator.evaluate(task, plan)
    _print_scores(scores)
    for suggestion in scores.get("suggestions", []):
        print(f"    → {suggestion}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_scores(scores: dict) -> None:
    print(f"  Completeness      : {scores.get('completeness', '?')}/5")
    print(f"  Correctness       : {scores.get('correctness', '?')}/5")
    print(f"  Constraint Adh.   : {scores.get('constraint_adherence', '?')}/5")
    print(f"  Quality           : {scores.get('quality', '?')}/5")
    print(f"  Overall           : {scores.get('overall', '?')}/5")
    if scores.get("feedback"):
        print(f"  Feedback          : {scores['feedback']}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'═' * 62}")
    print("  Parallel Agent Fan-out Demo  (app3)")
    print(f"  Provider : {config.PROVIDER}")
    print(f"  Model    : {config.active_model()}")
    print(f"  Tier     : {config.model_tier()}")
    print(f"{'═' * 62}")

    if config.model_tier() == "low":
        print(
            "\n  ⚠  Low-capability model detected — JSON and tool calls may be\n"
            "     less reliable. Consider TEMPERATURE=0 and a higher-tier model.\n"
        )

    for task_cfg in TASKS:
        print(f"\n{'─' * 62}")
        print(f"  {task_cfg['label']}")
        print(f"  Task: {task_cfg['task'][:72]}...")
        run_fan_out(task_cfg)

    print(f"\n{'═' * 62}")
    print("Done.")
    print("  Artifacts → artifacts/   (one per component + _merged.md)")
    print("  Memory    → memory/store.json")


if __name__ == "__main__":
    main()
