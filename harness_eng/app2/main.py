"""
Advanced Harness Engineering Demo
===================================
Demonstrates four advanced patterns from rar.design/posts/harness-engineering-guide

  Pattern A — Three-Agent Pipeline   : Planner → Generator → Evaluator feedback loop
  Pattern B — Multi-Session Handoff  : Save/resume progress across sessions
  Pattern C — Constraint Enforcement : Checker flags violations; Generator corrects
  Pattern D — Code Project           : Software Eng agent writes actual Python source

Each pattern corresponds to a real technique described in the guide.
Compare with app1 to see how the harness grows in complexity and capability.
"""

from harness import (
    code_evaluator,
    code_generator,
    config,
    constraint_checker,
    evaluator,
    generator,
    handoff,
    planner,
)

# ── Task definitions ───────────────────────────────────────────────────────────

TASKS = [
    # {
    #     "label": "A. Three-Agent Pipeline (Planner → Generator → Evaluator loop)",
    #     "task": (
    #         "Design a token-bucket rate-limiting system for a public REST API. "
    #         "Cover the algorithm, the storage backend, and failure handling."
    #     ),
    #     "mode": "pipeline",
    #     "max_retries": 1,       # retry once if score < threshold
    #     "pass_threshold": 3.5,  # overall score required to skip retry
    # },
    # {
    #     "label": "B. Multi-Session Handoff (resume from saved progress)",
    #     "task": (
    #         "Build a design spec for a session-resumable ETL pipeline "
    #         "that ingests CSV files and loads them into a PostgreSQL database."
    #     ),
    #     "mode": "handoff",
    #     "session_id": "etl_pipeline_v1",
    # },
    # {
    #     "label": "C. Constraint Enforcement Loop (check → correct → re-check)",
    #     "task": (
    #         "Define the authentication module for a multi-tenant SaaS application "
    #         "using JWT access tokens with refresh-token rotation."
    #     ),
    #     "mode": "constraint_loop",
    #     "max_corrections": 1,
    # },
    # {
    #     "label": "D. Code Project (Software Eng agent writes real Python)",
    #     "task": (
    #         "Build a Python token-bucket rate limiter. "
    #         "Write: (1) rate_limiter.py with a RateLimiter class and token-bucket logic, "
    #         "(2) storage.py with an InMemoryStore and a RedisStore stub, "
    #         "(3) test_rate_limiter.py with pytest tests covering allow/deny/refill, "
    #         "(4) demo.py that shows the limiter in action with printed output."
    #     ),
    #     "mode": "code_project",
    # },
        {
        "label": "E. Code Project (Software Eng agent writes real Javascript)",
        "task": (
            "Build a Javascript load balancer. "
            "Write: (1) load_balancer.js with a core logic, "
            "(2) write other needed code "
            "(3) test_load_balancer.js with pytest tests covering allow/deny/refill, "
            "(4) demo.py that shows the limiter in action with printed output."
        ),
        "mode": "code_project",
    },
]


# ── Pattern A ──────────────────────────────────────────────────────────────────

def run_pipeline(task_cfg: dict) -> None:
    task = task_cfg["task"]
    max_retries = task_cfg.get("max_retries", 1)
    threshold = task_cfg.get("pass_threshold", 3.5)

    print("\n  [Planner] Decomposing task...")
    plan = planner.plan(task)
    print(f"  Components : {', '.join(plan['components'])}")
    print(f"  Artifacts  : {', '.join(plan['artifacts'])}")

    for attempt in range(1, max_retries + 2):
        if attempt > 1:
            print(f"\n  [Retry {attempt}] Regenerating with evaluator feedback...")

        print(f"\n  [Generator] Implementing (attempt {attempt})...")
        for component in plan["components"]:
            print(f"    → {component}")
            generator.run(task, plan, component)

        print("\n  [Constraint Checker] Scanning artifacts...")
        violations = constraint_checker.check(plan["artifacts"])
        if violations:
            for v in violations:
                sev = v.get("severity", "warning").upper()
                print(f"    [{sev}] {v['artifact']}: {v['violation']}")
        else:
            print("    No violations.")

        print("\n  [Evaluator] Scoring with rubric...")
        scores = evaluator.evaluate(task, plan, violations)
        _print_scores(scores)

        if scores.get("overall", 0) >= threshold:
            print(f"  ✓ Passed threshold ({threshold}).")
            break
        if attempt <= max_retries:
            print(f"  ✗ Below threshold — triggering retry.")


# ── Pattern B ──────────────────────────────────────────────────────────────────

def run_handoff(task_cfg: dict) -> None:
    task = task_cfg["task"]
    session_id = task_cfg["session_id"]

    prior = handoff.load_progress(session_id)
    if prior:
        plan = prior["plan"]
        completed = set(prior.get("completed_components", []))
        print(f"\n  [Handoff] Resuming session '{session_id}'")
        print(f"  Already done : {', '.join(completed) or 'none'}")
        print(f"  Remaining    : {', '.join(c for c in plan['components'] if c not in completed)}")
    else:
        print(f"\n  [Planner] New session '{session_id}' — creating plan...")
        plan = planner.plan(task)
        completed = set()
        print(f"  Components: {', '.join(plan['components'])}")

    for component in plan["components"]:
        if component in completed:
            print(f"  [Skip] {component}")
            continue

        print(f"  [Generator] {component}")
        generator.run(task, plan, component)
        completed.add(component)

        handoff.save_progress(session_id, {
            "task": task,
            "plan": plan,
            "completed_components": list(completed),
        })
        done = len(completed)
        total = len(plan["components"])
        print(f"  [Handoff] Saved — {done}/{total} components complete")

    print("\n  [Evaluator] Final evaluation...")
    violations = constraint_checker.check(plan["artifacts"])
    scores = evaluator.evaluate(task, plan, violations)
    _print_scores(scores)


# ── Pattern C ──────────────────────────────────────────────────────────────────

def run_constraint_loop(task_cfg: dict) -> None:
    task = task_cfg["task"]
    max_corrections = task_cfg.get("max_corrections", 1)

    print("\n  [Planner] Planning task...")
    plan = planner.plan(task)
    print(f"  Components: {', '.join(plan['components'])}")

    print("\n  [Generator] Initial implementation...")
    for component in plan["components"]:
        print(f"    → {component}")
        generator.run(task, plan, component)

    for round_num in range(max_corrections + 1):
        print(f"\n  [Constraint Checker] Round {round_num + 1}...")
        violations = constraint_checker.check(plan["artifacts"])

        if not violations:
            print("  ✓ All constraints satisfied.")
            break

        for v in violations:
            sev = v.get("severity", "warning").upper()
            print(f"    [{sev}] {v['artifact']}: {v['violation']}")

        if round_num < max_corrections:
            print(f"\n  [Generator] Correcting {len(violations)} violation(s)...")
            unique_artifacts = list({v["artifact"] for v in violations})
            correction_plan = {
                **plan,
                "components": [f"fix_{a}" for a in unique_artifacts],
                "artifacts": unique_artifacts,
            }
            for v in violations:
                fix_prompt = (
                    f"Revise '{v['artifact']}' to fix: {v['violation']}. "
                    f"Read the existing artifact first, then overwrite it with write_artifact."
                )
                generator.run(task, correction_plan, fix_prompt)

    print("\n  [Evaluator] Final assessment...")
    violations = constraint_checker.check(plan["artifacts"])
    scores = evaluator.evaluate(task, plan, violations)
    _print_scores(scores)
    for suggestion in scores.get("suggestions", []):
        print(f"    → {suggestion}")


# ── Pattern D ──────────────────────────────────────────────────────────────────

def run_code_project(task_cfg: dict) -> None:
    task = task_cfg["task"]

    print("\n  [Planner] Planning code project...")
    plan = planner.plan(task)
    print(f"  Components : {', '.join(plan['components'])}")

    print("\n  [Software Eng] Writing source files...")
    for component in plan["components"]:
        print(f"    → {component}")
        code_generator.run(task, plan, component)

    py_files = code_evaluator.list_workspace_files()
    print(f"\n  [Workspace] {len(py_files)} file(s) written:")
    for f in py_files:
        lines = len(f.read_text().splitlines())
        print(f"    {f.name:<30} {lines:>4} lines")

    print("\n  [Code Evaluator] Scoring...")
    scores = code_evaluator.evaluate(task, py_files)
    _print_code_scores(scores)
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


def _print_code_scores(scores: dict) -> None:
    print(f"  Correctness       : {scores.get('correctness', '?')}/5")
    print(f"  Completeness      : {scores.get('completeness', '?')}/5")
    print(f"  Code Quality      : {scores.get('code_quality', '?')}/5")
    print(f"  Testability       : {scores.get('testability', '?')}/5")
    print(f"  Overall           : {scores.get('overall', '?')}/5")
    if scores.get("feedback"):
        print(f"  Feedback          : {scores['feedback']}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'═' * 62}")
    print("  Advanced Harness Engineering Demo  (app2)")
    print(f"  Provider : {config.PROVIDER}")
    print(f"  Model    : {config.active_model()}")
    print(f"  Tier     : {config.model_tier()}")
    print(f"{'═' * 62}")

    if config.model_tier() == "low":
        print(
            "\n  ⚠  Low-capability model detected. JSON outputs and tool calls\n"
            "     may be less reliable. Consider setting TEMPERATURE=0 and\n"
            "     testing with a higher-tier model if results are inconsistent.\n"
        )

    dispatch = {
        "pipeline": run_pipeline,
        "handoff": run_handoff,
        "constraint_loop": run_constraint_loop,
        "code_project": run_code_project,
    }

    for task_cfg in TASKS:
        print(f"\n{'─' * 62}")
        print(f"  {task_cfg['label']}")
        print(f"  Task: {task_cfg['task'][:72]}...")
        dispatch[task_cfg["mode"]](task_cfg)

    print(f"\n{'═' * 62}")
    print("Done.")
    print("  Design artifacts → artifacts/")
    print("  Source code      → workspace/")
    print("  Session memory   → memory/store.json")


if __name__ == "__main__":
    main()
