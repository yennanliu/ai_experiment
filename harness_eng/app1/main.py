"""
Harness Engineering Demo
========================
Illustrates all five harness components from rar.design/posts/harness-engineering-guide

  1. Repository Structure   → AGENTS.md gives the agent its standing instructions
  2. Architectural Constraint → tool allow-list enforced in tools.py (no arbitrary evals)
  3. Tools                  → calculate / remember / recall via tool_use loop in agent.py
  4. Context Management     → memory/ persists facts across sessions (memory.py)
  5. Evaluation Loop        → separate evaluator agent scores every answer (evaluator.py)
"""

from harness import agent, config, evaluator

QUESTIONS = [
    # Basic — single tool call, memory write
    "What is 18% of 240? Store the result as 'tip'.",

    # Memory read + arithmetic
    "Recall 'tip'. What is the total bill if the base was $240?",

    # Multi-tool — two calculations + two memory writes
    (
        "A project costs $12,500. Calculate 8% tax and store it as 'tax'. "
        "Then calculate a 5% contingency on the base cost and store it as 'contingency'. "
        "What are the two amounts?"
    ),

    # Compound recall + multi-step reasoning
    (
        "Recall 'tax' and 'contingency'. "
        "What is the final project budget (base + tax + contingency)? "
        "Store it as 'total_budget'."
    ),

    # Multi-person split + percentage breakdown
    (
        "Recall 'total_budget'. "
        "Split it equally among 6 stakeholders. "
        "Also calculate what percentage of 'total_budget' each of 'tax' and 'contingency' represents. "
        "Store the per-person share as 'share_per_person'."
    ),
]


def main() -> None:
    print(f"\n{'═' * 52}")
    print(f"  Provider : {config.PROVIDER}")
    print(f"  Model    : {config.active_model()}")
    print(f"{'═' * 52}")

    for q in QUESTIONS:
        print(f"\n{'─' * 52}")
        print(f"Q: {q}\n")

        answer = agent.run(q)
        print(f"\nA: {answer}")

        result = evaluator.evaluate(q, answer)
        score, reason = result.get("score", "?"), result.get("reason", "")
        print(f"Eval [{score}/5]: {reason}")

    print(f"\n{'═' * 52}")
    print("Done. Session memory saved to memory/store.json")


if __name__ == "__main__":
    main()
