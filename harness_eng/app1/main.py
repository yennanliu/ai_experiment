"""
Harness Engineering Demo
========================
Illustrates all five harness components from rar.design/posts/harness-engineering-guide

  1. Repository Structure  → AGENTS.md gives the agent its standing instructions
  2. Architectural Constraint → tool allow-list enforced in tools.py (no arbitrary evals)
  3. Tools & MCP            → calculate / remember / recall via tool_use loop in agent.py
  4. Context Management     → memory/ persists facts across sessions (memory.py)
  5. Evaluation Loop        → separate evaluator agent scores every answer (evaluator.py)
"""

from harness import agent, evaluator

QUESTIONS = [
    "What is 18% of 240? Store the result as 'tip'.",
    "Recall 'tip'. What is the total if the original bill was $240?",
]


def main() -> None:
    for q in QUESTIONS:
        print(f"\n{'─' * 52}")
        print(f"Q: {q}")

        answer = agent.run(q)
        print(f"A: {answer}")

        result = evaluator.evaluate(q, answer)
        score, reason = result.get("score", "?"), result.get("reason", "")
        print(f"Eval [{score}/5]: {reason}")

    print(f"\n{'─' * 52}")
    print("Done. Session memory saved to memory/store.json")


if __name__ == "__main__":
    main()
