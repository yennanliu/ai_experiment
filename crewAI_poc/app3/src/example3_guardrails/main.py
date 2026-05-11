import sys
from dotenv import load_dotenv
from src.example3_guardrails.crew import build_crew

load_dotenv()


def main():
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "remote work productivity trends"
    print(f"\n=== Example 3 — Guardrails + Callbacks: '{topic}' ===")
    print("Live step log  → output/steps.log")
    print("Task log       → output/tasks.log\n")

    crew = build_crew(topic)
    result = crew.kickoff()

    print("\n=== Final Report ===\n")
    print(result)
    print("\nMonitoring logs written to output/steps.log and output/tasks.log")


if __name__ == "__main__":
    main()
