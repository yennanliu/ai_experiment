import sys
from dotenv import load_dotenv
from src.example3_hierarchical.crew import build_crew

load_dotenv()


def main():
    product = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AI-powered personal finance app"
    print(f"\n=== Example 3 — Hierarchical Crew: Planning launch for '{product}' ===\n")
    crew = build_crew(product)
    result = crew.kickoff()
    print("\n=== Manager's Synthesized Output ===\n")
    print(result)


if __name__ == "__main__":
    main()
