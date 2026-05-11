import sys
from dotenv import load_dotenv
from src.example2_memory.crew import build_crew

load_dotenv()

# Three related questions — each crew run stores facts in long-term memory.
# Later runs recall earlier answers, so answers become progressively richer.
DEFAULT_QUESTIONS = [
    "What is the Python GIL (Global Interpreter Lock) and why does it exist?",
    "How does the Python GIL affect multithreaded programs in practice?",
    "What are the best alternatives to Python when you need true CPU-level parallelism?",
]


def main():
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = DEFAULT_QUESTIONS

    print("\n=== Example 2 — Memory: Session Research Assistant ===")
    print(f"Running {len(questions)} question(s). Memory persists across each run.\n")

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}/{len(questions)}: {question}")
        print("=" * 60)
        crew = build_crew(question)
        result = crew.kickoff()
        print(f"\n--- Answer {i} ---\n{result}\n")


if __name__ == "__main__":
    main()
