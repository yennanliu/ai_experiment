import sys
from dotenv import load_dotenv
from src.crew import build_crew

load_dotenv()


def main():
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "the rise of AI agents in 2025"
    print(f"\n=== Research Crew starting on: {topic} ===\n")

    crew = build_crew(topic)
    result = crew.kickoff()

    print("\n=== Final Report ===\n")
    print(result)


if __name__ == "__main__":
    main()
