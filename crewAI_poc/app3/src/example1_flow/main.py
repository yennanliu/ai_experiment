import sys
from dotenv import load_dotenv
from src.example1_flow.flow import SmartArticleFlow

load_dotenv()


def main():
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "transformer neural networks"
    print(f"\n=== Example 1 — Flow with Router: '{topic}' ===\n")

    flow = SmartArticleFlow()
    flow.state.topic = topic
    flow.kickoff()

    print("\n=== Final Article ===\n")
    print(f"Content type routed to: [{flow.state.content_type.upper()}]\n")
    print(flow.state.final_article)


if __name__ == "__main__":
    main()
