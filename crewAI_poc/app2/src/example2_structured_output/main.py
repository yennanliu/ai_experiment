import json
import sys
from dotenv import load_dotenv
from src.example2_structured_output.crew import build_crew
from src.example2_structured_output.models import MarketReport

load_dotenv()


def main():
    market = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "electric vehicles"
    print(f"\n=== Example 2 — Structured Output: Analyzing '{market}' market ===\n")
    crew = build_crew(market)
    result = crew.kickoff()

    print("\n=== Structured Result (Pydantic) ===\n")
    report: MarketReport = result.pydantic
    if isinstance(report, MarketReport):
        print(f"Topic:       {report.topic}")
        print(f"Confidence:  {report.confidence_score:.0%}\n")
        print("Key Facts:")
        for fact in report.key_facts:
            print(f"  • {fact}")
        print("\nOpportunities:")
        for opp in report.opportunities:
            print(f"  + {opp}")
        print("\nRisks:")
        for risk in report.risks:
            print(f"  - {risk}")
        print(f"\nRecommendation: {report.recommendation}")
        print("\n--- Raw JSON ---")
        print(json.dumps(report.model_dump(), indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
