import os
import sys
from dotenv import load_dotenv
from src.example1_custom_tools.crew import build_crew

load_dotenv()

SAMPLE_TEXT = """Artificial Intelligence in Modern Business

The rapid advancement of artificial intelligence has fundamentally transformed how
businesses operate in the 21st century. Companies across industries are leveraging
AI to automate repetitive tasks, gain deeper insights from data, and deliver
personalized experiences to customers.

Machine learning algorithms now power recommendation engines, fraud detection
systems, and predictive maintenance tools. Natural language processing enables
chatbots and virtual assistants to handle customer inquiries with increasing
sophistication.

However, the integration of AI also presents significant challenges. Organizations
must invest heavily in data infrastructure and talent acquisition. Ethical concerns
around bias, privacy, and job displacement require careful consideration.

Despite these challenges, businesses that successfully harness AI stand to gain
substantial competitive advantages. The key lies in identifying high-impact use
cases, building robust data pipelines, and fostering a culture of continuous learning.
"""


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "sample.txt"

    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            f.write(SAMPLE_TEXT)
        print(f"Created sample file: {filepath}")

    print(f"\n=== Example 1 — Custom Tools: Analyzing '{filepath}' ===\n")
    crew = build_crew(filepath)
    result = crew.kickoff()
    print("\n=== Result ===\n")
    print(result)


if __name__ == "__main__":
    main()
