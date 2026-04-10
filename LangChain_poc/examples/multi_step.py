"""Multi-step chain — research a topic, then generate a structured summary."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core import get_llm

llm = get_llm()

# Step 1: Research and extract key points
research_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Given a topic, list 5 key facts. Be factual and concise."),
    ("human", "Topic: {topic}"),
])

# Step 2: Turn key points into a structured summary
summary_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a technical writer. Given research notes, produce a structured summary with: "
     "1) One-line overview, 2) Key points as bullet list, 3) A 'Why it matters' conclusion."),
    ("human", "Research notes:\n{research}"),
])

# Compose the two steps
research_chain = research_prompt | llm | StrOutputParser()
summary_chain = summary_prompt | llm | StrOutputParser()

full_chain = research_chain | (lambda research: summary_chain.invoke({"research": research}))


if __name__ == "__main__":
    print("Multi-Step Research → Summary (type 'quit' to exit)\n")
    while True:
        topic = input("Topic: ").strip()
        if not topic or topic.lower() == "quit":
            break
        print("\nResearching & summarizing...\n")
        result = full_chain.invoke({"topic": topic})
        print(result)
        print("\n" + "=" * 60 + "\n")
