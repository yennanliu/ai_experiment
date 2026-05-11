from crewai import Crew, LLM, Process
from src.example2_memory.agents import researcher, synthesizer
from src.example2_memory.tasks import research_task, synthesis_task


def build_crew(question: str) -> Crew:
    llm = LLM(model="openai/gpt-4o-mini")
    r_agent = researcher(llm)
    s_agent = synthesizer(llm)
    t1 = research_task(r_agent, question)
    t2 = synthesis_task(s_agent, t1)
    return Crew(
        agents=[r_agent, s_agent],
        tasks=[t1, t2],
        process=Process.sequential,
        memory=True,
        embedder={"provider": "onnx"},  # local ONNX MiniLM — no API key required
        verbose=True,
    )
