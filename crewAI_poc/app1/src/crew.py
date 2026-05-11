from crewai import Crew, Process, LLM
from src.agents import researcher, writer
from src.tasks import research_task, writing_task


def build_crew(topic: str) -> Crew:
    llm = LLM(model="openai/gpt-4o-mini")

    r_agent = researcher(llm)
    w_agent = writer(llm)

    r_task = research_task(r_agent, topic)
    w_task = writing_task(w_agent, r_task)

    return Crew(
        agents=[r_agent, w_agent],
        tasks=[r_task, w_task],
        process=Process.sequential,
        verbose=True,
    )
