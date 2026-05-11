from crewai import Crew, Process, LLM
from src.example1_custom_tools.agents import document_analyst, report_writer
from src.example1_custom_tools.tasks import analyze_task, write_report_task


def build_crew(filepath: str) -> Crew:
    llm = LLM(model="openai/gpt-4o-mini")
    analyst = document_analyst(llm)
    writer = report_writer(llm)
    task1 = analyze_task(analyst, filepath)
    task2 = write_report_task(writer, task1)
    return Crew(
        agents=[analyst, writer],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True,
    )
