from crewai import Crew, Process, LLM
from src.example3_hierarchical.agents import market_analyst, technical_lead, marketing_specialist
from src.example3_hierarchical.tasks import market_analysis_task, tech_assessment_task, gtm_task


def build_crew(product: str) -> Crew:
    llm = LLM(model="openai/gpt-4o-mini")

    analyst = market_analyst(llm)
    tech = technical_lead(llm)
    marketer = marketing_specialist(llm)

    task1 = market_analysis_task(analyst, product)
    task2 = tech_assessment_task(tech, product)
    task3 = gtm_task(marketer, product)

    return Crew(
        agents=[analyst, tech, marketer],
        tasks=[task1, task2, task3],
        process=Process.hierarchical,
        manager_llm=LLM(model="openai/gpt-4o"),
        verbose=True,
    )
