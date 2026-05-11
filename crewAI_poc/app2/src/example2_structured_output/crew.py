from crewai import Crew, Process, LLM
from src.example2_structured_output.agents import market_analyst, strategist
from src.example2_structured_output.tasks import research_task, strategy_task


def build_crew(market: str) -> Crew:
    llm = LLM(model="openai/gpt-4o-mini")
    analyst = market_analyst(llm)
    strat = strategist(llm)
    task1 = research_task(analyst, market)
    task2 = strategy_task(strat, task1, market)
    return Crew(
        agents=[analyst, strat],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True,
    )
