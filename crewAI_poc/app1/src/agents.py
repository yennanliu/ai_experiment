from crewai import Agent
from crewai_tools import SerperDevTool

search_tool = SerperDevTool()


def researcher(llm) -> Agent:
    return Agent(
        role="Senior Research Analyst",
        goal="Uncover accurate, insightful information on the given topic",
        backstory=(
            "You are a meticulous researcher with a talent for finding "
            "credible sources and synthesizing complex information clearly."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
    )


def writer(llm) -> Agent:
    return Agent(
        role="Content Writer",
        goal="Transform research findings into a polished, structured report",
        backstory=(
            "You are a skilled writer who excels at turning dense research "
            "into engaging, well-structured reports for a general audience."
        ),
        llm=llm,
        verbose=True,
    )
