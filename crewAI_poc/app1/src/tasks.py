from crewai import Task
from crewai import Agent


def research_task(researcher: Agent, topic: str) -> Task:
    return Task(
        description=(
            f"Research the following topic thoroughly: **{topic}**\n\n"
            "Find key facts, recent developments, and notable insights. "
            "Summarize your findings with supporting sources."
        ),
        expected_output=(
            "A structured research summary with: key facts, recent developments, "
            "and a source list. Plain text, no markdown."
        ),
        agent=researcher,
    )


def writing_task(writer: Agent, research_task: Task) -> Task:
    return Task(
        description=(
            "Using the researcher's findings, write a concise report (400–600 words). "
            "Include an introduction, key insights section, and a conclusion."
        ),
        expected_output=(
            "A well-structured report in markdown with: # Title, ## Introduction, "
            "## Key Insights, ## Conclusion."
        ),
        agent=writer,
        context=[research_task],
    )
