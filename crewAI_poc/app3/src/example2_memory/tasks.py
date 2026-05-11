from crewai import Agent, Task


def research_task(agent: Agent, question: str) -> Task:
    return Task(
        description=(
            f"Answer this question thoroughly: **{question}**\n\n"
            "Draw on your knowledge and any prior context available to you. "
            "Be specific — cite concrete facts, names, and figures."
        ),
        expected_output="Factual, detailed answer (150–200 words) with specific facts.",
        agent=agent,
    )


def synthesis_task(agent: Agent, research_task: Task) -> Task:
    return Task(
        description=(
            "Review the researcher's answer and synthesize it into a final, polished response. "
            "If this answer connects to facts established in earlier questions, say so explicitly."
        ),
        expected_output=(
            "Polished answer (100–150 words) that integrates newly found facts with "
            "any relevant prior context, noting connections where they exist."
        ),
        agent=agent,
        context=[research_task],
    )
