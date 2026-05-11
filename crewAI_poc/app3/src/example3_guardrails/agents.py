from crewai import Agent


def analyst(llm) -> Agent:
    return Agent(
        role="Data Analyst",
        goal="Produce rigorous, evidence-based analysis with specific findings",
        backstory=(
            "Meticulous analyst who always backs claims with data and clear reasoning. "
            "You never write vague summaries — every point has supporting evidence."
        ),
        llm=llm,
        verbose=True,
    )


def writer(llm) -> Agent:
    return Agent(
        role="Technical Writer",
        goal="Transform analysis into a structured report with all required sections",
        backstory=(
            "Precise, well-organized writer who always follows the required format. "
            "You produce complete markdown reports with every mandatory section present."
        ),
        llm=llm,
        verbose=True,
    )
