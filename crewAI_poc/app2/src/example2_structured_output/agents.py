from crewai import Agent


def market_analyst(llm) -> Agent:
    return Agent(
        role="Market Research Analyst",
        goal="Research markets and compile structured, factual analysis",
        backstory=(
            "You are a seasoned market analyst with expertise in distilling "
            "complex market dynamics into clear, structured insights."
        ),
        llm=llm,
        verbose=True,
    )


def strategist(llm) -> Agent:
    return Agent(
        role="Business Strategist",
        goal="Synthesize market analysis into actionable structured recommendations",
        backstory=(
            "You translate research into precise business strategy, always grounding "
            "recommendations in data and expressing confidence honestly."
        ),
        llm=llm,
        verbose=True,
    )
