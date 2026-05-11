from crewai import Agent


def market_analyst(llm) -> Agent:
    return Agent(
        role="Market Analyst",
        goal="Analyze market conditions and competitive landscape for the product",
        backstory=(
            "You are a sharp market analyst who quickly identifies market trends, "
            "target audiences, and competitive dynamics."
        ),
        llm=llm,
        verbose=True,
    )


def technical_lead(llm) -> Agent:
    return Agent(
        role="Technical Lead",
        goal="Assess technical feasibility and define the technology stack",
        backstory=(
            "You are a pragmatic senior engineer who evaluates technical complexity "
            "and proposes concrete, buildable solutions."
        ),
        llm=llm,
        verbose=True,
    )


def marketing_specialist(llm) -> Agent:
    return Agent(
        role="Marketing Specialist",
        goal="Craft a compelling go-to-market strategy and launch messaging",
        backstory=(
            "You are a creative marketer who builds focused launch strategies "
            "that resonate with the right audiences across the right channels."
        ),
        llm=llm,
        verbose=True,
    )
