from crewai import Task, Agent


def market_analysis_task(analyst: Agent, product: str) -> Task:
    return Task(
        description=(
            f"Analyze the market for '{product}'. Cover: target audience profile, "
            "market size estimate, top 3 competitors, and key differentiators."
        ),
        expected_output=(
            "Market analysis: target audience profile, market size estimate, "
            "competitor comparison (3 players), and differentiation opportunities."
        ),
        agent=analyst,
    )


def tech_assessment_task(tech_lead: Agent, product: str) -> Task:
    return Task(
        description=(
            f"Assess the technical requirements for '{product}'. Cover: "
            "recommended tech stack, top 3 technical challenges with mitigations, "
            "and an estimated 3-phase MVP timeline."
        ),
        expected_output=(
            "Technical assessment: stack recommendation with rationale, "
            "3 challenges + mitigations, and a phased MVP roadmap."
        ),
        agent=tech_lead,
    )


def gtm_task(marketer: Agent, product: str) -> Task:
    return Task(
        description=(
            f"Design a go-to-market strategy for '{product}'. Cover: "
            "a one-sentence positioning statement, top 3 launch channels with rationale, "
            "and a 90-day milestone plan."
        ),
        expected_output=(
            "GTM strategy: positioning statement, 3 prioritized launch channels, "
            "and a 90-day milestone plan."
        ),
        agent=marketer,
    )
