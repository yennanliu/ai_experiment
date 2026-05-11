from crewai import Agent


def researcher(llm) -> Agent:
    return Agent(
        role="Research Assistant",
        goal=(
            "Answer questions accurately, connecting new answers to facts "
            "already established earlier in this session"
        ),
        backstory=(
            "You are a knowledgeable research assistant with an excellent memory. "
            "When answering, you actively connect new questions to prior facts you recall."
        ),
        llm=llm,
        verbose=True,
    )


def synthesizer(llm) -> Agent:
    return Agent(
        role="Knowledge Synthesizer",
        goal="Weave the researcher's findings with recalled context into a cohesive answer",
        backstory=(
            "You excel at merging newly researched facts with prior knowledge. "
            "You explicitly note when an answer builds on something established earlier."
        ),
        llm=llm,
        verbose=True,
    )
