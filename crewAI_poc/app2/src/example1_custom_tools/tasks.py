from crewai import Task, Agent


def analyze_task(analyst: Agent, filepath: str) -> Task:
    return Task(
        description=(
            f"Read the file at '{filepath}' using the Read Text File tool. "
            "Then use the Count Words tool to gather statistics. "
            "Analyze the content: identify main topics, key arguments, and writing style."
        ),
        expected_output=(
            "Structured analysis: word/sentence statistics, main topics, "
            "key arguments, and writing style observations."
        ),
        agent=analyst,
    )


def write_report_task(writer: Agent, analyze_task: Task) -> Task:
    return Task(
        description=(
            "Using the analyst's findings, write a polished markdown report. "
            "Then call the Save Report tool to persist it. "
            "Use sections: # Document Analysis Report, ## Statistics, "
            "## Main Topics, ## Key Arguments, ## Conclusion."
        ),
        expected_output=(
            "Confirmation message from the Save Report tool, plus the full report content."
        ),
        agent=writer,
        context=[analyze_task],
    )
