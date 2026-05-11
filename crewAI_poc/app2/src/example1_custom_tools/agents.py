from crewai import Agent
from src.example1_custom_tools.tools import read_text_file, count_words, save_report


def document_analyst(llm) -> Agent:
    return Agent(
        role="Document Analyst",
        goal="Read documents using tools and extract key insights",
        backstory=(
            "You are an expert at reading documents and identifying the most "
            "important information, themes, and writing patterns."
        ),
        tools=[read_text_file, count_words],
        llm=llm,
        verbose=True,
    )


def report_writer(llm) -> Agent:
    return Agent(
        role="Report Writer",
        goal="Write clear, structured analysis reports and save them to disk",
        backstory=(
            "You excel at synthesizing analysis into well-structured markdown "
            "reports, then persisting them for stakeholders."
        ),
        tools=[save_report],
        llm=llm,
        verbose=True,
    )
