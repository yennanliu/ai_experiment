"""Demo 8: Multi-Agent Workflow
Three specialized agents collaborate in a pipeline via AgentWorkflow:

  Researcher  →  finds raw facts from the document index
  Analyst     →  synthesizes findings into insights
  Writer      →  produces a structured final report

Each agent can hand off to the next one. Events are streamed so you can
watch the agents think, call tools, and pass work along in real time.
"""
import asyncio

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    AgentStream,
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

# ── shared state passed between agents via tool return values ──────────────
_findings: list[str] = []
_analysis: str = ""


def record_findings(findings: str) -> str:
    """Store research findings so the Analyst can retrieve them."""
    _findings.clear()
    _findings.append(findings)
    return "Findings recorded. Hand off to analyst_agent for analysis."


def get_findings() -> str:
    """Retrieve the research findings collected by the Researcher."""
    return _findings[0] if _findings else "No findings available yet."


def record_analysis(analysis: str) -> str:
    """Store the Analyst's synthesis so the Writer can retrieve it."""
    global _analysis
    _analysis = analysis
    return "Analysis recorded. Hand off to writer_agent to produce the final report."


def get_analysis() -> str:
    """Retrieve the analysis produced by the Analyst."""
    return _analysis or "No analysis available yet."


# ── build agents ───────────────────────────────────────────────────────────

def _build_workflow() -> AgentWorkflow:
    llm = OpenAI(model="gpt-4o-mini")

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    search_tool = QueryEngineTool(
        query_engine=index.as_query_engine(similarity_top_k=5),
        metadata=ToolMetadata(
            name="document_search",
            description=(
                "Search AI/ML documents covering machine learning, LLMs, "
                "vector databases, RAG, semantic search, embedding models, "
                "AI agents, NLP pipelines, and knowledge graphs."
            ),
        ),
    )

    researcher = FunctionAgent(
        name="researcher_agent",
        description="Searches documents and collects raw facts on a topic.",
        tools=[
            search_tool,
            FunctionTool.from_defaults(fn=record_findings),
        ],
        llm=llm,
        system_prompt=(
            "You are a meticulous researcher. Use document_search to gather "
            "relevant facts about the user's topic. Collect at least 3–5 key points, "
            "then call record_findings with a concise summary of what you found. "
            "After recording, hand off to analyst_agent."
        ),
    )

    analyst = FunctionAgent(
        name="analyst_agent",
        description="Synthesizes research findings into structured insights.",
        tools=[
            FunctionTool.from_defaults(fn=get_findings),
            FunctionTool.from_defaults(fn=record_analysis),
        ],
        llm=llm,
        system_prompt=(
            "You are a critical analyst. Call get_findings to retrieve the research. "
            "Identify patterns, compare concepts, and highlight implications. "
            "Then call record_analysis with your synthesis. "
            "After recording, hand off to writer_agent."
        ),
    )

    writer = FunctionAgent(
        name="writer_agent",
        description="Writes a clear, structured report from the analysis.",
        tools=[
            FunctionTool.from_defaults(fn=get_analysis),
        ],
        llm=llm,
        system_prompt=(
            "You are a professional technical writer. Call get_analysis to retrieve "
            "the analysis, then write a well-structured report with: "
            "an Executive Summary, Key Findings (bullet points), "
            "Analysis & Insights, and a Conclusion. "
            "The report should be clear and suitable for a technical audience."
        ),
    )

    return AgentWorkflow(
        agents=[researcher, analyst, writer],
        root_agent="researcher_agent",
        verbose=False,
    )


# ── streaming runner ───────────────────────────────────────────────────────

async def _run(workflow: AgentWorkflow, topic: str) -> None:
    handler = workflow.run(user_msg=f"Research and report on: {topic}")

    current_agent = None
    async for event in handler.stream_events():
        # Print a header when a new agent takes over
        agent_name = getattr(event, "current_agent_name", None)
        if agent_name and agent_name != current_agent:
            current_agent = agent_name
            label = {
                "researcher_agent": "Researcher",
                "analyst_agent":    "Analyst",
                "writer_agent":     "Writer",
            }.get(agent_name, agent_name)
            print(f"\n{'─' * 50}")
            print(f"  Agent: {label}")
            print(f"{'─' * 50}")

        if isinstance(event, AgentStream) and event.delta:
            print(event.delta, end="", flush=True)

        elif isinstance(event, ToolCall):
            print(f"\n  [tool call] {event.tool_name}({event.tool_kwargs})")

        elif isinstance(event, ToolCallResult):
            # Only print short results to avoid clutter
            result_str = str(event.tool_output)
            preview = result_str[:120] + "…" if len(result_str) > 120 else result_str
            print(f"  [tool result] {preview}")

        elif isinstance(event, AgentOutput) and event.response.content:
            # Final response from each agent
            print(f"\n{event.response.content}")

    await handler


# ── entry point ────────────────────────────────────────────────────────────

def run() -> None:
    print("\n[Multi-Agent Workflow]  Researcher → Analyst → Writer")
    print("Each agent hands off to the next. Events stream in real time.\n")

    topic = input("Enter a research topic (e.g. 'RAG systems vs vector databases'): ").strip()
    if not topic:
        print("No topic provided.")
        return

    print(f"\nStarting pipeline for: '{topic}'\n")
    workflow = _build_workflow()
    asyncio.run(_run(workflow, topic))
    print("\n\nPipeline complete.")
