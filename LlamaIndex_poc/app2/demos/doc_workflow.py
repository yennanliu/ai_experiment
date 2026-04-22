"""Demo 9: Document Processing Agent Workflow
A three-agent pipeline that processes any simple document end-to-end:

  Extractor   →  reads file, splits into chunks, stores in workflow state
  Summarizer  →  summarizes each chunk, builds structured notes
  QA Agent    →  answers questions about the document interactively

State is shared between agents via the workflow Context (ctx.store),
which is the idiomatic LlamaIndex approach — no globals needed.
"""
import asyncio
import os
from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    AgentStream,
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

# ── context-aware tools (state lives in ctx.store, key = "state") ─────────

async def store_chunks(ctx: Context, chunks: str) -> str:
    """Save extracted document chunks to workflow state."""
    async with ctx.store.edit_state() as s:
        s["state"]["chunks"] = chunks
    return "Chunks stored. Hand off to summarizer_agent to summarize them."


async def get_chunks(ctx: Context) -> str:
    """Retrieve stored document chunks."""
    state = await ctx.store.get("state", default={})
    return state.get("chunks", "No chunks found — extractor_agent must run first.")


async def store_summary(ctx: Context, summary: str) -> str:
    """Save the document summary to workflow state."""
    async with ctx.store.edit_state() as s:
        s["state"]["summary"] = summary
    return "Summary stored. Hand off to qa_agent for interactive Q&A."


async def get_summary(ctx: Context) -> str:
    """Retrieve the stored document summary."""
    state = await ctx.store.get("state", default={})
    return state.get("summary", "No summary found — summarizer_agent must run first.")


# ── helpers ────────────────────────────────────────────────────────────────

def _load_documents(path: str) -> list[Document]:
    """Load documents from a file path using SimpleDirectoryReader."""
    return SimpleDirectoryReader(input_files=[path]).load_data()


def _chunk_documents(docs: list[Document], chunk_size: int = 512) -> list[str]:
    """Split documents into chunks and return as plain text list."""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(docs)
    return [n.get_content() for n in nodes]


# ── workflow builder ───────────────────────────────────────────────────────

def _build_workflow(docs: list[Document]) -> AgentWorkflow:
    llm = OpenAI(model="gpt-4o-mini")

    # Build a search index the QA agent will use
    index = VectorStoreIndex.from_documents(docs)
    search_tool = QueryEngineTool(
        query_engine=index.as_query_engine(similarity_top_k=4),
        metadata=ToolMetadata(
            name="document_search",
            description="Search the processed document to answer specific questions.",
        ),
    )

    # Pre-chunk the document and pass it into the Extractor via initial_state
    chunks = _chunk_documents(docs)
    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks)
    )

    extractor = FunctionAgent(
        name="extractor_agent",
        description="Reads the pre-chunked document text and stores it in workflow state.",
        tools=[FunctionTool.from_defaults(async_fn=store_chunks)],
        llm=llm,
        can_handoff_to=["summarizer_agent"],
        system_prompt=(
            "You are a document extractor. The document has already been split into chunks "
            "and provided to you in the user message. Call store_chunks with ALL the chunk "
            "text exactly as given, then hand off to summarizer_agent."
        ),
    )

    summarizer = FunctionAgent(
        name="summarizer_agent",
        description="Reads document chunks and produces a structured summary.",
        tools=[
            FunctionTool.from_defaults(async_fn=get_chunks),
            FunctionTool.from_defaults(async_fn=store_summary),
        ],
        llm=llm,
        can_handoff_to=["qa_agent"],
        system_prompt=(
            "You are a document summarizer. Call get_chunks to retrieve the document text. "
            "Produce a structured summary with: Overview (2-3 sentences), "
            "Key Topics (bullet list), and Notable Details. "
            "Call store_summary with your summary, then hand off to qa_agent."
        ),
    )

    qa_agent = FunctionAgent(
        name="qa_agent",
        description="Answers user questions about the document.",
        tools=[
            search_tool,
            FunctionTool.from_defaults(async_fn=get_summary),
        ],
        llm=llm,
        system_prompt=(
            "You are a helpful document assistant. First call get_summary to understand "
            "the document. Then use document_search to answer the user's questions "
            "accurately. Be concise and cite relevant parts when helpful."
        ),
    )

    return AgentWorkflow(
        agents=[extractor, summarizer, qa_agent],
        root_agent="extractor_agent",
        initial_state={"chunks": "", "summary": ""},
        verbose=False,
    )


# ── streaming event printer ────────────────────────────────────────────────

async def _run_phase(workflow: AgentWorkflow, msg: str, show_agents: bool = True) -> str:
    """Run one turn of the workflow and stream events. Returns final text."""
    handler = workflow.run(user_msg=msg)
    current_agent = None
    final_response = ""

    async for event in handler.stream_events():
        agent_name = getattr(event, "current_agent_name", None)
        if show_agents and agent_name and agent_name != current_agent:
            current_agent = agent_name
            label = {
                "extractor_agent":  "Extractor",
                "summarizer_agent": "Summarizer",
                "qa_agent":         "Q&A",
            }.get(agent_name, agent_name)
            print(f"\n{'─' * 45}")
            print(f"  Agent: {label}")
            print(f"{'─' * 45}")

        if isinstance(event, AgentStream) and event.delta:
            print(event.delta, end="", flush=True)

        elif isinstance(event, ToolCall):
            print(f"\n  [→ {event.tool_name}]", end="")

        elif isinstance(event, ToolCallResult):
            preview = str(event.tool_output)[:80]
            if len(str(event.tool_output)) > 80:
                preview += "…"
            print(f"  [{preview}]")

        elif isinstance(event, AgentOutput) and event.response.content:
            final_response = event.response.content
            print(f"\n{event.response.content}")

    await handler
    return final_response


# ── entry point ────────────────────────────────────────────────────────────

def run() -> None:
    print("\n[Doc Workflow]  Extractor → Summarizer → Q&A Agent")
    print("Processes any document then opens an interactive Q&A session.\n")

    # File selection
    default = "data/sample.txt"
    path = input(f"Enter file path (or press Enter for '{default}'): ").strip().strip("'\"")
    if not path:
        path = default

    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"\nLoading '{path}' ...")
    try:
        docs = _load_documents(path)
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    total_chars = sum(len(d.get_content()) for d in docs)
    print(f"Loaded {len(docs)} document(s), ~{total_chars:,} characters.")

    # Build workflow and pre-chunk the docs for the extractor
    workflow = _build_workflow(docs)
    chunks = _chunk_documents(docs)
    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks)
    )

    print(f"\nStarting pipeline ({len(chunks)} chunks)...\n")

    # Phase 1: run extractor → summarizer → qa_agent setup
    asyncio.run(_run_phase(
        workflow,
        msg=f"Process this document. Here are the chunks:\n\n{chunks_text}",
    ))

    # Phase 2: interactive Q&A (reuse same workflow instance so state is preserved)
    print(f"\n\n{'=' * 45}")
    print("  Document processed. Ask anything about it.")
    print(f"  ('quit' to exit)")
    print(f"{'=' * 45}\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        asyncio.run(_run_phase(workflow, msg=q, show_agents=False))
        print()
