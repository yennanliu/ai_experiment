# Missing Topics for Google FDE GenAI Role

Based on gap analysis against the Google Cloud Forward Deployed Engineer, Generative AI JD (Taipei/HK).

## Status

| # | Notebook | Topic | Status |
|---|---|---|---|
| NB08 | `08_ReAct與Tool_Calling.ipynb` | ReAct pattern + function/tool calling | ⬜ TODO |
| NB09 | `09_多智能體系統_LangGraph.ipynb` | Multi-agent systems with LangGraph | ⬜ TODO |
| NB10 | `10_MCP伺服器與企業API整合.ipynb` | MCP server + OAuth-based enterprise API auth | ⬜ TODO |
| NB11 | `11_可觀測性與追蹤.ipynb` | Observability, granular tracing, LLM-native metrics | ⬜ TODO |
| NB12 | `12_GCP_Vertex_AI_RAG部署.ipynb` | Vertex AI RAG Engine + Vector Search + Cloud Run | ⬜ TODO |

---

## Detailed Gap Descriptions

### NB08 — ReAct + Tool Calling
**JD mapping:** Preferred qual — "ReAct, self-reflection"; Responsibility — "agentic workflows"

Missing concepts:
- Function/tool calling with OpenAI / Gemini API
- ReAct loop: Thought → Action → Observation → repeat
- Tool registry (search, calculator, SQL query, API call)
- Self-reflection / critique step before final answer
- Stopping conditions and max-iteration guards

---

### NB09 — Multi-Agent Systems with LangGraph
**JD mapping:** Preferred qual — "LangGraph, CrewAI, Google ADK", "hierarchical delegation"; Responsibility — "multi-agent systems, MCP servers"

Missing concepts:
- LangGraph: StateGraph, nodes, edges, conditional routing
- Planner → Executor → Critic agent pattern
- Hierarchical delegation (orchestrator spawning subagents)
- Human-in-the-loop checkpoints
- Parallel subagent execution
- Shared vs. isolated agent state

---

### NB10 — MCP Server + Enterprise API Integration
**JD mapping:** Preferred qual — "MCP, tool-calling, OAuth-based authentication"; Responsibility — "connect AI products to customers' live infrastructure, APIs, legacy data silos"

Missing concepts:
- Model Context Protocol (MCP) — what it is, server/client architecture
- Building a minimal MCP server in Python
- Exposing tools via MCP (file system, database, REST API)
- OAuth 2.0 flow for enterprise API auth
- Secure credential handling in agentic contexts

---

### NB11 — Observability & Granular Tracing
**JD mapping:** Preferred qual — "LLM-native metrics (tokens/sec, cost-per-request), granular tracing"; Responsibility — "evaluation pipelines and observability frameworks"

Missing concepts:
- Structured logging for each RAG/agent pipeline stage
- Latency breakdown: retrieval / rerank / LLM / total
- tokens/sec throughput and cost-per-request tracking
- OpenTelemetry spans for LLM calls
- LangSmith-style tracing (or manual equivalent)
- Safety/accuracy dashboards for agentic systems
- Alerting on latency regression or cost spikes

---

### NB12 — GCP: Vertex AI RAG + Deployment
**JD mapping:** Core platform requirement — Google Cloud; Minimum qual — "architecting AI systems on cloud platforms"

Missing concepts:
- Vertex AI Gemini API (replacement for OpenAI in all prior notebooks)
- Vertex AI Vector Search (replacement for ChromaDB at scale)
- AlloyDB pgvector for hybrid search
- Vertex AI RAG Engine (managed RAG pipeline)
- Cloud Run deployment of RAG API
- Vertex AI Pipelines for automated re-indexing
- Google ADK (Agent Development Kit) basics

---

## Priority Order

1. **NB08** — Tool calling is the foundation of all agentic work; implement first
2. **NB09** — LangGraph multi-agent; builds on NB08 tools
3. **NB11** — Observability; can be added alongside NB08/09
4. **NB10** — MCP; builds on tool-calling concepts from NB08
5. **NB12** — GCP; most setup-heavy, implement last
