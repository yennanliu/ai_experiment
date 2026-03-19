"""Smart router for task classification and agent selection."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import anthropic

from agent_team.agents import AgentRole


class TaskComplexity(Enum):
    """Task complexity levels for orchestration mode selection."""
    SIMPLE = "simple"      # Single agent can handle
    MEDIUM = "medium"      # 2-3 agents, sequential
    COMPLEX = "complex"    # Multiple agents, parallel possible


@dataclass
class RouteResult:
    """Result of routing a task."""
    primary_agent: AgentRole
    complexity: TaskComplexity
    suggested_agents: list[AgentRole]
    reasoning: str


# Keyword patterns for quick routing (avoids LLM call for obvious cases)
KEYWORD_PATTERNS = {
    AgentRole.ANALYST: ["analyze", "requirement", "understand", "clarify", "break down", "assess"],
    AgentRole.DEVELOPER: ["implement", "code", "build", "create", "develop", "fix bug", "write function"],
    AgentRole.REVIEWER: ["review", "check", "audit", "verify", "critique", "feedback"],
    AgentRole.DOC_WRITER: ["document", "readme", "explain", "api doc", "guide", "tutorial"],
}


class Router:
    """Routes tasks to appropriate agents based on classification."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._client = anthropic.Anthropic()

    def _keyword_match(self, task: str) -> Optional[AgentRole]:
        """Fast keyword-based routing."""
        task_lower = task.lower()
        for role, keywords in KEYWORD_PATTERNS.items():
            if any(kw in task_lower for kw in keywords):
                return role
        return None

    def route(self, task: str) -> RouteResult:
        """Route a task to appropriate agent(s)."""
        # Try keyword matching first (fast path)
        keyword_match = self._keyword_match(task)

        if keyword_match and len(task) < 100:
            # Simple task with clear keyword
            return RouteResult(
                primary_agent=keyword_match,
                complexity=TaskComplexity.SIMPLE,
                suggested_agents=[keyword_match],
                reasoning=f"Keyword match: {keyword_match.value}",
            )

        # Use LLM for complex classification
        return self._llm_classify(task)

    def _llm_classify(self, task: str) -> RouteResult:
        """Use LLM for nuanced task classification."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=500,
            system="""Classify the task and recommend agents. Available roles:
- analyst: requirements, analysis, problem decomposition
- developer: coding, implementation, bug fixes
- reviewer: code review, quality checks, security
- doc_writer: documentation, explanations

Respond in format:
PRIMARY: <role>
COMPLEXITY: simple|medium|complex
AGENTS: <comma-separated roles in execution order>
REASONING: <brief explanation>""",
            messages=[{"role": "user", "content": f"Task: {task}"}],
        )

        return self._parse_classification(response.content[0].text)

    def _parse_classification(self, response: str) -> RouteResult:
        """Parse LLM classification response."""
        lines = response.strip().split("\n")
        result = {}

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip().upper()] = value.strip()

        primary = AgentRole(result.get("PRIMARY", "developer").lower())
        complexity = TaskComplexity(result.get("COMPLEXITY", "simple").lower())

        agents_str = result.get("AGENTS", primary.value)
        agents = [AgentRole(a.strip().lower()) for a in agents_str.split(",")]

        return RouteResult(
            primary_agent=primary,
            complexity=complexity,
            suggested_agents=agents,
            reasoning=result.get("REASONING", "LLM classification"),
        )
