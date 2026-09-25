"""Smart router for task classification and agent selection."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import anthropic
from pydantic import BaseModel

from agent_team.agents import AgentRole


class TaskComplexity(Enum):
    """Task complexity levels for orchestration mode selection."""
    SIMPLE = "simple"      # Single agent can handle
    MEDIUM = "medium"      # 2-3 agents, sequential
    COMPLEX = "complex"    # Multiple agents, parallel possible


class _Classification(BaseModel):
    primary: AgentRole
    complexity: TaskComplexity
    agents: list[AgentRole]  # in execution order
    reasoning: str


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

    def __init__(self, model: str = "claude-sonnet-5"):
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
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=500,
            system="""Classify the task and recommend agents, listing them in execution order. Available roles:
- analyst: requirements, analysis, problem decomposition
- developer: coding, implementation, bug fixes
- reviewer: code review, quality checks, security
- doc_writer: documentation, explanations
The orchestrator role is reserved for the coordinator; don't recommend it.""",
            messages=[{"role": "user", "content": f"Task: {task}"}],
            output_format=_Classification,
        )

        parsed = response.parsed_output
        if parsed is None:
            return RouteResult(
                primary_agent=AgentRole.DEVELOPER,
                complexity=TaskComplexity.SIMPLE,
                suggested_agents=[AgentRole.DEVELOPER],
                reasoning="LLM classification unavailable; defaulted to developer",
            )
        return RouteResult(
            primary_agent=parsed.primary,
            complexity=parsed.complexity,
            suggested_agents=parsed.agents or [parsed.primary],
            reasoning=parsed.reasoning,
        )
