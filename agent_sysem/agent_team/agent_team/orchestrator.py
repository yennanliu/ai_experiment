"""Main orchestrator that coordinates the agent team."""

from dataclasses import dataclass, field
from typing import Optional

from agent_team.agents import Agent, AgentRole, AgentResponse
from agent_team.router import Router, RouteResult, TaskComplexity
from agent_team.patterns import (
    OrchestrationPattern,
    WorkflowState,
    get_pattern,
    HubAndSpoke,
    Pipeline,
    ParallelMerge,
)


@dataclass
class TokenBudget:
    """Manages token consumption across agents."""
    total_budget: int = 100_000
    reserve_ratio: float = 0.1  # Reserve for coordination overhead
    consumed: int = 0

    @property
    def available(self) -> int:
        reserve = int(self.total_budget * self.reserve_ratio)
        return self.total_budget - self.consumed - reserve

    def can_afford(self, estimated_tokens: int) -> bool:
        return self.available >= estimated_tokens

    def consume(self, tokens: int):
        self.consumed += tokens


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    model: str = "claude-sonnet-4-20250514"
    token_budget: int = 100_000
    max_retries: int = 2
    auto_select_pattern: bool = True


class Orchestrator:
    """Coordinates specialized agents to complete tasks.

    Features:
    - Smart routing to select appropriate agents
    - Automatic pattern selection based on task complexity
    - Token budget management
    - Error handling with retries
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.router = Router(model=self.config.model)
        self.budget = TokenBudget(total_budget=self.config.token_budget)

        # Pre-initialize agent pool
        self._agents = {
            role: Agent(role=role, model=self.config.model)
            for role in AgentRole
        }

    def run(
        self,
        task: str,
        pattern: Optional[OrchestrationPattern] = None,
        agents: Optional[list[AgentRole]] = None,
    ) -> WorkflowState:
        """Execute a task with the agent team.

        Args:
            task: The task to complete
            pattern: Orchestration pattern (auto-selected if None)
            agents: Specific agents to use (auto-selected if None)

        Returns:
            WorkflowState with all responses and metrics
        """
        # Route task to determine agents and complexity
        route_result = self.router.route(task)

        # Select agents
        agent_roles = agents or route_result.suggested_agents
        selected_agents = [self._agents[role] for role in agent_roles]

        # Select pattern based on complexity
        if pattern is None and self.config.auto_select_pattern:
            pattern = self._select_pattern(route_result)
        elif pattern is None:
            pattern = OrchestrationPattern.PIPELINE

        # Execute with selected pattern
        executor = get_pattern(pattern)
        state = executor.execute(task, selected_agents)

        # Update budget
        self.budget.consume(state.total_tokens)

        return state

    def _select_pattern(self, route: RouteResult) -> OrchestrationPattern:
        """Auto-select pattern based on task complexity."""
        match route.complexity:
            case TaskComplexity.SIMPLE:
                return OrchestrationPattern.PIPELINE
            case TaskComplexity.MEDIUM:
                return OrchestrationPattern.PIPELINE
            case TaskComplexity.COMPLEX:
                # Use parallel for multiple independent agents
                if len(route.suggested_agents) >= 3:
                    return OrchestrationPattern.PARALLEL_MERGE
                return OrchestrationPattern.HUB_AND_SPOKE

    def simple(self, task: str, role: AgentRole = AgentRole.DEVELOPER) -> AgentResponse:
        """Execute a simple single-agent task.

        Convenience method for straightforward tasks.
        """
        agent = self._agents[role]
        response = agent.execute(task)
        self.budget.consume(response.tokens_used)
        return response

    def analyze_and_implement(self, task: str) -> WorkflowState:
        """Common pattern: Analyst -> Developer -> Reviewer."""
        return self.run(
            task,
            pattern=OrchestrationPattern.PIPELINE,
            agents=[AgentRole.ANALYST, AgentRole.DEVELOPER, AgentRole.REVIEWER],
        )

    def review_from_all_angles(self, code: str) -> WorkflowState:
        """Parallel review: Multiple reviewers check simultaneously."""
        # Create multiple reviewer instances with different focus
        task = f"Review this code:\n\n{code}"
        return self.run(
            task,
            pattern=OrchestrationPattern.PARALLEL_MERGE,
            agents=[AgentRole.REVIEWER, AgentRole.DEVELOPER, AgentRole.ANALYST],
        )

    @property
    def tokens_remaining(self) -> int:
        return self.budget.available
