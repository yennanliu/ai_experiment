"""Orchestration patterns: Hub-and-Spoke, Pipeline, Parallel with Merge."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
import asyncio

from agent_team.agents import Agent, AgentRole, AgentResponse


class OrchestrationPattern(Enum):
    """Available orchestration patterns."""
    HUB_AND_SPOKE = "hub_and_spoke"
    PIPELINE = "pipeline"
    PARALLEL_MERGE = "parallel_merge"


@dataclass
class WorkflowState:
    """Tracks workflow execution state."""
    pattern: OrchestrationPattern
    agents_executed: list[AgentRole] = field(default_factory=list)
    responses: list[AgentResponse] = field(default_factory=list)
    total_tokens: int = 0
    success: bool = True
    error: Optional[str] = None

    def add_response(self, response: AgentResponse):
        self.agents_executed.append(response.role)
        self.responses.append(response)
        self.total_tokens += response.tokens_used
        if not response.success:
            self.success = False
            self.error = response.error


from typing import Optional


class BasePattern(ABC):
    """Base class for orchestration patterns."""

    @abstractmethod
    def execute(self, task: str, agents: list[Agent]) -> WorkflowState:
        """Execute the pattern with given agents."""
        pass


class HubAndSpoke(BasePattern):
    """All communications pass through a central orchestrator.

    The orchestrator decomposes the task, distributes to agents,
    and integrates results. No direct peer communication.
    """

    def __init__(self, orchestrator: Optional[Agent] = None):
        self.orchestrator = orchestrator or Agent(role=AgentRole.ORCHESTRATOR)

    def execute(self, task: str, agents: list[Agent]) -> WorkflowState:
        state = WorkflowState(pattern=OrchestrationPattern.HUB_AND_SPOKE)

        # Step 1: Orchestrator decomposes task
        decompose_prompt = f"""Decompose this task into subtasks for these agents: {[a.role.value for a in agents]}

Task: {task}

Provide a numbered list of subtasks with agent assignments."""

        orchestrator_response = self.orchestrator.execute(decompose_prompt)
        state.add_response(orchestrator_response)

        if not orchestrator_response.success:
            return state

        # Step 2: Execute each agent with orchestrator's instructions
        context = orchestrator_response.content
        for agent in agents:
            agent_prompt = f"Based on the task plan, complete your assigned work:\n\n{task}"
            response = agent.execute(agent_prompt, context=context)
            state.add_response(response)
            context = response.content  # Pass to next agent

        # Step 3: Orchestrator integrates results
        integrate_prompt = f"""Integrate these agent results into a coherent response:

Original task: {task}

Agent outputs:
{self._format_outputs(state.responses[1:])}  # Skip orchestrator's decomposition

Provide a unified, coherent final result."""

        final_response = self.orchestrator.execute(integrate_prompt)
        state.add_response(final_response)

        return state

    def _format_outputs(self, responses: list[AgentResponse]) -> str:
        return "\n\n---\n\n".join(
            f"[{r.role.value}]:\n{r.content}" for r in responses
        )


class Pipeline(BasePattern):
    """Sequential execution where each stage's output becomes next stage's input.

    Ideal for workflows with clear dependency chains.
    """

    def execute(self, task: str, agents: list[Agent]) -> WorkflowState:
        state = WorkflowState(pattern=OrchestrationPattern.PIPELINE)
        context = None

        for i, agent in enumerate(agents):
            if i == 0:
                prompt = task
            else:
                prompt = f"Continue working on this task based on previous agent's output:\n\n{task}"

            response = agent.execute(prompt, context=context)
            state.add_response(response)

            if not response.success:
                return state

            # Pass output as context to next agent
            context = response.content

        return state


class ParallelMerge(BasePattern):
    """Execute independent subtasks concurrently, then merge results.

    Dramatically reduces latency for non-dependent work.
    """

    def __init__(self, merge_strategy: str = "summarize"):
        """
        Args:
            merge_strategy: How to merge results - "concatenate", "summarize", or "structured"
        """
        self.merge_strategy = merge_strategy
        self._merger = Agent(role=AgentRole.ORCHESTRATOR)

    def execute(self, task: str, agents: list[Agent]) -> WorkflowState:
        """Synchronous wrapper for async execution."""
        return asyncio.run(self.execute_async(task, agents))

    async def execute_async(self, task: str, agents: list[Agent]) -> WorkflowState:
        state = WorkflowState(pattern=OrchestrationPattern.PARALLEL_MERGE)

        # Execute all agents in parallel
        tasks = [agent.execute_async(task) for agent in agents]
        responses = await asyncio.gather(*tasks)

        for response in responses:
            state.add_response(response)

        if not state.success:
            return state

        # Merge results
        merged = self._merge_results(task, responses)
        state.add_response(merged)

        return state

    def _merge_results(self, task: str, responses: list[AgentResponse]) -> AgentResponse:
        """Merge parallel results based on strategy."""
        if self.merge_strategy == "concatenate":
            content = "\n\n---\n\n".join(
                f"## {r.role.value.title()} Output\n\n{r.content}"
                for r in responses
            )
            return AgentResponse(
                content=content,
                role=AgentRole.ORCHESTRATOR,
                tokens_used=0,
            )

        # Use orchestrator to merge
        merge_prompt = f"""Merge these parallel agent outputs into a coherent response:

Task: {task}

Agent outputs:
{self._format_outputs(responses)}

Strategy: {self.merge_strategy}
Provide a unified result that incorporates insights from all agents."""

        return self._merger.execute(merge_prompt)

    def _format_outputs(self, responses: list[AgentResponse]) -> str:
        return "\n\n---\n\n".join(
            f"[{r.role.value}]:\n{r.content}" for r in responses
        )


def get_pattern(pattern: OrchestrationPattern, **kwargs) -> BasePattern:
    """Factory function to get pattern instance."""
    patterns = {
        OrchestrationPattern.HUB_AND_SPOKE: HubAndSpoke,
        OrchestrationPattern.PIPELINE: Pipeline,
        OrchestrationPattern.PARALLEL_MERGE: ParallelMerge,
    }
    return patterns[pattern](**kwargs)
