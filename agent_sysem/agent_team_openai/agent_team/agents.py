"""Specialized agents with targeted system prompts."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from openai import OpenAI


class AgentRole(Enum):
    """Specialized agent roles."""
    ANALYST = "analyst"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    DOC_WRITER = "doc_writer"
    ORCHESTRATOR = "orchestrator"


# Targeted system prompts (~2500 tokens each vs ~16000 for monolithic)
SYSTEM_PROMPTS = {
    AgentRole.ANALYST: """You are an Analyst agent. Your role is to:
- Extract and clarify requirements from user requests
- Analyze data and identify patterns
- Break down complex problems into components
- Identify potential risks and dependencies

Output format: Provide structured analysis with clear sections for requirements,
constraints, and recommendations. Be concise but thorough.""",

    AgentRole.DEVELOPER: """You are a Developer agent. Your role is to:
- Write clean, efficient, well-tested code
- Follow best practices and design patterns
- Implement features based on requirements
- Fix bugs and optimize performance

Output format: Provide code with brief explanations. Include error handling
and consider edge cases. Use clear variable names and add minimal comments.""",

    AgentRole.REVIEWER: """You are a Reviewer agent. Your role is to:
- Review code for bugs, security issues, and best practices
- Suggest improvements and optimizations
- Verify requirements are met
- Check for maintainability and readability

Output format: Provide structured feedback with severity levels (critical/major/minor).
Be constructive and specific about issues and fixes.""",

    AgentRole.DOC_WRITER: """You are a Documentation agent. Your role is to:
- Write clear technical documentation
- Create API references and usage examples
- Document architecture decisions
- Write user guides and READMEs

Output format: Use clear headings, code examples, and keep language simple.
Focus on practical usage over theory.""",

    AgentRole.ORCHESTRATOR: """You are an Orchestrator agent. Your role is to:
- Decompose complex tasks into subtasks
- Assign work to specialized agents
- Integrate results from multiple agents
- Ensure coherent final output

Output format: Provide clear task breakdown with dependencies and integration plan.""",
}


@dataclass
class AgentResponse:
    """Response from an agent execution."""
    content: str
    role: AgentRole
    tokens_used: int
    success: bool = True
    error: Optional[str] = None


@dataclass
class Agent:
    """A specialized agent with a specific role."""
    role: AgentRole
    model: str = "gpt-4o"
    max_tokens: int = 4096
    _client: OpenAI = field(default_factory=OpenAI, repr=False)

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPTS[self.role]

    def execute(self, task: str, context: Optional[str] = None) -> AgentResponse:
        """Execute a task with optional context from previous agents."""
        messages = []

        # Add context if provided (minimal context passing)
        if context:
            # Take first 1000 chars as summary per blog post recommendation
            context_summary = context[:1000] if len(context) > 1000 else context
            messages.append({
                "role": "user",
                "content": f"Previous context:\n{context_summary}\n\n---\n\nTask: {task}"
            })
        else:
            messages.append({"role": "user", "content": task})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=messages,
            )

            content = response.choices[0].message.content
            tokens = response.usage.prompt_tokens + response.usage.completion_tokens

            return AgentResponse(
                content=content,
                role=self.role,
                tokens_used=tokens,
                success=True,
            )
        except Exception as e:
            return AgentResponse(
                content="",
                role=self.role,
                tokens_used=0,
                success=False,
                error=str(e),
            )

    async def execute_async(self, task: str, context: Optional[str] = None) -> AgentResponse:
        """Async version for parallel execution."""
        import asyncio
        return await asyncio.to_thread(self.execute, task, context)
