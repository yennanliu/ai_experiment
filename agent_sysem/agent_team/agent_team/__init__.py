"""Specialized Agent Orchestration Framework.

A minimal, elegant implementation of specialized agent patterns including:
- Hub-and-Spoke orchestration
- Pipeline execution
- Parallel with merge
"""

from agent_team.agents import Agent, AgentRole
from agent_team.orchestrator import Orchestrator
from agent_team.router import Router
from agent_team.patterns import OrchestrationPattern

__all__ = ["Agent", "AgentRole", "Orchestrator", "Router", "OrchestrationPattern"]
__version__ = "0.1.0"
