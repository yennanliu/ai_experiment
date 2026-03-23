"""Command-line interface for agent-team."""

import argparse
import sys

from agent_team.agents import AgentRole
from agent_team.orchestrator import Orchestrator, OrchestratorConfig
from agent_team.patterns import OrchestrationPattern


def main():
    parser = argparse.ArgumentParser(
        description="Specialized Agent Orchestration Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple task with auto-routing
  agent-team "Write a function to parse JSON"

  # Specify pattern
  agent-team --pattern pipeline "Analyze requirements and implement"

  # Specify agents
  agent-team --agents analyst,developer,reviewer "Build a REST API"

  # Full development workflow
  agent-team --workflow dev "Create a user authentication system"
""",
    )

    parser.add_argument("task", help="The task to execute")
    parser.add_argument(
        "--pattern",
        choices=["pipeline", "hub_and_spoke", "parallel_merge"],
        help="Orchestration pattern to use",
    )
    parser.add_argument(
        "--agents",
        help="Comma-separated list of agents (analyst,developer,reviewer,doc_writer)",
    )
    parser.add_argument(
        "--workflow",
        choices=["dev", "review", "docs"],
        help="Predefined workflow",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100_000,
        help="Token budget (default: 100000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    # Configure orchestrator
    config = OrchestratorConfig(
        model=args.model,
        token_budget=args.budget,
    )
    orchestrator = Orchestrator(config)

    # Parse agents if specified
    agents = None
    if args.agents:
        agents = [AgentRole(a.strip()) for a in args.agents.split(",")]

    # Parse pattern if specified
    pattern = None
    if args.pattern:
        pattern = OrchestrationPattern(args.pattern)

    # Execute based on workflow or custom
    if args.workflow == "dev":
        state = orchestrator.analyze_and_implement(args.task)
    elif args.workflow == "review":
        state = orchestrator.review_from_all_angles(args.task)
    elif args.workflow == "docs":
        state = orchestrator.run(
            args.task,
            pattern=OrchestrationPattern.PIPELINE,
            agents=[AgentRole.ANALYST, AgentRole.DOC_WRITER],
        )
    else:
        state = orchestrator.run(args.task, pattern=pattern, agents=agents)

    # Output results
    print(f"\n{'='*60}")
    print(f"Pattern: {state.pattern.value}")
    print(f"Agents: {[r.value for r in state.agents_executed]}")
    print(f"Tokens used: {state.total_tokens:,}")
    print(f"Success: {state.success}")
    print(f"{'='*60}\n")

    if args.verbose:
        for i, response in enumerate(state.responses):
            print(f"\n--- {response.role.value.upper()} ---")
            print(response.content)
            print()
    else:
        # Show final result only
        if state.responses:
            print(state.responses[-1].content)

    if not state.success:
        print(f"\nError: {state.error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
