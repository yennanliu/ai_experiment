"""Example usage of the agent-team framework."""

from agent_team import Orchestrator, AgentRole, OrchestrationPattern


def main():
    # Initialize orchestrator
    orchestrator = Orchestrator()

    # Example 1: Simple single-agent task
    print("=" * 60)
    print("Example 1: Simple task (auto-routed)")
    print("=" * 60)

    response = orchestrator.simple(
        "Write a Python function to validate email addresses",
        role=AgentRole.DEVELOPER,
    )
    print(f"Tokens: {response.tokens_used}")
    print(response.content[:500] + "..." if len(response.content) > 500 else response.content)

    # Example 2: Pipeline pattern (Analyst -> Developer -> Reviewer)
    print("\n" + "=" * 60)
    print("Example 2: Full development workflow (Pipeline)")
    print("=" * 60)

    state = orchestrator.analyze_and_implement(
        "Create a rate limiter class using the token bucket algorithm"
    )
    print(f"Pattern: {state.pattern.value}")
    print(f"Agents: {[r.value for r in state.agents_executed]}")
    print(f"Total tokens: {state.total_tokens}")
    print(f"\nFinal output:\n{state.responses[-1].content[:500]}...")

    # Example 3: Parallel review
    print("\n" + "=" * 60)
    print("Example 3: Parallel code review")
    print("=" * 60)

    code = '''
def process_user_input(data):
    query = f"SELECT * FROM users WHERE id = {data['id']}"
    return db.execute(query)
'''

    state = orchestrator.review_from_all_angles(code)
    print(f"Pattern: {state.pattern.value}")
    print(f"Agents: {[r.value for r in state.agents_executed]}")
    print(f"Total tokens: {state.total_tokens}")

    # Show remaining budget
    print(f"\n{'='*60}")
    print(f"Remaining token budget: {orchestrator.tokens_remaining:,}")


if __name__ == "__main__":
    main()
