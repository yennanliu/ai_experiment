"""
Example usage of Model Tiering Router.

This script demonstrates practical usage patterns for the model tiering system.
Requires ANTHROPIC_API_KEY environment variable to be set.
"""

import os
from main import (
    HybridRouter,
    RuleBasedRouter,
    LLMClassifierRouter,
    DynamicEscalationRouter,
    CostEstimator,
    RoutingRule,
    ModelTier,
)


def demo_rule_based_routing() -> None:
    """Demonstrate rule-based routing (no API calls)."""
    print("\n" + "=" * 60)
    print("Demo 1: Rule-Based Routing (No API Cost)")
    print("=" * 60)

    router = RuleBasedRouter()

    tasks = [
        ("Translate 'hello' to French", "Simple translation"),
        ("Summarize this document", "Summarization task"),
        ("Write a Python function to sort a list", "Code generation"),
        ("Find security vulnerabilities in this code", "Security task"),
        ("What is the capital of France?", "General question"),
        ("Design a microservices architecture", "Complex design"),
    ]

    for task, description in tasks:
        decision = router.route(task)
        print(f"\n{description}:")
        print(f"  Task: {task}")
        print(f"  Model: {decision.model.name}")
        print(f"  Rule: {decision.rule_matched}")


def demo_custom_rules() -> None:
    """Demonstrate adding custom routing rules."""
    print("\n" + "=" * 60)
    print("Demo 2: Custom Routing Rules")
    print("=" * 60)

    router = RuleBasedRouter()

    # Add custom rule for medical tasks
    router.add_rule(RoutingRule(
        name="medical_tasks",
        condition=lambda t: any(kw in t.lower() for kw in [
            "diagnosis", "symptoms", "medical", "patient", "treatment"
        ]),
        model=ModelTier.OPUS,
        priority=95,
        description="Medical tasks require high accuracy"
    ))

    # Add custom rule for math homework
    router.add_rule(RoutingRule(
        name="simple_math",
        condition=lambda t: any(kw in t.lower() for kw in [
            "calculate", "what is", "solve for x", "equation"
        ]) and "prove" not in t.lower(),
        model=ModelTier.HAIKU,
        priority=25,
        description="Simple math calculations"
    ))

    tasks = [
        "What are the symptoms of flu?",
        "Calculate 15% of 280",
        "Prove the fundamental theorem of calculus",
    ]

    for task in tasks:
        decision = router.route(task)
        print(f"\nTask: {task}")
        print(f"  Model: {decision.model.name}")
        print(f"  Rule: {decision.rule_matched}")


def demo_llm_classification() -> None:
    """Demonstrate LLM-assisted classification."""
    print("\n" + "=" * 60)
    print("Demo 3: LLM-Assisted Classification")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Skipped: ANTHROPIC_API_KEY not set")
        return

    router = LLMClassifierRouter()

    tasks = [
        "Explain how photosynthesis works in simple terms",
        "Write a comprehensive guide to building a REST API",
        "What color is the sky?",
    ]

    for task in tasks:
        decision = router.route(task)
        print(f"\nTask: {task}")
        print(f"  Model: {decision.model.name}")
        print(f"  Complexity: {decision.classification.complexity.value}")
        print(f"  Category: {decision.classification.category.value}")
        print(f"  Confidence: {decision.classification.confidence:.2f}")
        print(f"  Reasoning: {decision.classification.reasoning}")


def demo_hybrid_router_with_execution() -> None:
    """Demonstrate full hybrid routing with execution."""
    print("\n" + "=" * 60)
    print("Demo 4: Hybrid Router with Execution")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Skipped: ANTHROPIC_API_KEY not set")
        return

    router = HybridRouter(
        use_llm_classification=True,
        enable_escalation=False,  # Disable for demo to reduce API calls
    )

    tasks = [
        "Translate 'good morning' to Spanish",
        "Write a haiku about programming",
    ]

    for task in tasks:
        print(f"\nExecuting: {task}")
        result = router.execute(task, max_tokens=256)
        print(f"  Model used: {result.model_used.name}")
        print(f"  Tokens: {result.input_tokens} in, {result.output_tokens} out")
        print(f"  Cost: ${result.cost:.6f}")
        print(f"  Response: {result.content[:100]}...")

    # Print statistics
    print("\nRouter Statistics:")
    stats = router.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_cost_comparison() -> None:
    """Demonstrate cost comparison for different scenarios."""
    print("\n" + "=" * 60)
    print("Demo 5: Cost Comparison Scenarios")
    print("=" * 60)

    scenarios = [
        {"name": "Startup (Low Volume)", "requests_per_day": 50},
        {"name": "Growing Business", "requests_per_day": 500},
        {"name": "Enterprise (High Volume)", "requests_per_day": 5000},
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']} ({scenario['requests_per_day']} req/day):")
        estimates = CostEstimator.estimate_monthly_cost(
            requests_per_day=scenario["requests_per_day"],
            avg_input_tokens=500,
            avg_output_tokens=1000,
        )
        print(f"  All Opus:      ${estimates['all_opus']:>10,.2f}/month")
        print(f"  All Sonnet:    ${estimates['all_sonnet']:>10,.2f}/month")
        print(f"  Smart Tiering: ${estimates['smart_tiering']:>10,.2f}/month")
        print(f"  Monthly Savings: ${estimates['all_opus'] - estimates['smart_tiering']:>,.2f}")


def demo_distribution_optimization() -> None:
    """Demonstrate optimizing model distribution."""
    print("\n" + "=" * 60)
    print("Demo 6: Distribution Optimization")
    print("=" * 60)

    distributions = [
        {
            "name": "Conservative (More Opus)",
            "dist": {ModelTier.HAIKU: 0.5, ModelTier.SONNET: 0.3, ModelTier.OPUS: 0.2}
        },
        {
            "name": "Balanced",
            "dist": {ModelTier.HAIKU: 0.6, ModelTier.SONNET: 0.3, ModelTier.OPUS: 0.1}
        },
        {
            "name": "Aggressive (More Haiku)",
            "dist": {ModelTier.HAIKU: 0.8, ModelTier.SONNET: 0.15, ModelTier.OPUS: 0.05}
        },
    ]

    for config in distributions:
        estimates = CostEstimator.estimate_monthly_cost(
            requests_per_day=1000,
            model_distribution=config["dist"],
        )
        print(f"\n{config['name']}:")
        print(f"  Distribution: H={config['dist'][ModelTier.HAIKU]:.0%}, "
              f"S={config['dist'][ModelTier.SONNET]:.0%}, "
              f"O={config['dist'][ModelTier.OPUS]:.0%}")
        print(f"  Monthly Cost: ${estimates['smart_tiering']:,.2f}")
        print(f"  Savings vs Opus: {estimates['savings_vs_opus']}%")


def main() -> None:
    """Run all demos."""
    print("=" * 60)
    print("Model Tiering - Complete Demo Suite")
    print("=" * 60)

    # Always run (no API needed)
    demo_rule_based_routing()
    demo_custom_rules()
    demo_cost_comparison()
    demo_distribution_optimization()

    # Requires API key
    demo_llm_classification()
    demo_hybrid_router_with_execution()

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
