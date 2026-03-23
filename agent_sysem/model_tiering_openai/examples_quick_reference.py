"""
Quick Reference Guide - Model Tiering Examples.

Copy-paste ready examples for common scenarios.
"""

from main import HybridRouter, RuleBasedRouter, RoutingRule, ModelTier, CostEstimator


def example_1_basic_routing():
    """Example 1: Basic rule-based routing."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Routing (No API Calls)")
    print("=" * 60)

    router = RuleBasedRouter()

    tasks = [
        "Translate this to Spanish",
        "Write Python code to sort a list",
        "Summarize this article",
        "Check for security vulnerabilities",
    ]

    for task in tasks:
        decision = router.route(task)
        print(f"\n📋 Task: {task}")
        print(f"   → Model: {decision.model.value}")
        print(f"   → Confidence: {decision.confidence:.1%}")


def example_2_quick_recommendation():
    """Example 2: Get quick model recommendation."""
    print("\n" + "=" * 60)
    print("Example 2: Quick Model Recommendation")
    print("=" * 60)

    from main import get_recommended_model

    queries = [
        "What is 2+2?",
        "Explain quantum computing",
        "Write a REST API",
        "Prove this theorem",
    ]

    for query in queries:
        model = get_recommended_model(query)
        print(f"\n❓ Query: {query}")
        print(f"   → Recommended: {model.value}")


def example_3_custom_rules():
    """Example 3: Add custom routing rules."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Routing Rules")
    print("=" * 60)

    router = RuleBasedRouter()

    # Add custom rules for specific domains
    router.add_rule(RoutingRule(
        name="email_tasks",
        condition=lambda t: "email" in t.lower() or "mail" in t.lower(),
        model=ModelTier.GPT_3_5,
        priority=40,
        description="Email composition and replies"
    ))

    router.add_rule(RoutingRule(
        name="research",
        condition=lambda t: "research" in t.lower() or "study" in t.lower(),
        model=ModelTier.GPT_4,
        priority=70,
        description="Research and analysis"
    ))

    test_cases = [
        "Draft an email to the client",
        "Research the history of AI",
        "Help me write an email",
    ]

    for task in test_cases:
        decision = router.route(task)
        print(f"\n✉️  Task: {task}")
        print(f"   → Model: {decision.model.value}")
        print(f"   → Rule: {decision.rule_matched}")


def example_4_cost_estimation():
    """Example 4: Estimate monthly costs."""
    print("\n" + "=" * 60)
    print("Example 4: Monthly Cost Estimation")
    print("=" * 60)

    scenarios = [
        {"name": "Startup", "requests": 100, "in_tokens": 300, "out_tokens": 600},
        {"name": "Scale-up", "requests": 1000, "in_tokens": 500, "out_tokens": 1000},
        {"name": "Enterprise", "requests": 5000, "in_tokens": 600, "out_tokens": 1200},
    ]

    for scenario in scenarios:
        estimates = CostEstimator.estimate_monthly_cost(
            requests_per_day=scenario["requests"],
            avg_input_tokens=scenario["in_tokens"],
            avg_output_tokens=scenario["out_tokens"],
        )

        print(f"\n💰 {scenario['name']} ({scenario['requests']}/day requests):")
        print(f"   GPT-3.5-turbo only: ${estimates['all_gpt3_5']:>8.2f}/month")
        print(f"   GPT-4 only:         ${estimates['all_gpt4']:>8.2f}/month")
        print(f"   Smart Tiering:      ${estimates['smart_tiering']:>8.2f}/month ✓")
        print(f"   Monthly Savings:    {estimates['savings_vs_gpt4']:.0f}% vs GPT-4")


def example_5_model_comparison():
    """Example 5: Compare models side-by-side."""
    print("\n" + "=" * 60)
    print("Example 5: Model Comparison")
    print("=" * 60)

    print("\n📊 Model Comparison (per 1M tokens):")
    print("\n{:<20} {:<12} {:<12} {:<10}".format("Model", "Input", "Output", "Speed"))
    print("-" * 60)

    models = [
        ("GPT-3.5-turbo", "$0.50", "$1.50", "Fast"),
        ("GPT-4", "$30.00", "$60.00", "Moderate"),
        ("GPT-4-turbo", "$10.00", "$30.00", "Moderate+"),
    ]

    for name, input_price, output_price, speed in models:
        print("{:<20} {:<12} {:<12} {:<10}".format(name, input_price, output_price, speed))

    print("\n✨ Use Cases:")
    print("  GPT-3.5-turbo: Simple tasks, translations, summaries, extractions")
    print("  GPT-4:         Code generation, analysis, content writing")
    print("  GPT-4-turbo:   Security, complex reasoning, expert tasks")


def example_6_distribution_analysis():
    """Example 6: Analyze different task distributions."""
    print("\n" + "=" * 60)
    print("Example 6: Task Distribution Analysis")
    print("=" * 60)

    distributions = [
        {
            "name": "Simple-Heavy",
            "desc": "80% simple, 15% medium, 5% complex",
            "dist": {ModelTier.GPT_3_5: 0.8, ModelTier.GPT_4: 0.15, ModelTier.GPT_4_TURBO: 0.05}
        },
        {
            "name": "Balanced",
            "desc": "50% simple, 35% medium, 15% complex",
            "dist": {ModelTier.GPT_3_5: 0.5, ModelTier.GPT_4: 0.35, ModelTier.GPT_4_TURBO: 0.15}
        },
        {
            "name": "Complex-Heavy",
            "desc": "30% simple, 40% medium, 30% complex",
            "dist": {ModelTier.GPT_3_5: 0.3, ModelTier.GPT_4: 0.4, ModelTier.GPT_4_TURBO: 0.3}
        },
    ]

    for config in distributions:
        estimates = CostEstimator.estimate_monthly_cost(
            requests_per_day=1000,
            model_distribution=config["dist"],
        )

        print(f"\n🎯 {config['name']}")
        print(f"   {config['desc']}")
        print(f"   Monthly Cost: ${estimates['smart_tiering']:,.2f}")
        print(f"   Savings: {estimates['savings_vs_gpt4']}% vs GPT-4")


def example_7_integration_pattern():
    """Example 7: Integration pattern with HybridRouter."""
    print("\n" + "=" * 60)
    print("Example 7: Integration Pattern")
    print("=" * 60)

    print("""
class MyService:
    def __init__(self):
        self.router = HybridRouter(
            use_llm_classification=True,
            enable_escalation=True,
            quality_threshold=70.0,
        )

    def process_task(self, task: str):
        # Route and execute
        result = self.router.execute(task, max_tokens=1024)

        # Track costs
        print(f"Model: {result.model_used.name}")
        print(f"Cost: ${result.cost:.6f}")
        print(f"Response: {result.content}")

        # Get statistics
        stats = self.router.get_stats()
        print(f"Total Cost: ${stats['total_cost']:.2f}")

# Usage:
service = MyService()
service.process_task("Translate hello to Spanish")
    """)


def example_8_best_practices():
    """Example 8: Best practices and tips."""
    print("\n" + "=" * 60)
    print("Example 8: Best Practices")
    print("=" * 60)

    tips = [
        ("Routing Strategy", [
            "✓ Use rule-based for known patterns (zero cost)",
            "✓ Add custom rules for your domain",
            "✓ Use LLM classification for ambiguous tasks",
            "✗ Don't use GPT-4-turbo for everything",
        ]),
        ("Cost Optimization", [
            "✓ Monitor router statistics regularly",
            "✓ Adjust quality thresholds based on results",
            "✓ Analyze task distribution in your workload",
            "✗ Don't sacrifice quality for cost alone",
        ]),
        ("Quality Control", [
            "✓ Start with dynamic escalation enabled",
            "✓ Set quality_threshold around 70",
            "✓ Review escalation rate in statistics",
            "✗ Don't escalate everything to GPT-4-turbo",
        ]),
        ("Monitoring", [
            "✓ Track costs per model and rule",
            "✓ Monitor escalation rate",
            "✓ Review rule hit rates",
            "✗ Don't ignore cost growth patterns",
        ]),
    ]

    for category, items in tips:
        print(f"\n📌 {category}:")
        for item in items:
            print(f"  {item}")


def main():
    """Run all quick reference examples."""
    print("=" * 60)
    print("⚡ Quick Reference Guide - Copy-Paste Examples")
    print("=" * 60)

    example_1_basic_routing()
    example_2_quick_recommendation()
    example_3_custom_rules()
    example_4_cost_estimation()
    example_5_model_comparison()
    example_6_distribution_analysis()
    example_7_integration_pattern()
    example_8_best_practices()

    print("\n" + "=" * 60)
    print("✅ Quick Reference Complete!")
    print("=" * 60)
    print("\n📚 Next Steps:")
    print("  1. Copy examples that match your use case")
    print("  2. Customize rules for your domain")
    print("  3. Run with: uv run demo, uv run example, etc.")
    print("  4. Monitor with: router.get_stats()")


if __name__ == "__main__":
    main()
