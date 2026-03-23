"""
Advanced examples of Model Tiering Router.

This script demonstrates advanced usage patterns including:
- Custom rule configurations
- Domain-specific routing
- Performance benchmarking
- Cost analysis across scenarios
"""

import os
from main import (
    HybridRouter,
    RuleBasedRouter,
    LLMClassifierRouter,
    CostEstimator,
    RoutingRule,
    ModelTier,
)


def demo_domain_specific_routing() -> None:
    """Demonstrate domain-specific routing rules."""
    print("\n" + "=" * 60)
    print("Demo 1: Domain-Specific Routing Rules")
    print("=" * 60)

    router = RuleBasedRouter()

    # Technical Support Domain
    router.add_rule(RoutingRule(
        name="tech_support",
        condition=lambda t: any(kw in t.lower() for kw in [
            "bug", "error", "fix", "troubleshoot", "debug", "issue"
        ]),
        model=ModelTier.GPT_4,
        priority=70,
        description="Technical troubleshooting"
    ))

    # Legal/Compliance Domain
    router.add_rule(RoutingRule(
        name="legal_compliance",
        condition=lambda t: any(kw in t.lower() for kw in [
            "legal", "compliance", "regulation", "gdpr", "contract", "policy"
        ]),
        model=ModelTier.GPT_4_TURBO,
        priority=85,
        description="Legal and compliance tasks"
    ))

    # Content Writing Domain
    router.add_rule(RoutingRule(
        name="content_creation",
        condition=lambda t: any(kw in t.lower() for kw in [
            "write", "article", "blog", "content", "description", "copy"
        ]) and "code" not in t.lower(),
        model=ModelTier.GPT_4,
        priority=60,
        description="Content creation"
    ))

    # Data Processing Domain
    router.add_rule(RoutingRule(
        name="data_processing",
        condition=lambda t: any(kw in t.lower() for kw in [
            "csv", "json", "data", "parse", "convert", "format", "xml"
        ]),
        model=ModelTier.GPT_3_5,
        priority=50,
        description="Data processing and conversion"
    ))

    test_cases = [
        ("Fix the TypeError in my Python code", "tech_support"),
        ("Draft a GDPR-compliant privacy policy", "legal_compliance"),
        ("Write a product description for an e-commerce site", "content_creation"),
        ("Convert this XML data to JSON format", "data_processing"),
    ]

    for task, expected_domain in test_cases:
        decision = router.route(task)
        print(f"\n📌 Task: {task}")
        print(f"   Domain: {expected_domain}")
        print(f"   ✓ Model: {decision.model.value}")
        print(f"   Rule matched: {decision.rule_matched}")


def demo_industry_specific_rules() -> None:
    """Demonstrate industry-specific routing configurations."""
    print("\n" + "=" * 60)
    print("Demo 2: Industry-Specific Configurations")
    print("=" * 60)

    # Healthcare Industry Configuration
    print("\n🏥 Healthcare Configuration:")
    healthcare_router = RuleBasedRouter()
    healthcare_router.add_rule(RoutingRule(
        name="patient_safety",
        condition=lambda t: any(kw in t.lower() for kw in [
            "diagnosis", "treatment", "prescription", "patient", "medication"
        ]),
        model=ModelTier.GPT_4_TURBO,
        priority=100,
        description="Patient safety critical"
    ))
    healthcare_router.add_rule(RoutingRule(
        name="admin_tasks",
        condition=lambda t: any(kw in t.lower() for kw in [
            "appointment", "schedule", "billing", "insurance"
        ]),
        model=ModelTier.GPT_3_5,
        priority=30,
        description="Administrative tasks"
    ))

    healthcare_tasks = [
        "What is the recommended treatment for hypertension?",
        "Schedule patient appointment for next week",
    ]
    for task in healthcare_tasks:
        decision = healthcare_router.route(task)
        print(f"  Task: {task}")
        print(f"    → {decision.model.value}")

    # Finance Industry Configuration
    print("\n💰 Finance Configuration:")
    finance_router = RuleBasedRouter()
    finance_router.add_rule(RoutingRule(
        name="fraud_detection",
        condition=lambda t: any(kw in t.lower() for kw in [
            "fraud", "suspicious", "anomaly", "risk", "compliance", "aml"
        ]),
        model=ModelTier.GPT_4_TURBO,
        priority=100,
        description="Fraud and compliance critical"
    ))
    finance_router.add_rule(RoutingRule(
        name="financial_analysis",
        condition=lambda t: any(kw in t.lower() for kw in [
            "analysis", "forecast", "trend", "portfolio", "investment"
        ]),
        model=ModelTier.GPT_4,
        priority=75,
        description="Financial analysis"
    ))
    finance_router.add_rule(RoutingRule(
        name="routine_operations",
        condition=lambda t: any(kw in t.lower() for kw in [
            "balance", "deposit", "withdraw", "transfer", "account"
        ]),
        model=ModelTier.GPT_3_5,
        priority=20,
        description="Routine operations"
    ))

    finance_tasks = [
        "Analyze investment portfolio performance",
        "Process customer bank transfer",
        "Detect potential fraudulent transactions",
    ]
    for task in finance_tasks:
        decision = finance_router.route(task)
        print(f"  Task: {task}")
        print(f"    → {decision.model.value}")

    # E-commerce Configuration
    print("\n🛒 E-Commerce Configuration:")
    ecommerce_router = RuleBasedRouter()
    ecommerce_router.add_rule(RoutingRule(
        name="product_recommendation",
        condition=lambda t: any(kw in t.lower() for kw in [
            "recommend", "suggestion", "like", "similar", "alternative"
        ]),
        model=ModelTier.GPT_4,
        priority=65,
        description="Product recommendation"
    ))
    ecommerce_router.add_rule(RoutingRule(
        name="customer_service",
        condition=lambda t: any(kw in t.lower() for kw in [
            "help", "support", "problem", "issue", "return", "refund", "complaint"
        ]),
        model=ModelTier.GPT_4,
        priority=60,
        description="Customer support"
    ))
    ecommerce_router.add_rule(RoutingRule(
        name="simple_queries",
        condition=lambda t: any(kw in t.lower() for kw in [
            "hours", "address", "shipping", "price", "available", "stock"
        ]),
        model=ModelTier.GPT_3_5,
        priority=25,
        description="Simple informational queries"
    ))

    ecommerce_tasks = [
        "Recommend similar shoes for this customer",
        "Process return request for damaged item",
        "What are your store hours?",
    ]
    for task in ecommerce_tasks:
        decision = ecommerce_router.route(task)
        print(f"  Task: {task}")
        print(f"    → {decision.model.value}")


def demo_cost_scenario_analysis() -> None:
    """Analyze costs across different business scenarios."""
    print("\n" + "=" * 60)
    print("Demo 3: Cost Scenario Analysis")
    print("=" * 60)

    scenarios = [
        {
            "name": "Startup (MVP Phase)",
            "requests_per_day": 50,
            "input_tokens": 200,
            "output_tokens": 500,
            "description": "High variety, experimental tasks"
        },
        {
            "name": "Growing SaaS",
            "requests_per_day": 500,
            "input_tokens": 400,
            "output_tokens": 1000,
            "description": "Mixed workload, some complex analysis"
        },
        {
            "name": "Established Enterprise",
            "requests_per_day": 5000,
            "input_tokens": 600,
            "output_tokens": 1500,
            "description": "Standardized processes, efficient routing"
        },
        {
            "name": "High-Volume API Service",
            "requests_per_day": 50000,
            "input_tokens": 300,
            "output_tokens": 800,
            "description": "Optimized for speed and cost"
        },
    ]

    for scenario in scenarios:
        print(f"\n📊 {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   {scenario['requests_per_day']:,} requests/day")

        estimates = CostEstimator.estimate_monthly_cost(
            requests_per_day=scenario["requests_per_day"],
            avg_input_tokens=scenario["input_tokens"],
            avg_output_tokens=scenario["output_tokens"],
        )

        monthly_requests = scenario["requests_per_day"] * 30
        daily_cost = estimates['smart_tiering'] / 30

        print(f"\n   Monthly Costs:")
        print(f"     GPT-3.5-turbo only:  ${estimates['all_gpt3_5']:>10,.2f}")
        print(f"     GPT-4 only:          ${estimates['all_gpt4']:>10,.2f}")
        print(f"     GPT-4-Turbo only:    ${estimates['all_gpt4_turbo']:>10,.2f}")
        print(f"     Smart Tiering:       ${estimates['smart_tiering']:>10,.2f} ✓")
        print(f"\n   Daily Cost with Smart Tiering: ${daily_cost:.2f}")
        print(f"   Savings vs GPT-4-Turbo: {estimates['savings_vs_gpt4_turbo']}%")
        print(f"   Savings vs GPT-4: {estimates['savings_vs_gpt4']}%")


def demo_workload_distribution() -> None:
    """Show cost impact of different task distributions."""
    print("\n" + "=" * 60)
    print("Demo 4: Workload Distribution Analysis")
    print("=" * 60)

    distributions = [
        {
            "name": "Heavy on Complex Tasks",
            "description": "High-quality requirements",
            "dist": {ModelTier.GPT_3_5: 0.3, ModelTier.GPT_4: 0.4, ModelTier.GPT_4_TURBO: 0.3}
        },
        {
            "name": "Balanced Workload",
            "description": "Mixed complexity tasks",
            "dist": {ModelTier.GPT_3_5: 0.6, ModelTier.GPT_4: 0.3, ModelTier.GPT_4_TURBO: 0.1}
        },
        {
            "name": "Heavy on Simple Tasks",
            "description": "Mostly routine operations",
            "dist": {ModelTier.GPT_3_5: 0.8, ModelTier.GPT_4: 0.15, ModelTier.GPT_4_TURBO: 0.05}
        },
    ]

    base_requests = 1000

    for config in distributions:
        estimates = CostEstimator.estimate_monthly_cost(
            requests_per_day=base_requests,
            model_distribution=config["dist"],
        )

        print(f"\n📈 {config['name']}")
        print(f"   {config['description']}")
        print(f"   Distribution:")
        print(f"     GPT-3.5: {config['dist'][ModelTier.GPT_3_5]:.0%}")
        print(f"     GPT-4:   {config['dist'][ModelTier.GPT_4]:.0%}")
        print(f"     GPT-4-Turbo: {config['dist'][ModelTier.GPT_4_TURBO]:.0%}")
        print(f"\n   Monthly Cost: ${estimates['smart_tiering']:,.2f}")
        print(f"   vs GPT-4-Turbo only: {estimates['savings_vs_gpt4_turbo']}% savings")
        print(f"   vs GPT-4 only: {estimates['savings_vs_gpt4']}% savings")


def demo_task_complexity_impact() -> None:
    """Show how input/output token sizes affect costs."""
    print("\n" + "=" * 60)
    print("Demo 5: Token Size Impact on Costs")
    print("=" * 60)

    token_scenarios = [
        {
            "name": "Short Tasks (Chatbot)",
            "input": 100,
            "output": 200,
            "description": "Quick Q&A, short responses"
        },
        {
            "name": "Medium Tasks (Standard)",
            "input": 500,
            "output": 1000,
            "description": "Normal analysis, moderate responses"
        },
        {
            "name": "Long-form Content",
            "input": 2000,
            "output": 3000,
            "description": "Document analysis, detailed reports"
        },
        {
            "name": "Research-Heavy",
            "input": 4000,
            "output": 5000,
            "description": "Multi-document analysis, comprehensive outputs"
        },
    ]

    requests_per_day = 100

    for scenario in token_scenarios:
        estimates = CostEstimator.estimate_monthly_cost(
            requests_per_day=requests_per_day,
            avg_input_tokens=scenario["input"],
            avg_output_tokens=scenario["output"],
        )

        print(f"\n💬 {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   Input: {scenario['input']} tokens | Output: {scenario['output']} tokens")
        print(f"   Monthly Cost: ${estimates['smart_tiering']:,.2f}")
        print(f"   Savings: {estimates['savings_vs_gpt4_turbo']}% vs GPT-4-Turbo")


def demo_routing_statistics() -> None:
    """Demonstrate statistics tracking with HybridRouter."""
    print("\n" + "=" * 60)
    print("Demo 6: Routing Statistics (Without API Calls)")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping live demo (OPENAI_API_KEY not set)")
        print("Showing example statistics output:\n")

        example_stats = {
            "total_requests": 1000,
            "model_distribution": {
                "gpt-3.5-turbo": 650,
                "gpt-4": 300,
                "gpt-4-turbo": 50,
            },
            "total_cost": 65.17,
            "escalation_rate": 0.05,
            "rule_hits": {
                "translation": 120,
                "summarization": 150,
                "code_tasks": 200,
                "security_critical": 45,
                "default": 485,
            }
        }

        print("📊 Sample Statistics Output:")
        print(f"  Total Requests: {example_stats['total_requests']}")
        print(f"  Total Cost: ${example_stats['total_cost']:.2f}")
        print(f"  Escalation Rate: {example_stats['escalation_rate']:.1%}")
        print(f"\n  Model Distribution:")
        for model, count in example_stats['model_distribution'].items():
            percentage = (count / example_stats['total_requests']) * 100
            print(f"    {model}: {count} ({percentage:.1f}%)")
        print(f"\n  Top Rule Hits:")
        sorted_rules = sorted(
            example_stats['rule_hits'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for rule, count in sorted_rules[:5]:
            print(f"    {rule}: {count} times")
        return

    print("This demo shows how to track routing decisions")
    print("and calculate statistics for optimization.")


def demo_custom_quality_thresholds() -> None:
    """Show how quality thresholds affect cost and behavior."""
    print("\n" + "=" * 60)
    print("Demo 7: Quality Threshold Impact")
    print("=" * 60)

    thresholds = [
        {
            "threshold": 50.0,
            "name": "Speed Optimized",
            "description": "Accept lower quality, more cost savings"
        },
        {
            "threshold": 70.0,
            "name": "Balanced (Recommended)",
            "description": "Good balance of quality and cost"
        },
        {
            "threshold": 85.0,
            "name": "Quality Focused",
            "description": "High quality, higher cost"
        },
        {
            "threshold": 95.0,
            "name": "Premium",
            "description": "Highest quality, maximum cost"
        },
    ]

    print("\nQuality Threshold Configurations:\n")
    for config in thresholds:
        print(f"🎯 {config['name']}")
        print(f"   Threshold: {config['threshold']:.0f}")
        print(f"   {config['description']}")
        if config['threshold'] <= 70:
            print(f"   → More escalations to GPT-4/turbo needed")
        elif config['threshold'] >= 85:
            print(f"   → Frequent escalations, higher costs")
        else:
            print(f"   → Optimal balance for most use cases")
        print()


def main() -> None:
    """Run all advanced demos."""
    print("=" * 60)
    print("🚀 Advanced Model Tiering Examples")
    print("=" * 60)

    demo_domain_specific_routing()
    demo_industry_specific_rules()
    demo_cost_scenario_analysis()
    demo_workload_distribution()
    demo_task_complexity_impact()
    demo_routing_statistics()
    demo_custom_quality_thresholds()

    print("\n" + "=" * 60)
    print("✅ All advanced demos complete!")
    print("=" * 60)
    print("\n💡 Next Steps:")
    print("  1. Customize routing rules for your domain")
    print("  2. Monitor costs with router.get_stats()")
    print("  3. Adjust quality thresholds based on results")
    print("  4. Track savings in your use case")


if __name__ == "__main__":
    main()
