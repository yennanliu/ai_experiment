"""
Real-World Use Cases for Model Tiering Router.

This script demonstrates practical, production-ready scenarios including:
- Customer support automation
- Content generation pipeline
- Data processing workflows
- Code review system
"""

import os
from main import HybridRouter, RuleBasedRouter, RoutingRule, ModelTier


class CustomerSupportSystem:
    """Simulate a customer support automation system."""

    def __init__(self):
        self.router = RuleBasedRouter()
        # Add specific rules for support tickets
        self.router.add_rule(RoutingRule(
            name="urgent_issue",
            condition=lambda t: any(kw in t.lower() for kw in [
                "urgent", "emergency", "critical", "broken", "down", "crash"
            ]),
            model=ModelTier.GPT_4_TURBO,
            priority=100,
            description="Urgent support tickets"
        ))
        self.router.add_rule(RoutingRule(
            name="bug_report",
            condition=lambda t: any(kw in t.lower() for kw in [
                "bug", "error", "not working", "issue", "problem"
            ]),
            model=ModelTier.GPT_4,
            priority=80,
            description="Technical bug reports"
        ))
        self.router.add_rule(RoutingRule(
            name="simple_question",
            condition=lambda t: any(kw in t.lower() for kw in [
                "how to", "can i", "what is", "where is", "when"
            ]),
            model=ModelTier.GPT_3_5,
            priority=30,
            description="Simple FAQ questions"
        ))
        self.tickets_processed = 0
        self.total_cost = 0.0

    def process_ticket(self, ticket_id: str, query: str) -> dict:
        """Process a support ticket."""
        decision = self.router.route(query)
        # Simulate cost
        estimated_cost = self._estimate_cost(decision.model, query)
        self.total_cost += estimated_cost
        self.tickets_processed += 1

        return {
            "ticket_id": ticket_id,
            "query": query[:50] + "..." if len(query) > 50 else query,
            "model": decision.model.value,
            "rule": decision.rule_matched,
            "estimated_cost": f"${estimated_cost:.4f}",
        }

    @staticmethod
    def _estimate_cost(model: ModelTier, query: str) -> float:
        """Rough cost estimation."""
        tokens = len(query.split()) * 1.3
        pricing = {
            ModelTier.GPT_3_5: 0.0005,
            ModelTier.GPT_4: 0.003,
            ModelTier.GPT_4_TURBO: 0.01,
        }
        return (tokens / 1000) * pricing.get(model, 0.001)

    def report(self) -> None:
        """Print processing report."""
        print(f"\n📊 Customer Support Report:")
        print(f"  Total Tickets: {self.tickets_processed}")
        print(f"  Total Cost: ${self.total_cost:.2f}")
        print(f"  Avg Cost per Ticket: ${self.total_cost / self.tickets_processed:.4f}")


class ContentGenerationPipeline:
    """Content generation with intelligent model selection."""

    def __init__(self):
        self.router = RuleBasedRouter()
        self.router.add_rule(RoutingRule(
            name="social_media",
            condition=lambda t: any(kw in t.lower() for kw in [
                "tweet", "social", "instagram", "facebook", "short", "brief"
            ]),
            model=ModelTier.GPT_3_5,
            priority=40,
            description="Short social media content"
        ))
        self.router.add_rule(RoutingRule(
            name="blog_article",
            condition=lambda t: any(kw in t.lower() for kw in [
                "blog", "article", "post", "write about", "explain"
            ]),
            model=ModelTier.GPT_4,
            priority=70,
            description="Blog and article writing"
        ))
        self.router.add_rule(RoutingRule(
            name="marketing_copy",
            condition=lambda t: any(kw in t.lower() for kw in [
                "marketing", "sales", "promote", "ad copy", "pitch"
            ]),
            model=ModelTier.GPT_4,
            priority=75,
            description="Marketing and sales copy"
        ))
        self.router.add_rule(RoutingRule(
            name="technical_writing",
            condition=lambda t: any(kw in t.lower() for kw in [
                "technical", "documentation", "guide", "tutorial", "manual"
            ]),
            model=ModelTier.GPT_4,
            priority=70,
            description="Technical documentation"
        ))
        self.content_items = []

    def generate_content(self, content_type: str, description: str) -> dict:
        """Route content generation to appropriate model."""
        decision = self.router.route(description)
        self.content_items.append({
            "type": content_type,
            "model": decision.model,
            "rule": decision.rule_matched,
        })

        return {
            "content_type": content_type,
            "description": description[:40] + "..." if len(description) > 40 else description,
            "assigned_model": decision.model.value,
            "reason": decision.rule_matched,
        }

    def report(self) -> None:
        """Print content generation report."""
        print(f"\n📝 Content Generation Report:")
        print(f"  Total Items: {len(self.content_items)}")
        model_usage = {}
        for item in self.content_items:
            model = item["model"].value
            model_usage[model] = model_usage.get(model, 0) + 1

        print(f"  Model Distribution:")
        for model, count in sorted(model_usage.items()):
            percentage = (count / len(self.content_items)) * 100
            print(f"    {model}: {count} ({percentage:.1f}%)")


class DataProcessingWorkflow:
    """Data processing and transformation workflow."""

    def __init__(self):
        self.router = RuleBasedRouter()
        self.router.add_rule(RoutingRule(
            name="simple_conversion",
            condition=lambda t: any(kw in t.lower() for kw in [
                "csv", "json", "xml", "convert", "parse"
            ]),
            model=ModelTier.GPT_3_5,
            priority=50,
            description="Simple data conversion"
        ))
        self.router.add_rule(RoutingRule(
            name="data_analysis",
            condition=lambda t: any(kw in t.lower() for kw in [
                "analyze", "trend", "pattern", "statistics", "summary"
            ]),
            model=ModelTier.GPT_4,
            priority=70,
            description="Data analysis and insights"
        ))
        self.router.add_rule(RoutingRule(
            name="data_cleaning",
            condition=lambda t: any(kw in t.lower() for kw in [
                "clean", "normalize", "deduplicate", "validate", "quality"
            ]),
            model=ModelTier.GPT_4,
            priority=65,
            description="Data cleaning and validation"
        ))
        self.processed_items = 0
        self.total_cost = 0.0

    def process_data(self, data_id: str, task_description: str, data_size_mb: float) -> dict:
        """Process data with appropriate model."""
        decision = self.router.route(task_description)
        # Cost increases with data size
        cost_multiplier = min(data_size_mb / 100, 2.0)  # Cap at 2x
        estimated_cost = self._estimate_cost(decision.model) * cost_multiplier
        self.total_cost += estimated_cost
        self.processed_items += 1

        return {
            "data_id": data_id,
            "task": task_description[:35] + "..." if len(task_description) > 35 else task_description,
            "data_size": f"{data_size_mb}MB",
            "model": decision.model.value,
            "cost": f"${estimated_cost:.4f}",
        }

    @staticmethod
    def _estimate_cost(model: ModelTier) -> float:
        """Cost estimation based on model."""
        costs = {
            ModelTier.GPT_3_5: 0.001,
            ModelTier.GPT_4: 0.003,
            ModelTier.GPT_4_TURBO: 0.01,
        }
        return costs.get(model, 0.001)

    def report(self) -> None:
        """Print data processing report."""
        print(f"\n🔄 Data Processing Report:")
        print(f"  Items Processed: {self.processed_items}")
        print(f"  Total Cost: ${self.total_cost:.2f}")
        print(f"  Avg Cost per Item: ${self.total_cost / self.processed_items:.4f}")


class CodeReviewSystem:
    """Automated code review system using model tiering."""

    def __init__(self):
        self.router = RuleBasedRouter()
        self.router.add_rule(RoutingRule(
            name="security_review",
            condition=lambda t: any(kw in t.lower() for kw in [
                "security", "vulnerability", "auth", "encrypt", "injection"
            ]),
            model=ModelTier.GPT_4_TURBO,
            priority=95,
            description="Security-focused code review"
        ))
        self.router.add_rule(RoutingRule(
            name="performance_review",
            condition=lambda t: any(kw in t.lower() for kw in [
                "performance", "optimization", "scalability", "efficiency"
            ]),
            model=ModelTier.GPT_4,
            priority=75,
            description="Performance analysis"
        ))
        self.router.add_rule(RoutingRule(
            name="style_review",
            condition=lambda t: any(kw in t.lower() for kw in [
                "style", "format", "naming", "convention", "lint"
            ]),
            model=ModelTier.GPT_3_5,
            priority=30,
            description="Code style and conventions"
        ))
        self.reviews = []

    def review_code(self, pr_id: str, review_focus: str, code_lines: int) -> dict:
        """Review code with appropriate model."""
        decision = self.router.route(review_focus)
        self.reviews.append({
            "pr_id": pr_id,
            "focus": review_focus,
            "model": decision.model,
            "lines": code_lines,
        })

        return {
            "pr_id": pr_id,
            "focus": review_focus[:30] + "..." if len(review_focus) > 30 else review_focus,
            "code_lines": code_lines,
            "model": decision.model.value,
            "priority": decision.rule_matched,
        }

    def report(self) -> None:
        """Print code review report."""
        print(f"\n💻 Code Review System Report:")
        print(f"  Total Reviews: {len(self.reviews)}")

        total_lines = sum(r["lines"] for r in self.reviews)
        print(f"  Total Lines Reviewed: {total_lines:,}")

        model_usage = {}
        for review in self.reviews:
            model = review["model"].value
            model_usage[model] = model_usage.get(model, 0) + 1

        print(f"  Model Usage:")
        for model, count in sorted(model_usage.items()):
            percentage = (count / len(self.reviews)) * 100
            print(f"    {model}: {count} ({percentage:.1f}%)")


def demo_customer_support() -> None:
    """Demo customer support automation."""
    print("\n" + "=" * 60)
    print("Demo 1: Customer Support Automation")
    print("=" * 60)

    support = CustomerSupportSystem()

    support_tickets = [
        ("TICKET-001", "The app keeps crashing on startup!"),
        ("TICKET-002", "How do I reset my password?"),
        ("TICKET-003", "Bug: Payment button not working on mobile"),
        ("TICKET-004", "Can I upgrade my account?"),
        ("TICKET-005", "Critical: Database connection failure"),
        ("TICKET-006", "What payment methods do you accept?"),
    ]

    print("\n🎫 Processing Support Tickets:")
    for ticket_id, query in support_tickets:
        result = support.process_ticket(ticket_id, query)
        print(f"\n  {result['ticket_id']}: {result['query']}")
        print(f"    → Model: {result['model']}")
        print(f"    → Rule: {result['rule']}")
        print(f"    → Cost: {result['estimated_cost']}")

    support.report()


def demo_content_generation() -> None:
    """Demo content generation pipeline."""
    print("\n" + "=" * 60)
    print("Demo 2: Content Generation Pipeline")
    print("=" * 60)

    pipeline = ContentGenerationPipeline()

    content_requests = [
        ("Twitter Post", "Write a funny tweet about AI"),
        ("Blog Article", "Write a comprehensive guide to machine learning"),
        ("Product Copy", "Create marketing copy for a SaaS product"),
        ("Social Media", "Create an Instagram caption for a vacation photo"),
        ("Technical Doc", "Write a tutorial for integrating our API"),
        ("LinkedIn Post", "Write about the future of AI in business"),
    ]

    print("\n📝 Generating Content:")
    for content_type, description in content_requests:
        result = pipeline.generate_content(content_type, description)
        print(f"\n  {result['content_type']}: {result['description']}")
        print(f"    → Model: {result['assigned_model']}")
        print(f"    → Reason: {result['reason']}")

    pipeline.report()


def demo_data_processing() -> None:
    """Demo data processing workflow."""
    print("\n" + "=" * 60)
    print("Demo 3: Data Processing Workflow")
    print("=" * 60)

    workflow = DataProcessingWorkflow()

    processing_tasks = [
        ("DATA-001", "Convert CSV to JSON format", 5.0),
        ("DATA-002", "Analyze sales trends in quarterly report", 50.0),
        ("DATA-003", "Clean and deduplicate customer database", 100.0),
        ("DATA-004", "Parse XML configuration files", 2.0),
        ("DATA-005", "Generate statistical summary of dataset", 75.0),
    ]

    print("\n🔄 Processing Data:")
    for data_id, task, size in processing_tasks:
        result = workflow.process_data(data_id, task, size)
        print(f"\n  {result['data_id']}: {result['task']}")
        print(f"    → Data Size: {result['data_size']}")
        print(f"    → Model: {result['model']}")
        print(f"    → Cost: {result['cost']}")

    workflow.report()


def demo_code_review() -> None:
    """Demo code review system."""
    print("\n" + "=" * 60)
    print("Demo 4: Automated Code Review System")
    print("=" * 60)

    code_review = CodeReviewSystem()

    review_requests = [
        ("PR-123", "Security review of authentication module", 450),
        ("PR-124", "Performance optimization for database queries", 200),
        ("PR-125", "Code style and formatting check", 1500),
        ("PR-126", "Security audit of API endpoints", 350),
        ("PR-127", "General code review", 800),
        ("PR-128", "Check code standards and naming conventions", 1200),
    ]

    print("\n💻 Reviewing Code:")
    for pr_id, focus, lines in review_requests:
        result = code_review.review_code(pr_id, focus, lines)
        print(f"\n  {result['pr_id']}: {result['focus']}")
        print(f"    → Lines: {result['code_lines']:,}")
        print(f"    → Model: {result['model']}")
        print(f"    → Priority: {result['priority']}")

    code_review.report()


def main() -> None:
    """Run all real-world demos."""
    print("=" * 60)
    print("🌍 Real-World Model Tiering Use Cases")
    print("=" * 60)

    demo_customer_support()
    demo_content_generation()
    demo_data_processing()
    demo_code_review()

    print("\n" + "=" * 60)
    print("✅ All real-world demos complete!")
    print("=" * 60)
    print("\n💡 Key Insights:")
    print("  • Each system routes tasks based on complexity")
    print("  • Cost savings vary by workload type")
    print("  • Rules can be customized per domain")
    print("  • Statistics help identify optimization opportunities")


if __name__ == "__main__":
    main()
