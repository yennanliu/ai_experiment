"""
Model Tiering Cost Optimization for Claude API

This module implements intelligent routing strategies to optimize costs
by routing tasks to appropriate Claude models based on complexity.

Strategies:
1. Rule-Based Routing - Predefined keyword/pattern matching
2. LLM-Assisted Classification - Use Haiku for task classification
3. Dynamic Quality Escalation - Start cheap, upgrade if needed
4. Hybrid Intelligent Router - Combines all approaches

Reference: https://yennj12.js.org/yennj12_blog_V4/posts/model-tiering-cost-optimization-guide-zh/
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import anthropic
from pydantic import BaseModel


class ModelTier(str, Enum):
    """Claude model tiers from cheapest to most capable."""
    HAIKU = "claude-3-5-haiku-latest"
    SONNET = "claude-sonnet-4-20250514"
    OPUS = "claude-opus-4-20250514"


class TaskComplexity(str, Enum):
    """Task complexity levels for routing decisions."""
    TRIVIAL = "trivial"      # Simple, single-step tasks
    SIMPLE = "simple"        # Basic tasks with clear logic
    MODERATE = "moderate"    # Multi-step tasks
    COMPLEX = "complex"      # Tasks requiring deep reasoning
    EXPERT = "expert"        # Tasks requiring expert-level capability


class TaskCategory(str, Enum):
    """Task categories for specialized routing."""
    EXTRACTION = "extraction"      # Data extraction, parsing
    CODE = "code"                  # Code generation, review
    ANALYSIS = "analysis"          # Data analysis, insights
    REASONING = "reasoning"        # Complex logic, math
    TRANSLATION = "translation"    # Language translation
    SUMMARIZATION = "summarization"  # Text summarization
    CREATIVE = "creative"          # Creative writing
    SECURITY = "security"          # Security-related tasks
    GENERAL = "general"            # General purpose


# Pricing per 1M tokens (as of 2025)
MODEL_PRICING = {
    ModelTier.HAIKU: {"input": 0.80, "output": 4.00},
    ModelTier.SONNET: {"input": 3.00, "output": 15.00},
    ModelTier.OPUS: {"input": 15.00, "output": 75.00},
}


@dataclass
class RoutingRule:
    """A single routing rule for task-to-model mapping."""
    name: str
    condition: Callable[[str], bool]
    model: ModelTier
    priority: int = 0
    description: str = ""


@dataclass
class ClassificationResult:
    """Result of LLM-assisted task classification."""
    complexity: TaskComplexity
    category: TaskCategory
    confidence: float
    reasoning: str = ""


@dataclass
class RoutingDecision:
    """Complete routing decision with metadata."""
    model: ModelTier
    strategy: str
    confidence: float
    classification: ClassificationResult | None = None
    rule_matched: str | None = None
    cost_estimate: float = 0.0


@dataclass
class ExecutionResult:
    """Result of model execution."""
    content: str
    model_used: ModelTier
    input_tokens: int
    output_tokens: int
    cost: float
    quality_score: float | None = None
    escalated: bool = False


@dataclass
class RouterStats:
    """Statistics for router performance tracking."""
    total_requests: int = 0
    requests_by_model: dict[ModelTier, int] = field(default_factory=dict)
    total_cost: float = 0.0
    escalations: int = 0
    rule_hits: dict[str, int] = field(default_factory=dict)

    def record(self, result: ExecutionResult, decision: RoutingDecision) -> None:
        """Record an execution result."""
        self.total_requests += 1
        self.requests_by_model[result.model_used] = (
            self.requests_by_model.get(result.model_used, 0) + 1
        )
        self.total_cost += result.cost
        if result.escalated:
            self.escalations += 1
        if decision.rule_matched:
            self.rule_hits[decision.rule_matched] = (
                self.rule_hits.get(decision.rule_matched, 0) + 1
            )

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        return {
            "total_requests": self.total_requests,
            "model_distribution": {
                k.value: v for k, v in self.requests_by_model.items()
            },
            "total_cost": round(self.total_cost, 4),
            "escalation_rate": (
                self.escalations / self.total_requests
                if self.total_requests > 0 else 0
            ),
            "rule_hits": self.rule_hits,
        }


def calculate_cost(
    model: ModelTier,
    input_tokens: int,
    output_tokens: int
) -> float:
    """Calculate cost for a model call."""
    pricing = MODEL_PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


class RoutingStrategy(ABC):
    """Abstract base class for routing strategies."""

    @abstractmethod
    def route(self, task: str) -> RoutingDecision:
        """Determine which model to use for a task."""
        pass


class RuleBasedRouter(RoutingStrategy):
    """
    Rule-based routing using predefined conditions.

    Fastest approach with zero LLM cost for routing decisions.
    """

    def __init__(self) -> None:
        self.rules: list[RoutingRule] = []
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Initialize default routing rules."""
        # Security-critical tasks -> Opus
        self.add_rule(RoutingRule(
            name="security_critical",
            condition=lambda t: any(kw in t.lower() for kw in [
                "security", "vulnerability", "exploit", "authentication",
                "authorization", "encryption", "password", "credential",
                "injection", "xss", "csrf", "sql injection"
            ]),
            model=ModelTier.OPUS,
            priority=100,
            description="Security-critical tasks require highest accuracy"
        ))

        # Complex reasoning -> Opus
        self.add_rule(RoutingRule(
            name="complex_reasoning",
            condition=lambda t: any(kw in t.lower() for kw in [
                "prove", "derive", "mathematical proof", "complex algorithm",
                "architectural design", "system design", "trade-off analysis"
            ]),
            model=ModelTier.OPUS,
            priority=90,
            description="Complex reasoning tasks"
        ))

        # Code generation/review -> Sonnet
        self.add_rule(RoutingRule(
            name="code_tasks",
            condition=lambda t: any(kw in t.lower() for kw in [
                "implement", "refactor", "debug", "code review",
                "write function", "create class", "fix bug"
            ]),
            model=ModelTier.SONNET,
            priority=50,
            description="Code-related tasks"
        ))

        # Analysis tasks -> Sonnet
        self.add_rule(RoutingRule(
            name="analysis_tasks",
            condition=lambda t: any(kw in t.lower() for kw in [
                "analyze", "compare", "evaluate", "assess", "investigate"
            ]),
            model=ModelTier.SONNET,
            priority=40,
            description="Analysis tasks"
        ))

        # Simple translation -> Haiku
        self.add_rule(RoutingRule(
            name="translation",
            condition=lambda t: any(kw in t.lower() for kw in [
                "translate", "翻譯", "翻译", "translation"
            ]),
            model=ModelTier.HAIKU,
            priority=30,
            description="Translation tasks"
        ))

        # Summarization -> Haiku
        self.add_rule(RoutingRule(
            name="summarization",
            condition=lambda t: any(kw in t.lower() for kw in [
                "summarize", "summary", "tldr", "brief overview", "摘要"
            ]),
            model=ModelTier.HAIKU,
            priority=30,
            description="Summarization tasks"
        ))

        # Simple extraction -> Haiku
        self.add_rule(RoutingRule(
            name="extraction",
            condition=lambda t: any(kw in t.lower() for kw in [
                "extract", "parse", "find all", "list the", "get the"
            ]),
            model=ModelTier.HAIKU,
            priority=20,
            description="Data extraction tasks"
        ))

        # Format conversion -> Haiku
        self.add_rule(RoutingRule(
            name="format_conversion",
            condition=lambda t: any(kw in t.lower() for kw in [
                "convert to json", "convert to yaml", "format as",
                "reformat", "convert format"
            ]),
            model=ModelTier.HAIKU,
            priority=20,
            description="Format conversion tasks"
        ))

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def route(self, task: str) -> RoutingDecision:
        """Route task based on rules."""
        for rule in self.rules:
            if rule.condition(task):
                return RoutingDecision(
                    model=rule.model,
                    strategy="rule_based",
                    confidence=1.0,
                    rule_matched=rule.name,
                )

        # Default to Sonnet for unmatched tasks
        return RoutingDecision(
            model=ModelTier.SONNET,
            strategy="rule_based",
            confidence=0.5,
            rule_matched="default",
        )


class LLMClassifierRouter(RoutingStrategy):
    """
    LLM-assisted classification using Haiku for task analysis.

    More accurate than rule-based, with minimal cost (~$0.001/request).
    """

    CLASSIFICATION_PROMPT = """Analyze the following task and classify it.

Task: {task}

Respond with a JSON object containing:
- complexity: one of [trivial, simple, moderate, complex, expert]
- category: one of [extraction, code, analysis, reasoning, translation, summarization, creative, security, general]
- confidence: float between 0 and 1
- reasoning: brief explanation of your classification

Classification criteria:
- trivial: Single-step, obvious tasks (e.g., "What is 2+2?")
- simple: Basic tasks with clear logic (e.g., "Translate hello to Spanish")
- moderate: Multi-step tasks (e.g., "Summarize this article and extract key points")
- complex: Tasks requiring deep reasoning (e.g., "Design a caching strategy")
- expert: Tasks requiring expert-level capability (e.g., "Prove this mathematical theorem")

Respond ONLY with the JSON object, no other text."""

    COMPLEXITY_TO_MODEL = {
        TaskComplexity.TRIVIAL: ModelTier.HAIKU,
        TaskComplexity.SIMPLE: ModelTier.HAIKU,
        TaskComplexity.MODERATE: ModelTier.SONNET,
        TaskComplexity.COMPLEX: ModelTier.SONNET,
        TaskComplexity.EXPERT: ModelTier.OPUS,
    }

    # Categories that always need specific models
    CATEGORY_OVERRIDES = {
        TaskCategory.SECURITY: ModelTier.OPUS,
        TaskCategory.TRANSLATION: ModelTier.HAIKU,
        TaskCategory.EXTRACTION: ModelTier.HAIKU,
    }

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self.client = client or anthropic.Anthropic()
        self._cache: dict[str, ClassificationResult] = {}

    def _classify_task(self, task: str) -> ClassificationResult:
        """Classify task using Haiku."""
        # Check cache first
        cache_key = task[:200]  # Use first 200 chars as key
        if cache_key in self._cache:
            return self._cache[cache_key]

        response = self.client.messages.create(
            model=ModelTier.HAIKU.value,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": self.CLASSIFICATION_PROMPT.format(task=task)
            }]
        )

        try:
            result_text = response.content[0].text
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(result_text)

            classification = ClassificationResult(
                complexity=TaskComplexity(data.get("complexity", "moderate")),
                category=TaskCategory(data.get("category", "general")),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback classification
            classification = ClassificationResult(
                complexity=TaskComplexity.MODERATE,
                category=TaskCategory.GENERAL,
                confidence=0.3,
                reasoning="Failed to parse classification, using fallback",
            )

        self._cache[cache_key] = classification
        return classification

    def route(self, task: str) -> RoutingDecision:
        """Route task based on LLM classification."""
        classification = self._classify_task(task)

        # Check category overrides first
        if classification.category in self.CATEGORY_OVERRIDES:
            model = self.CATEGORY_OVERRIDES[classification.category]
        else:
            model = self.COMPLEXITY_TO_MODEL[classification.complexity]

        return RoutingDecision(
            model=model,
            strategy="llm_classifier",
            confidence=classification.confidence,
            classification=classification,
        )


class DynamicEscalationRouter(RoutingStrategy):
    """
    Dynamic quality escalation: start cheap, upgrade if needed.

    Attempts with Haiku first, escalates based on quality evaluation.
    """

    QUALITY_PROMPT = """Evaluate the quality of this response on a scale of 0-100.

Original task: {task}
Response: {response}

Consider:
- Accuracy and correctness
- Completeness
- Clarity and coherence
- Relevance to the task

Respond with ONLY a number between 0 and 100."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        quality_threshold: float = 70.0,
        max_escalations: int = 2,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.quality_threshold = quality_threshold
        self.max_escalations = max_escalations
        self.escalation_order = [ModelTier.HAIKU, ModelTier.SONNET, ModelTier.OPUS]

    def _evaluate_quality(self, task: str, response: str) -> float:
        """Evaluate response quality using Haiku."""
        eval_response = self.client.messages.create(
            model=ModelTier.HAIKU.value,
            max_tokens=32,
            messages=[{
                "role": "user",
                "content": self.QUALITY_PROMPT.format(task=task, response=response)
            }]
        )

        try:
            score_text = eval_response.content[0].text.strip()
            score = float(re.search(r'\d+', score_text).group())
            return min(100.0, max(0.0, score))
        except (ValueError, AttributeError):
            return 50.0  # Default to middle score

    def _execute_with_model(
        self,
        task: str,
        model: ModelTier,
        max_tokens: int = 4096,
    ) -> tuple[str, int, int]:
        """Execute task with specified model."""
        response = self.client.messages.create(
            model=model.value,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": task}]
        )
        return (
            response.content[0].text,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    def route(self, task: str) -> RoutingDecision:
        """Route starting with cheapest model."""
        # Always start with Haiku
        return RoutingDecision(
            model=ModelTier.HAIKU,
            strategy="dynamic_escalation",
            confidence=0.7,
        )

    def execute_with_escalation(
        self,
        task: str,
        max_tokens: int = 4096,
    ) -> ExecutionResult:
        """Execute task with automatic escalation if quality is low."""
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        escalations = 0

        for model in self.escalation_order:
            if escalations >= self.max_escalations:
                break

            content, input_tokens, output_tokens = self._execute_with_model(
                task, model, max_tokens
            )
            cost = calculate_cost(model, input_tokens, output_tokens)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_cost += cost

            # Evaluate quality (skip for Opus - highest tier)
            if model == ModelTier.OPUS:
                return ExecutionResult(
                    content=content,
                    model_used=model,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cost=total_cost,
                    quality_score=100.0,
                    escalated=escalations > 0,
                )

            quality = self._evaluate_quality(task, content)

            if quality >= self.quality_threshold:
                return ExecutionResult(
                    content=content,
                    model_used=model,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cost=total_cost,
                    quality_score=quality,
                    escalated=escalations > 0,
                )

            escalations += 1

        # Return last result if all escalations used
        return ExecutionResult(
            content=content,
            model_used=model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost=total_cost,
            quality_score=quality,
            escalated=True,
        )


class HybridRouter(RoutingStrategy):
    """
    Hybrid intelligent router combining all strategies.

    Production-grade approach:
    1. Fast rule filtering (zero cost)
    2. Optional LLM classification for ambiguous cases
    3. Quality monitoring with escalation
    4. Statistics tracking for optimization
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        use_llm_classification: bool = True,
        enable_escalation: bool = True,
        quality_threshold: float = 70.0,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.rule_router = RuleBasedRouter()
        self.llm_router = LLMClassifierRouter(client) if use_llm_classification else None
        self.escalation_router = (
            DynamicEscalationRouter(client, quality_threshold)
            if enable_escalation else None
        )
        self.stats = RouterStats()
        self.enable_escalation = enable_escalation

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a custom routing rule."""
        self.rule_router.add_rule(rule)

    def route(self, task: str) -> RoutingDecision:
        """
        Intelligent routing decision.

        1. Try rule-based first (fast, free)
        2. If low confidence, use LLM classification
        3. Return final decision
        """
        # Step 1: Rule-based routing
        rule_decision = self.rule_router.route(task)

        # If high confidence match, use it
        if rule_decision.confidence >= 0.8:
            return rule_decision

        # Step 2: LLM classification for ambiguous cases
        if self.llm_router and rule_decision.rule_matched == "default":
            llm_decision = self.llm_router.route(task)

            # Combine decisions - prefer LLM if confident
            if llm_decision.confidence >= 0.7:
                return llm_decision

        return rule_decision

    def execute(
        self,
        task: str,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
    ) -> ExecutionResult:
        """Execute task with intelligent routing."""
        decision = self.route(task)

        # Use escalation router if enabled and starting with Haiku
        if (
            self.enable_escalation
            and self.escalation_router
            and decision.model == ModelTier.HAIKU
        ):
            result = self.escalation_router.execute_with_escalation(task, max_tokens)
        else:
            # Direct execution
            messages = [{"role": "user", "content": task}]
            response = self.client.messages.create(
                model=decision.model.value,
                max_tokens=max_tokens,
                system=system_prompt or "",
                messages=messages,
            )
            result = ExecutionResult(
                content=response.content[0].text,
                model_used=decision.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost=calculate_cost(
                    decision.model,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                ),
            )

        # Record statistics
        self.stats.record(result, decision)

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics."""
        return self.stats.summary()


class CostEstimator:
    """Utility class for cost estimation and comparison."""

    @staticmethod
    def estimate_monthly_cost(
        requests_per_day: int,
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 1000,
        model_distribution: dict[ModelTier, float] | None = None,
    ) -> dict[str, float]:
        """
        Estimate monthly costs for different scenarios.

        Args:
            requests_per_day: Average daily requests
            avg_input_tokens: Average input tokens per request
            avg_output_tokens: Average output tokens per request
            model_distribution: Distribution of requests by model (sums to 1.0)

        Returns:
            Dict with cost estimates for different strategies
        """
        monthly_requests = requests_per_day * 30

        # All Opus
        opus_cost = calculate_cost(
            ModelTier.OPUS, avg_input_tokens, avg_output_tokens
        ) * monthly_requests

        # All Sonnet
        sonnet_cost = calculate_cost(
            ModelTier.SONNET, avg_input_tokens, avg_output_tokens
        ) * monthly_requests

        # All Haiku
        haiku_cost = calculate_cost(
            ModelTier.HAIKU, avg_input_tokens, avg_output_tokens
        ) * monthly_requests

        # Smart tiering (default: 70% Haiku, 25% Sonnet, 5% Opus)
        if model_distribution is None:
            model_distribution = {
                ModelTier.HAIKU: 0.70,
                ModelTier.SONNET: 0.25,
                ModelTier.OPUS: 0.05,
            }

        tiered_cost = sum(
            calculate_cost(model, avg_input_tokens, avg_output_tokens)
            * monthly_requests
            * ratio
            for model, ratio in model_distribution.items()
        )

        return {
            "all_opus": round(opus_cost, 2),
            "all_sonnet": round(sonnet_cost, 2),
            "all_haiku": round(haiku_cost, 2),
            "smart_tiering": round(tiered_cost, 2),
            "savings_vs_opus": round((1 - tiered_cost / opus_cost) * 100, 1),
            "savings_vs_sonnet": round((1 - tiered_cost / sonnet_cost) * 100, 1),
        }


# Convenience function for quick routing
def get_recommended_model(task: str) -> ModelTier:
    """Quick function to get recommended model for a task."""
    router = RuleBasedRouter()
    decision = router.route(task)
    return decision.model


def main() -> None:
    """Demo usage of model tiering."""
    print("=" * 60)
    print("Model Tiering Cost Optimization Demo")
    print("=" * 60)

    # Demo 1: Rule-based routing
    print("\n1. Rule-Based Routing Examples:")
    print("-" * 40)

    router = RuleBasedRouter()
    test_tasks = [
        "Translate this text to Japanese: Hello world",
        "Summarize this article in 3 bullet points",
        "Implement a binary search algorithm in Python",
        "Review this code for security vulnerabilities",
        "Extract all email addresses from this text",
        "Design a distributed caching system for high availability",
    ]

    for task in test_tasks:
        decision = router.route(task)
        print(f"Task: {task[:50]}...")
        print(f"  -> Model: {decision.model.value}")
        print(f"  -> Rule: {decision.rule_matched}")
        print()

    # Demo 2: Cost estimation
    print("\n2. Monthly Cost Estimation:")
    print("-" * 40)

    estimates = CostEstimator.estimate_monthly_cost(
        requests_per_day=100,
        avg_input_tokens=500,
        avg_output_tokens=1000,
    )

    print(f"Scenario: 100 requests/day, 500 input tokens, 1000 output tokens")
    print(f"  All Opus:      ${estimates['all_opus']:.2f}/month")
    print(f"  All Sonnet:    ${estimates['all_sonnet']:.2f}/month")
    print(f"  All Haiku:     ${estimates['all_haiku']:.2f}/month")
    print(f"  Smart Tiering: ${estimates['smart_tiering']:.2f}/month")
    print(f"  Savings vs Opus: {estimates['savings_vs_opus']}%")
    print(f"  Savings vs Sonnet: {estimates['savings_vs_sonnet']}%")

    # Demo 3: Quick model recommendation
    print("\n3. Quick Model Recommendations:")
    print("-" * 40)

    queries = [
        "What is 2+2?",
        "Write a REST API with authentication",
        "Prove the Pythagorean theorem",
    ]

    for query in queries:
        model = get_recommended_model(query)
        print(f"'{query}' -> {model.value}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
