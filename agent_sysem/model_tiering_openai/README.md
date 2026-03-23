# Model Tiering Cost Optimization

Intelligent model routing for OpenAI API to optimize costs while maintaining quality.

## Overview

This module implements multiple routing strategies to automatically select the appropriate OpenAI model (GPT-3.5-turbo, GPT-4, or GPT-4-turbo) based on task complexity, achieving up to **71% cost savings** compared to using GPT-4 for all requests.

## Installation

```bash
cd model_tiering_openai
uv sync
```

## Configuration

### Setting Up OpenAI API Key

1. Create a `.env` file in the project root:

```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-api-key-here
```

3. The `.env` file is automatically loaded when running the scripts.

**Security Note:** Never commit your `.env` file to version control. It's already added to `.gitignore`.

## Quick Start

```python
from main import HybridRouter, get_recommended_model

# Quick recommendation (no API call)
model = get_recommended_model("Translate hello to Spanish")
print(model)  # ModelTier.HAIKU

# Full routing with execution
router = HybridRouter()
result = router.execute("Summarize this article")
print(f"Used: {result.model_used.name}, Cost: ${result.cost:.6f}")
```

## Routing Strategies

### 1. Rule-Based Routing
Zero-cost routing using keyword matching. Best for well-defined task patterns.

```python
from main import RuleBasedRouter

router = RuleBasedRouter()
decision = router.route("Translate this text to Japanese")
# -> ModelTier.HAIKU (matches translation rule)
```

### 2. LLM-Assisted Classification
Uses GPT-3.5-turbo (~$0.0005/request) to classify task complexity and category.

```python
from main import LLMClassifierRouter

router = LLMClassifierRouter()
decision = router.route("Design a distributed system")
# -> ModelTier.GPT_4_TURBO (classified as expert-level)
```

### 3. Dynamic Quality Escalation
Starts with GPT-3.5-turbo, escalates if quality is insufficient.

```python
from main import DynamicEscalationRouter

router = DynamicEscalationRouter(quality_threshold=70.0)
result = router.execute_with_escalation("Complex reasoning task")
# Automatically escalates if GPT-3.5-turbo's response quality < 70
```

### 4. Hybrid Router (Recommended)
Production-grade router combining all strategies.

```python
from main import HybridRouter

router = HybridRouter(
    use_llm_classification=True,
    enable_escalation=True,
    quality_threshold=70.0,
)
result = router.execute("Your task here")
print(router.get_stats())  # View routing statistics
```

## Custom Rules

Add domain-specific routing rules:

```python
from main import RuleBasedRouter, RoutingRule, ModelTier

router = RuleBasedRouter()

# Medical tasks need highest accuracy
router.add_rule(RoutingRule(
    name="medical",
    condition=lambda t: "diagnosis" in t.lower(),
    model=ModelTier.GPT_4_TURBO,
    priority=95,
))
```

## Cost Estimation

```python
from main import CostEstimator

estimates = CostEstimator.estimate_monthly_cost(
    requests_per_day=1000,
    avg_input_tokens=500,
    avg_output_tokens=1000,
)
print(f"All GPT-4-Turbo: ${estimates['all_gpt4_turbo']:.2f}/month")
print(f"Smart Tiering: ${estimates['smart_tiering']:.2f}/month")
print(f"Savings: {estimates['savings_vs_gpt4_turbo']}%")
```

## Model Selection Criteria

| Complexity | Model | Use Cases |
|------------|-------|-----------|
| Trivial | GPT-3.5-turbo | Simple Q&A, translations, extraction |
| Simple | GPT-3.5-turbo | Formatting, summarization, basic tasks |
| Moderate | GPT-4 | Code generation, analysis, multi-step |
| Complex | GPT-4 | Debugging, review, detailed analysis |
| Expert | GPT-4-turbo | Security, proofs, architecture design |

## Typical Distribution

Based on analysis, ~70% of tasks don't need the strongest model:

- **GPT-3.5-turbo**: 70% (trivial + simple tasks)
- **GPT-4**: 25% (moderate + complex tasks)
- **GPT-4-turbo**: 5% (expert tasks only)

## Reference

Based on: [Model Tiering Cost Optimization Guide](https://yennj12.js.org/yennj12_blog_V4/posts/model-tiering-cost-optimization-guide-zh/)
