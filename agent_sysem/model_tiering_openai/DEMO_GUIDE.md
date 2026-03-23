# 🚀 Demo Guide - Model Tiering

Visual guide for running and understanding all demos.

## 📺 Live Demo Commands

Copy-paste any of these to see instant results:

```bash
# ⚡ Quick 2-minute intro
uv run demo

# 📚 Full feature walkthrough (15 minutes)
uv run example

# 🔬 Advanced analysis & scenarios (5 minutes)
uv run advanced

# 🏢 Real-world production systems (5 minutes)
uv run realworld

# ⚡ Quick copy-paste examples (3 minutes)
uv run quickref
```

---

## 📊 What Each Demo Shows

### 1️⃣ `uv run demo` - The Starter

**Duration:** 2 minutes | **No API needed** ✓

```
══════════════════════════════════════════════════
Model Tiering Cost Optimization Demo (OpenAI)
══════════════════════════════════════════════════

1. Rule-Based Routing Examples:
   ✓ Translate text → GPT-3.5-turbo
   ✓ Code review → GPT-4-turbo
   ✓ Summarize → GPT-3.5-turbo

2. Monthly Cost Estimation:
   Scenario: 100 requests/day, 500 input, 1000 output tokens

   All GPT-4-Turbo:    $105.00/month
   All GPT-4:          $225.00/month
   All GPT-3.5-turbo:  $5.25/month
   Smart Tiering:      $65.17/month ✓ 37.9% savings!

3. Quick Model Recommendations:
   'What is 2+2?' → GPT-4
   'Write REST API' → GPT-4-turbo
   'Prove theorem' → GPT-4-turbo
```

**When to use:** Getting started, quick overview

---

### 2️⃣ `uv run example` - The Comprehensive

**Duration:** 10-30 minutes | **Optional API** (will skip if not configured)

```
══════════════════════════════════════════════════
Model Tiering - Complete Demo Suite
══════════════════════════════════════════════════

Demo 1: Rule-Based Routing (No API Cost)
  ✓ Different task types → different models

Demo 2: Custom Routing Rules
  ✓ Add domain-specific rules
  ✓ Medical, math, proof tasks

Demo 3: LLM-Assisted Classification
  ✓ Use GPT-3.5-turbo to classify tasks
  ✓ More accurate than rules alone

Demo 4: Hybrid Router with Execution
  ✓ Full integration example
  ✓ Cost tracking

Demo 5: Cost Comparison Scenarios
  ✓ Startup vs Grow vs Enterprise
  ✓ Monthly cost breakdowns

Demo 6: Distribution Optimization
  ✓ How task mix affects costs
  ✓ Conservative vs Balanced vs Aggressive
```

**When to use:** Understanding all strategies

---

### 3️⃣ `uv run advanced` - The Deep Dive

**Duration:** 5 minutes | **No API needed** ✓

```
══════════════════════════════════════════════════
🚀 Advanced Model Tiering Examples
══════════════════════════════════════════════════

Demo 1: Domain-Specific Routing Rules
  Tech Support    → GPT-4
  Legal/Compliance → GPT-4-turbo
  Content Writing → GPT-4
  Data Processing → GPT-3.5-turbo

Demo 2: Industry-Specific Configurations
  🏥 Healthcare
    Patient safety   → GPT-4-turbo
    Admin tasks      → GPT-3.5-turbo

  💰 Finance
    Fraud detection  → GPT-4-turbo
    Financial analysis → GPT-4
    Routine ops      → GPT-3.5-turbo

  🛒 E-Commerce
    Recommendations  → GPT-4
    Customer service → GPT-4
    Simple queries   → GPT-3.5-turbo

Demo 3: Cost Scenario Analysis
  Startup (50/day):        $15.67/month (smart tiering)
  Growing SaaS (500/day):  $156.75/month
  Enterprise (5000/day):   $1,567.50/month

Demo 4: Workload Distribution
  Heavy Complex: 30% simple, 40% med, 30% expert
  Balanced:      60% simple, 30% med, 10% expert
  Heavy Simple:  80% simple, 15% med, 5% expert

Demo 5: Token Size Impact
  Short Tasks (chatbot):     100 in / 200 out
  Medium Tasks (standard):   500 in / 1000 out
  Long-form (content):       2000 in / 3000 out
  Research-heavy:           4000 in / 5000 out

Demo 6: Routing Statistics
  Example output showing:
    Total requests, costs per model, rule hits

Demo 7: Quality Threshold Impact
  Speed Optimized (50):     More cost savings
  Balanced (70):            Good balance
  Quality Focused (85):     High quality
  Premium (95):             Maximum quality
```

**When to use:** Detailed cost analysis, industry-specific setup

---

### 4️⃣ `uv run realworld` - The Practical

**Duration:** 5 minutes | **No API needed** ✓

```
══════════════════════════════════════════════════
🌍 Real-World Model Tiering Use Cases
══════════════════════════════════════════════════

Demo 1: Customer Support Automation
  Input:  Support tickets
  Output: Routed to appropriate model

  TICKET-001: "App keeps crashing"
    → GPT-4-turbo (urgent)
    → Cost: $0.0001

  TICKET-002: "How to reset password?"
    → GPT-3.5-turbo (simple)
    → Cost: $0.0000

  📊 Report: 6 tickets, $0.00 total cost

Demo 2: Content Generation Pipeline
  Input:  Content type + description
  Output: Assigned model + reason

  Twitter Post        → GPT-3.5-turbo (social_media)
  Blog Article        → GPT-4 (technical_writing)
  Product Copy        → GPT-4 (marketing_copy)
  LinkedIn Post       → GPT-4 (blog_article)

  📊 Report: 6 items, 33% GPT-3.5, 67% GPT-4

Demo 3: Data Processing Workflow
  Input:  Data task + file size
  Output: Model + cost

  CSV to JSON (5MB)        → GPT-3.5 ($0.0001)
  Sales analysis (50MB)    → GPT-4 ($0.0015)
  Customer cleaning (100MB) → GPT-4 ($0.0030)

  📊 Report: 5 items processed, $0.0046 total

Demo 4: Code Review System
  Input:  PR + review focus
  Output: Model + priority

  Security review (450 lines) → GPT-4-turbo (urgent)
  Performance (200 lines)     → GPT-4 (high)
  Style check (1500 lines)    → GPT-3.5-turbo (routine)

  📊 Report: 6 PRs, 4,950 lines reviewed
```

**When to use:** Finding production system examples

---

### 5️⃣ `uv run quickref` - The Reference

**Duration:** 3 minutes | **No API needed** ✓

```
══════════════════════════════════════════════════
⚡ Quick Reference Guide - Copy-Paste Examples
══════════════════════════════════════════════════

Example 1: Basic Routing (No API Calls)
  router = RuleBasedRouter()
  decision = router.route("Translate hello to Spanish")
  # → ModelTier.GPT_3_5 (100% confidence)

Example 2: Quick Model Recommendation
  model = get_recommended_model("Write a REST API")
  # → gpt-4

Example 3: Custom Routing Rules
  router.add_rule(RoutingRule(
    name="medical",
    condition=lambda t: "diagnosis" in t.lower(),
    model=ModelTier.GPT_4_TURBO,
    priority=95,
  ))

Example 4: Monthly Cost Estimation
  estimates = CostEstimator.estimate_monthly_cost(
    requests_per_day=1000,
    avg_input_tokens=500,
    avg_output_tokens=1000,
  )
  # Output shows costs for all models + smart tiering

Example 5: Model Comparison Table
  Model          Input    Output   Speed
  GPT-3.5-turbo  $0.50    $1.50    Fast
  GPT-4          $30.00   $60.00   Moderate
  GPT-4-turbo    $10.00   $30.00   Moderate+

Example 6: Task Distribution Analysis
  Simple-Heavy:     80% simple → $432/month
  Balanced:         50% simple → $971/month
  Complex-Heavy:    30% simple → $1,231/month

Example 7: Integration Pattern
  router = HybridRouter()
  result = router.execute(task)
  stats = router.get_stats()

Example 8: Best Practices
  ✓ Use rule-based for known patterns
  ✓ Monitor statistics regularly
  ✓ Start with dynamic escalation
  ✗ Don't use GPT-4-turbo for everything
```

**When to use:** Copy-paste code snippets

---

## 🎯 Pick Your Path

### 🟢 Just Getting Started?
```bash
uv run demo
# 2 minutes, understand the basics
```

### 🟡 Want to Learn Everything?
```bash
uv run demo
uv run example
uv run quickref
# 20 minutes, complete overview
```

### 🔴 Need Production Solution?
```bash
uv run realworld
uv run quickref
# Find matching use case, copy example
```

### 🟣 Doing Cost Analysis?
```bash
uv run advanced
uv run quickref # Example 4 & 6
# Study cost scenarios
```

### 🔵 Building for Your Industry?
```bash
uv run advanced # Demo 2
uv run realworld
# Find industry + matching system
```

---

## 📈 Progressive Learning

```
Day 1: Quick Start (5 minutes)
  └─ uv run demo

Day 2: Understand Features (20 minutes)
  ├─ uv run example
  └─ uv run quickref

Day 3: Deep Analysis (30 minutes)
  ├─ uv run advanced
  └─ uv run realworld

Day 4: Implementation
  ├─ Copy from quickref
  ├─ Customize rules
  └─ Test with your data
```

---

## 💾 Running Everything at Once

```bash
# Run all demos with nice spacing
echo "=== DEMO ===" && uv run demo && \
echo -e "\n=== EXAMPLE ===" && uv run example && \
echo -e "\n=== ADVANCED ===" && uv run advanced && \
echo -e "\n=== REALWORLD ===" && uv run realworld && \
echo -e "\n=== QUICKREF ===" && uv run quickref

# Save to file
uv run demo > all_demos.txt && \
uv run example >> all_demos.txt && \
uv run advanced >> all_demos.txt && \
uv run realworld >> all_demos.txt && \
uv run quickref >> all_demos.txt
```

---

## 🎬 Demo Highlights

### Cost Savings
```
GPT-4-turbo only: $105/month
GPT-4 only:       $225/month
Smart Tiering:    $65.17/month ✓

Savings:
  vs GPT-4-turbo:  37.9%
  vs GPT-4:        71.0% 🎉
```

### Real-World Metrics
```
Customer Support:  6 tickets, $0.00
Content Pipeline:  6 items, 33% simple
Data Processing:   5 tasks, $0.0046
Code Reviews:      6 PRs, 4,950 lines
```

### Model Distribution
```
GPT-3.5-turbo: 70% (70% cost savings)
GPT-4:         25% (moderate complexity)
GPT-4-turbo:   5% (expert only)
```

---

## ✨ Key Takeaways

After demos, you'll understand:

1. ✅ **Routing Strategies**
   - Rule-based (free, fast)
   - LLM classification (accurate)
   - Dynamic escalation (quality assured)
   - Hybrid (production-grade)

2. ✅ **Cost Impact**
   - Model selection matters
   - Distribution is critical
   - Smart tiering saves 37-71%

3. ✅ **Implementation**
   - Easy to integrate
   - Copy-paste ready
   - Highly customizable

4. ✅ **Real Applications**
   - Customer support
   - Content generation
   - Data processing
   - Code review

---

## 🚀 Ready to Start?

```bash
# The quickest way to understand everything:
uv run demo && uv run quickref

# That's it! 5 minutes, full understanding.
```

---

**Pro Tips:**
- 💡 Run demos in order for best understanding
- 📝 Keep quickref open for copy-paste
- 🔍 Use advanced for your specific scenario
- 📊 Monitor with `router.get_stats()`
- 🎯 Customize rules for your domain

**Questions?** Check EXAMPLES.md or COMMANDS.md

Happy demoing! 🎉
