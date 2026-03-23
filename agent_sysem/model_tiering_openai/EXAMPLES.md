# Examples & Demos - Model Tiering

Complete guide to all available examples and how to run them.

## 📚 Quick Start

```bash
# Run any example with:
uv run <example-name>

# Or use Python directly:
python <filename>
```

## 📋 Available Examples

### 1. **Basic Demo** - Quick Routing & Cost Estimation
```bash
uv run demo
```
**File:** `main.py`

Shows:
- ✨ Rule-based routing examples
- 💰 Monthly cost estimation
- 🎯 Quick model recommendations

**Best for:** Getting started, understanding the basics

---

### 2. **Full Examples Suite** - All Features
```bash
uv run example
```
**File:** `example.py`

Shows:
- 🎯 Rule-based routing (no API cost)
- 🎨 Custom routing rules
- 🧠 LLM-assisted classification
- 🚀 Hybrid router with execution
- 📊 Cost comparison scenarios
- 🔄 Distribution optimization

**Best for:** Learning all features, understanding different strategies

---

### 3. **Advanced Scenarios** - Deep Dive Analysis
```bash
uv run advanced
```
**File:** `examples_advanced.py`

Shows:
- 🏥 Domain-specific routing (healthcare, finance, e-commerce)
- 🛒 Industry-specific configurations
- 📈 Cost scenario analysis (startup → enterprise)
- 💹 Workload distribution impact
- 🎚️ Token size impact on costs
- 📊 Routing statistics tracking
- ⚙️ Quality threshold configuration

**Best for:** Understanding domain applications, detailed cost analysis

---

### 4. **Real-World Use Cases** - Production Scenarios
```bash
uv run realworld
```
**File:** `examples_realworld.py`

Shows:
- 🎫 **Customer Support Automation** - Route support tickets by severity
- 📝 **Content Generation Pipeline** - Assign appropriate models for content types
- 🔄 **Data Processing Workflow** - Handle different data tasks efficiently
- 💻 **Code Review System** - Automated code review with tier-based routing

**Best for:** Understanding practical applications, building your own system

---

### 5. **Quick Reference** - Copy-Paste Examples
```bash
uv run quickref
```
**File:** `examples_quick_reference.py`

Shows:
- 📌 Basic routing (copy-paste ready)
- 🚀 Quick recommendations
- ✏️ Custom rules examples
- 💰 Cost estimation templates
- 🔀 Model comparison table
- 📊 Distribution analysis
- 🛠️ Integration patterns
- ✅ Best practices & tips

**Best for:** Quick copy-paste solutions, reference guide

---

## 🎯 Choose Your Example

| Goal | Run Command | File |
|------|-------------|------|
| **Get started quickly** | `uv run demo` | `main.py` |
| **Learn all features** | `uv run example` | `example.py` |
| **Deep analysis** | `uv run advanced` | `examples_advanced.py` |
| **Real apps** | `uv run realworld` | `examples_realworld.py` |
| **Copy examples** | `uv run quickref` | `examples_quick_reference.py` |

---

## 📊 Example Details

### Demo (main.py)
**Runtime:** ~2 seconds (no API calls)
**Output:** 3 demo sections
- Rule-based routing examples
- Cost estimation
- Quick recommendations

### Example (example.py)
**Runtime:** 5-30 seconds (depending on API availability)
**Output:** 6 demo sections
- Rule-based routing
- Custom rules
- LLM classification
- Hybrid router
- Cost comparison
- Distribution optimization

### Advanced (examples_advanced.py)
**Runtime:** ~5 seconds (no API calls)
**Output:** 7 detailed sections
- Domain-specific routing
- Industry configurations
- Cost scenario analysis
- Workload distribution
- Token impact analysis
- Statistics tracking
- Quality threshold impact

### Real-World (examples_realworld.py)
**Runtime:** ~3 seconds (no API calls)
**Output:** 4 complete systems
- Customer support system
- Content generation pipeline
- Data processing workflow
- Code review system

### Quick Reference (examples_quick_reference.py)
**Runtime:** ~3 seconds (no API calls)
**Output:** 8 reference sections
- Basic routing
- Quick recommendations
- Custom rules
- Cost estimation
- Model comparison
- Distribution analysis
- Integration pattern
- Best practices

---

## 💡 Demo Scenarios

### Scenario 1: Startup (100 requests/day)
```bash
uv run demo
uv run quickref  # Example 4
```
**Key insight:** 71% cost savings with smart tiering ($39/month vs $135/month)

### Scenario 2: Growing Business (1000 requests/day)
```bash
uv run advanced  # Demo 3
```
**Key insight:** $652/month smart tiering vs $2,250/month GPT-4 only

### Scenario 3: Enterprise (5000+ requests/day)
```bash
uv run advanced  # Demo 3 & 4
```
**Key insight:** Distribution optimization can save thousands monthly

### Scenario 4: Specific Domain
```bash
uv run advanced  # Demo 2 (choose your industry)
uv run realworld # Choose matching system
```
**Key insight:** Domain-specific rules provide 30-50% additional savings

---

## 🔧 Customization Examples

### Add Custom Rules
See: `examples_quick_reference.py` - Example 3
```python
router.add_rule(RoutingRule(
    name="my_domain",
    condition=lambda t: "keyword" in t.lower(),
    model=ModelTier.GPT_4,
    priority=70,
))
```

### Integrate in Your Code
See: `examples_quick_reference.py` - Example 7
```python
class MyService:
    def __init__(self):
        self.router = HybridRouter()

    def process(self, task):
        return self.router.execute(task)
```

### Analyze Costs
See: `examples_advanced.py` - Demo 3
```python
estimates = CostEstimator.estimate_monthly_cost(
    requests_per_day=1000,
    model_distribution={...}
)
```

---

## 📈 Learning Path

**Beginner:**
1. Start with `uv run demo` (2 minutes)
2. Read output, understand routing concepts
3. Run `uv run quickref` (5 minutes)

**Intermediate:**
1. Run `uv run example` (10 minutes)
2. Study different strategies
3. Customize rules from quickref

**Advanced:**
1. Run `uv run advanced` (10 minutes)
2. Analyze industry scenarios
3. Study cost impact analysis

**Production:**
1. Run `uv run realworld` (5 minutes)
2. Choose relevant use case
3. Adapt to your system

---

## 🎬 Live Demo Commands

```bash
# Quick 2-min demo
uv run demo

# Full 15-min learning
uv run example && uv run quickref

# Deep dive (30 min)
uv run advanced && uv run realworld

# Complete walkthrough (45 min)
uv run demo && uv run example && uv run advanced && uv run realworld && uv run quickref
```

---

## 🔍 Finding What You Need

### "I want to understand routing"
→ `uv run demo` then `uv run quickref` (Example 1-3)

### "I need cost estimates"
→ `uv run demo` (Section 2) or `uv run quickref` (Example 4)

### "I'm building a support system"
→ `uv run realworld` (Demo 1)

### "I'm generating content"
→ `uv run realworld` (Demo 2)

### "I'm processing data"
→ `uv run realworld` (Demo 3)

### "I'm reviewing code"
→ `uv run realworld` (Demo 4)

### "I need industry-specific rules"
→ `uv run advanced` (Demo 2)

### "I want to optimize distribution"
→ `uv run advanced` (Demo 4)

---

## 📝 Files Summary

| File | Lines | Focus | API Calls |
|------|-------|-------|-----------|
| `main.py` | 784 | Core library & demo | Optional |
| `example.py` | 233 | Feature showcase | Optional |
| `examples_advanced.py` | 420 | Deep analysis | None |
| `examples_realworld.py` | 480 | Practical systems | None |
| `examples_quick_reference.py` | 260 | Quick examples | None |

---

## 🚀 Next Steps

1. ✅ Run a demo (`uv run demo`)
2. ✅ Pick a relevant example
3. ✅ Customize for your use case
4. ✅ Integrate into your application
5. ✅ Monitor with `router.get_stats()`

---

## 💬 Questions?

Each example file includes:
- Clear docstrings
- Inline comments
- Descriptive output

Check the file directly or run with Python debugger:
```bash
python -i examples_advanced.py  # Interactive mode
```

---

**Total Demo Time:** ~1 hour (all examples)
**Time to ROI:** ~15 minutes (understand main concepts)
**Integration Time:** ~30 minutes (customize for your use case)
