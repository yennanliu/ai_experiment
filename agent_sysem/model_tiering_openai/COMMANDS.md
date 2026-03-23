# Command Reference - Model Tiering

Quick reference for all available commands.

## 🚀 Run Examples

### Demos & Examples
```bash
# Basic demo (2 min, no API needed)
uv run demo

# Full feature demo (10 min, optional API)
uv run example

# Advanced scenarios & analysis (5 min)
uv run advanced

# Real-world use cases (5 min)
uv run realworld

# Quick reference & copy-paste examples (3 min)
uv run quickref
```

### Direct Python Execution
```bash
# Instead of uv run, use:
python main.py          # Basic demo
python example.py       # Full features
python examples_advanced.py      # Advanced
python examples_realworld.py     # Real-world
python examples_quick_reference.py  # Quick ref
```

---

## ⚙️ Setup & Installation

```bash
# Install dependencies
uv sync

# Create environment file
cp .env.example .env

# Edit .env with your OpenAI API key
nano .env  # or your editor
```

---

## 📚 Documentation

```bash
# View main documentation
cat README.md           # English

# View Traditional Chinese docs
cat README.zh-TW.md     # 繁體中文

# View setup instructions
cat SETUP.md

# View all examples guide
cat EXAMPLES.md

# View this command reference
cat COMMANDS.md
```

---

## 🎯 Quick Demo Matrix

| Time | Command | Description |
|------|---------|-------------|
| **2 min** | `uv run demo` | Basics & cost estimation |
| **5 min** | `uv run quickref` | Copy-paste examples |
| **10 min** | `uv run example` | All features |
| **5 min** | `uv run advanced` | Deep analysis |
| **5 min** | `uv run realworld` | Production systems |
| **30 min** | All above | Complete learning |

---

## 💻 Usage Examples

### Scenario 1: First Time Users
```bash
uv sync
uv run demo          # See basic concepts
uv run quickref      # Get code examples
```

### Scenario 2: Understand Features
```bash
uv run example       # Learn all strategies
uv run advanced      # Study cost analysis
uv run realworld     # See real applications
```

### Scenario 3: Find Specific Solution
```bash
uv run realworld     # Find matching use case
uv run quickref      # Copy relevant example
# Customize and integrate
```

### Scenario 4: Cost Analysis
```bash
uv run advanced      # See cost scenarios
uv run quickref      # Example 4 & 6
# Analyze for your workload
```

---

## 🔍 Example Content Quick Map

### `uv run demo`
- Rule-based routing examples
- Monthly cost estimation
- Quick model recommendations

### `uv run example`
- Rule-based routing
- Custom rules
- LLM classification
- Hybrid router
- Cost comparison
- Distribution optimization

### `uv run advanced`
- Domain-specific routing
- Healthcare, Finance, E-commerce configurations
- Cost scenario analysis (startup to enterprise)
- Workload distribution analysis
- Token impact analysis
- Statistics tracking
- Quality threshold impact

### `uv run realworld`
- Customer support system
- Content generation pipeline
- Data processing workflow
- Code review system

### `uv run quickref`
- Basic routing (copy-paste)
- Quick recommendations
- Custom rules
- Cost estimation templates
- Model comparison table
- Distribution analysis
- Integration patterns
- Best practices

---

## 📊 Demo Output Examples

### demo (3 sections)
```
1. Rule-Based Routing Examples
2. Monthly Cost Estimation
3. Quick Model Recommendations
```

### example (6 sections)
```
1. Rule-Based Routing
2. Custom Routing Rules
3. LLM-Assisted Classification
4. Hybrid Router with Execution
5. Cost Comparison Scenarios
6. Distribution Optimization
```

### advanced (7 sections)
```
1. Domain-Specific Routing Rules
2. Industry-Specific Configurations
3. Cost Scenario Analysis
4. Workload Distribution Analysis
5. Token Size Impact on Costs
6. Routing Statistics
7. Quality Threshold Impact
```

### realworld (4 systems)
```
1. Customer Support Automation
2. Content Generation Pipeline
3. Data Processing Workflow
4. Code Review System
```

### quickref (8 sections)
```
1. Basic Routing
2. Quick Recommendations
3. Custom Rules
4. Cost Estimation
5. Model Comparison
6. Distribution Analysis
7. Integration Pattern
8. Best Practices
```

---

## 🛠️ Development Commands

```bash
# Check for errors
python -m py_compile main.py example.py

# Run with debug output
python -c "import main; main.main()" 2>&1 | head -50

# Interactive shell
python -i main.py
```

---

## 📝 File Locations

```
model_tiering_openai/
├── main.py                      # Core library + demo
├── example.py                   # Full examples suite
├── examples_advanced.py         # Advanced scenarios
├── examples_realworld.py        # Real-world use cases
├── examples_quick_reference.py  # Quick reference
├── pyproject.toml               # Project config
├── README.md                    # English docs
├── README.zh-TW.md             # Chinese docs
├── SETUP.md                     # Setup guide
├── EXAMPLES.md                  # Examples guide
├── COMMANDS.md                  # This file
├── .env.example                 # Config template
├── .env                         # Your config (git-ignored)
└── .gitignore                   # Git ignore rules
```

---

## 🎬 Common Workflows

### Workflow 1: Learn the System
```bash
$ uv run demo
$ uv run quickref
# Read the output, understand routing concepts
```

### Workflow 2: Evaluate for Your Use Case
```bash
$ uv run advanced
# Analyze cost scenarios matching your workload
$ uv run realworld
# Find matching use case
```

### Workflow 3: Implement Solution
```bash
$ uv run quickref
# Copy relevant example
# Customize routing rules
# Test with your data
```

### Workflow 4: Monitor & Optimize
```python
router = HybridRouter()
result = router.execute(task)
print(router.get_stats())  # Monitor statistics
# Adjust rules and quality thresholds
```

---

## ⚡ Speed Reference

| Command | Time | Requires API |
|---------|------|-------------|
| `uv run demo` | 2 sec | No |
| `uv run quickref` | 3 sec | No |
| `uv run realworld` | 3 sec | No |
| `uv run advanced` | 5 sec | No |
| `uv run example` | 5-30 sec | Optional |

---

## 🔐 Security Reminders

```bash
# Never commit your API key
cat .env >> .gitignore  # Already done

# Check your .env is git-ignored
git check-ignore .env   # Should output: .env

# Verify no secrets in code
grep -r "sk-" .        # Should not find API key
```

---

## 🚨 Troubleshooting Commands

```bash
# Check dependencies installed
pip list | grep openai

# Verify .env is readable
test -r .env && echo "✓ .env exists" || echo "✗ .env missing"

# Check API key is set
grep OPENAI_API_KEY .env

# Test import
python -c "from main import HybridRouter; print('✓ Imports OK')"

# Test routing (no API call)
python -c "from main import RuleBasedRouter; r = RuleBasedRouter(); print(r.route('translate').model.value)"
```

---

## 📞 Getting Help

```bash
# View example help
python examples_advanced.py --help 2>/dev/null || python examples_advanced.py

# Read docstrings
python -c "import main; help(main.HybridRouter)"

# Check file comments
grep -n "def\|#" main.py | head -30
```

---

## 💾 Saving Demo Output

```bash
# Save demo output to file
uv run demo > demo_output.txt

# Save with timestamp
uv run advanced > advanced_$(date +%Y%m%d_%H%M%S).txt

# Compare outputs
diff <(uv run demo) <(uv run demo)
```

---

## 🎓 Learning Path by Level

**Beginner (5 min)**
```bash
uv run demo
# Understand basic concepts
```

**Intermediate (20 min)**
```bash
uv run example
uv run quickref
# Learn all features and patterns
```

**Advanced (30 min)**
```bash
uv run advanced
uv run realworld
# Study detailed scenarios
```

**Expert (1 hour)**
```bash
# Run all examples
for cmd in demo example advanced realworld quickref; do
  echo "=== $cmd ==="
  uv run $cmd
done
```

---

## 📈 Next Steps After Demo

1. Choose your use case (consumer support, content, data, code review)
2. Run `uv run realworld` to find the matching example
3. Copy the relevant code from `uv run quickref`
4. Customize routing rules for your domain
5. Integrate into your application
6. Monitor with `router.get_stats()`

---

**Version:** 1.0
**Last Updated:** 2025-03-23
**OpenAI Models:** GPT-3.5-turbo, GPT-4, GPT-4-turbo
