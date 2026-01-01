# 🤖 Fully Automated Orchestration Guide

## Overview

This guide shows how to use the **Claude API-based orchestrator** for fully automatic multi-agent orchestration with **zero manual intervention**.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
npm install
```

This installs `@anthropic-ai/sdk` for Claude API access.

### 2. Set API Key

Get your API key from: https://console.anthropic.com/settings/keys

**Option A: Environment Variable**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

**Option B: .env File**
```bash
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
```

### 3. Run Orchestration

```bash
node orchestrate-with-api.js "Your feature request"
```

**Examples:**
```bash
node orchestrate-with-api.js "Add dark mode toggle"
node orchestrate-with-api.js "Add export notes to PDF"
node orchestrate-with-api.js "Add note favorites and pinning"
```

---

## 🎬 What Happens Automatically

When you run the command, the orchestrator:

1. **🤖 Calls PM Agent** via Claude API
   - Analyzes your feature request
   - Generates `feature-requirements.md`
   - Includes user stories, acceptance criteria, technical considerations

2. **🤖 Calls Backend Agent** via Claude API
   - Reviews PM requirements
   - Determines API changes needed
   - Generates `feature-api-changes.json`

3. **🤖 Calls Frontend Agent** via Claude API
   - Reviews PM requirements + Backend analysis
   - Designs UI implementation
   - Generates `feature-ui-implementation.md`

**All automatically - no manual steps!**

---

## 📊 Example Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🚀 Automated Agent Orchestration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Feature Request: Add dark mode toggle

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 Phase 1: Product Manager Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 Running PM Agent...
   Thinking...
   ✓ PM Agent completed
   📊 Tokens: 1250 in / 890 out
   💾 Saved: feature-requirements.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚙️  Phase 2: Backend Engineer Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 Running Backend Agent...
   Thinking...
   ✓ Backend Agent completed
   📊 Tokens: 1450 in / 520 out
   💾 Saved: feature-api-changes.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  💻 Phase 3: Frontend Engineer Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 Running Frontend Agent...
   Thinking...
   ✓ Frontend Agent completed
   📊 Tokens: 2100 in / 1850 out
   💾 Saved: feature-ui-implementation.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ Orchestration Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Generated Files:
   ✓ workspace/feature-request.txt
   ✓ workspace/feature-requirements.md
   ✓ workspace/feature-api-changes.json
   ✓ workspace/feature-ui-implementation.md

⏱️  Total Time: 12.4s

💡 Next Steps:
   1. Review the generated specifications in workspace/
   2. Ask Claude Code to implement the feature
   3. Test in your running application
```

---

## 🔧 Configuration

Edit `orchestrate-with-api.js` to customize:

```javascript
const CONFIG = {
  model: 'claude-sonnet-4-5-20250929',  // Claude model to use
  maxTokens: 4096,                       // Max tokens per agent
  workspaceDir: './workspace',           // Output directory
  agentsDir: './.claude/agents'          // Agent definitions
};
```

---

## 💰 Cost Estimation

Approximate costs per orchestration (using Sonnet 4.5):

- **PM Agent**: ~1,500 tokens → $0.018
- **Backend Agent**: ~2,000 tokens → $0.024
- **Frontend Agent**: ~4,000 tokens → $0.048

**Total per feature**: ~$0.09 USD

*(Prices as of 2025, check current pricing at anthropic.com)*

---

## 🆚 Comparison: Manual vs API

| Feature | Manual Orchestrator | API Orchestrator |
|---------|-------------------|------------------|
| **Setup** | None needed | Requires API key |
| **Speed** | Manual steps (slow) | Fully automatic (fast) |
| **Cost** | Free (uses current session) | ~$0.09 per feature |
| **Convenience** | Requires interaction | One command |
| **Production Ready** | No | Yes |

---

## 🐛 Troubleshooting

### Error: "ANTHROPIC_API_KEY not set"
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Error: "Cannot find module '@anthropic-ai/sdk'"
```bash
npm install
```

### Error: "Failed to read agent file"
Check that `.claude/agents/` directory exists with:
- `pm.md`
- `backend.md`
- `frontend.md`

### API Rate Limits
If you hit rate limits, the script will fail. Wait a moment and retry.

---

## 🎯 Use Cases

Perfect for:
- ✅ Rapid feature prototyping
- ✅ Architecture planning
- ✅ Automated documentation
- ✅ CI/CD integration
- ✅ Large-scale projects

---

## 🔐 Security Notes

- **Never commit** your API key to git
- Add `.env` to `.gitignore`
- Use environment variables in production
- Rotate keys regularly

---

## 📚 Files Generated

After orchestration, you'll have:

```
workspace/
├── feature-request.txt          # Original request
├── feature-requirements.md      # PM Agent output
├── feature-api-changes.json     # Backend Agent output
└── feature-ui-implementation.md # Frontend Agent output
```

Review these before implementing the feature.

---

## 🚀 Next: Implement the Feature

Once orchestration is complete:

1. **Review specifications**
   ```bash
   cat workspace/feature-requirements.md
   cat workspace/feature-ui-implementation.md
   ```

2. **Ask Claude Code to implement**
   ```
   "Implement dark mode based on the generated specifications in workspace/"
   ```

3. **Test in running app**
   ```
   http://localhost:5173
   ```

---

## 📝 Advanced: NPM Script

Add to your workflow:

```bash
# Run via npm
npm run orchestrate "Add dark mode"

# Or create an alias
alias orchestrate='node orchestrate-with-api.js'
orchestrate "Add feature X"
```

---

## ✨ That's It!

You now have a **fully automated orchestration system** that runs all agents with a single command!

Try it now:
```bash
node orchestrate-with-api.js "Add dark mode toggle to the application"
```
