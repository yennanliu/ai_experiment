# 104.com.tw Auto Apply - Using Chrome CDP Skill

## Overview

This setup uses **chrome-cdp-skill** (from [pasky/chrome-cdp-skill](https://github.com/pasky/chrome-cdp-skill)) to connect to your **existing Chrome browser**. This means:
- No need to start Chrome with special flags
- Reuse your already logged-in sessions
- Works with your normal Chrome profile (not test/dev)
- Persistent daemon connections (no repeated permission prompts)

---

## Prerequisites

### 1. Node.js 22+
Chrome CDP requires Node.js 22 or higher.

```bash
# Check current version
node --version

# Install Node 22+ via nvm
nvm install 22
nvm use 22

# Or download from https://nodejs.org/
```

### 2. Enable Remote Debugging in Chrome
Open Chrome and navigate to:
```
chrome://inspect/#remote-debugging
```
Toggle **ON** the "Enable remote debugging" switch.

That's it! No need to restart Chrome with special flags.

---

## Quick Start

### Step 1: List Your Open Tabs
```bash
cd /Users/jerryliu/ai_experiment/104/gemini
./cdp list
```

This shows all open tabs with their `targetId` prefix. Example:
```
0A3B  https://www.104.com.tw/jobs/search/...  104人力銀行
1C2D  https://mail.google.com/...             Gmail
```

### Step 2: Navigate to Job Search (if needed)
```bash
./cdp nav 0A3B "https://www.104.com.tw/jobs/search/?keyword=軟體工程師&order=15&remoteWork=1,2"
```

### Step 3: Take a Snapshot to See Page Structure
```bash
./cdp snap 0A3B
```

### Step 4: Interact with the Page
```bash
# Click an element
./cdp click 0A3B ".apply-button__button"

# Type text
./cdp type 0A3B "Hello World"

# Execute JavaScript
./cdp eval 0A3B "document.title"
```

---

## Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `list` | Show all open tabs | `./cdp list` |
| `snap <id>` | Get accessibility tree | `./cdp snap 0A3B` |
| `shot <id>` | Take screenshot | `./cdp shot 0A3B` |
| `html <id> [selector]` | Get HTML content | `./cdp html 0A3B ".job-list"` |
| `nav <id> <url>` | Navigate to URL | `./cdp nav 0A3B "https://..."` |
| `click <id> <selector>` | Click element | `./cdp click 0A3B "button.submit"` |
| `clickxy <id> <x> <y>` | Click at coordinates | `./cdp clickxy 0A3B 100 200` |
| `type <id> <text>` | Type text | `./cdp type 0A3B "text"` |
| `eval <id> <expr>` | Run JavaScript | `./cdp eval 0A3B "location.href"` |
| `open [url]` | Open new tab | `./cdp open "https://..."` |
| `stop [id]` | Stop daemon(s) | `./cdp stop` |

**Note:** `<id>` is the unique prefix from `./cdp list` output.

---

## 104 Job Application Workflow

### 1. Open 104 Job Search in Chrome
Navigate to 104.com.tw in your browser and login if needed.

### 2. Get the Tab ID
```bash
./cdp list
# Find the 104 tab ID (e.g., 0A3B)
```

### 3. Apply to Jobs via CLI

```bash
# Take snapshot to see current state
./cdp snap 0A3B

# Click apply button (first one on the page)
./cdp eval 0A3B "document.querySelector('.apply-button__button').click()"

# Wait a moment for new tab to open, then list tabs
./cdp list

# Switch to new tab (application form) - get its ID
# e.g., new tab has ID 1X2Y

# Select cover letter from dropdown
./cdp eval 1X2Y "
  // Open dropdown
  const dropdown = document.querySelector('.multiselect');
  if (dropdown) dropdown.click();
"

# Select the custom cover letter
./cdp eval 1X2Y "
  setTimeout(() => {
    const options = document.querySelectorAll('.multiselect__option');
    for (const opt of options) {
      if (opt.textContent.includes('自訂推薦信1')) {
        opt.click();
        break;
      }
    }
  }, 500);
"

# Submit application
./cdp eval 1X2Y "
  const submitBtn = Array.from(document.querySelectorAll('button'))
    .find(b => b.textContent.includes('確認送出'));
  if (submitBtn) submitBtn.click();
"
```

---

## For AI Agent (Gemini CLI / Claude Code)

### Basic Instructions
```
Use ./cdp commands to interact with my Chrome browser.

1. First run: ./cdp list
   - Find the 104.com.tw tab

2. To apply to jobs:
   - ./cdp snap <id> to see the page structure
   - ./cdp eval <id> "js code" to click buttons and fill forms

3. Important selectors for 104:
   - Apply button: .apply-button__button
   - Cover letter dropdown: .multiselect
   - Cover letter option: .multiselect__option containing "自訂推薦信1"
   - Submit button: button containing "確認送出"
```

### Example AI Prompt
```
Connect to my Chrome browser using ./cdp commands.
List tabs with ./cdp list, find the 104.com.tw tab.
Then help me apply to jobs on that page.
Use ./cdp eval to run JavaScript for clicking and form filling.
```

---

## Why Chrome CDP Skill?

| Feature | Playwright MCP | Chrome CDP Skill |
|---------|---------------|------------------|
| Session reuse | Need to start Chrome with debug flags | Uses existing Chrome |
| Login persistence | Requires manual login or .env credentials | Reuses logged-in session |
| Permission prompts | Per-command | Per-tab (persistent daemon) |
| Tab handling | New browser contexts | Your actual tabs |
| 100+ tabs | May timeout | Reliable |

---

## Troubleshooting

### "Cannot find DevToolsActivePort"
Enable remote debugging in Chrome:
1. Go to `chrome://inspect/#remote-debugging`
2. Toggle ON "Enable remote debugging"

### "Node.js 22+ required"
```bash
nvm install 22
nvm use 22
```

### Commands not working
Make sure the target ID is correct:
```bash
./cdp list
```

### Daemon issues
Stop all daemons and retry:
```bash
./cdp stop
```

---

## File Structure

```
/Users/jerryliu/ai_experiment/104/gemini/
├── cdp                        # CLI wrapper script
├── scripts/
│   └── cdp.mjs               # Chrome CDP skill (from pasky/chrome-cdp-skill)
├── GEMINI_CLI_INSTRUCTIONS.md # This file
├── CHROME_CDP_SKILL.md        # Full command reference
└── (old files)
    ├── 104_auto_apply_gemini.js  # Legacy Playwright script
    └── start-chrome-debug.sh     # Legacy Chrome debug helper
```

---

## Credits

Chrome CDP Skill by [pasky/chrome-cdp-skill](https://github.com/pasky/chrome-cdp-skill)
