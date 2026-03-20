# LinkedIn Job Application Automation Guide

## Overview
Automated job application system for LinkedIn using Chrome DevTools MCP tools.
Connects to existing Chrome browser session - no login handling needed.

## Prerequisites

### 1. Enable Chrome Remote Debugging
1. Open Chrome
2. Navigate to `chrome://inspect/#remote-debugging`
3. Toggle the switch to enable remote debugging

### 2. Log into LinkedIn
- Open LinkedIn in Chrome
- Ensure you're logged in before running automation

### 3. Node.js 22+
- Required for chrome-cdp-skill
- Install from: https://nodejs.org/

---

## Configuration

Edit `linkedin_automation_controller.js` CONFIG object:

```javascript
const CONFIG = {
  // Search parameters
  jobTitle: 'Software Engineer',
  locations: ['Taiwan', 'Remote'],
  datePosted: 'Past week', // 'Any time', 'Past month', 'Past week', 'Past 24 hours'

  // Filters
  easyApplyOnly: true,
  remoteOptions: ['Remote', 'Hybrid'], // 'On-site', 'Remote', 'Hybrid'
  experienceLevel: [], // 'Entry level', 'Mid-Senior level', etc.

  // Automation settings
  maxApplications: 50,
  delayBetweenApps: { min: 3000, max: 6000 },
  skipIfQuestionsRequired: false,
  maxPages: 10
};
```

---

## LinkedIn URL Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| `keywords` | Job search keywords | URL encoded string |
| `location` | Location filter | City/Country name |
| `f_AL` | Easy Apply filter | `true` |
| `f_TPR` | Date posted | `r86400` (24h), `r604800` (week), `r2592000` (month) |
| `f_WT` | Work type | `1` (On-site), `2` (Remote), `3` (Hybrid) |
| `f_E` | Experience level | `1-6` (Internship to Executive) |

**Example URL:**
```
https://www.linkedin.com/jobs/search/?keywords=Software%20Engineer&location=Taiwan&f_AL=true&f_TPR=r604800&f_WT=2,3
```

---

## Automation Process

### Step 1: Navigate to Job Search
```javascript
// Using MCP tool
mcp__chrome-devtools__navigate_page({
  type: 'url',
  url: 'https://www.linkedin.com/jobs/search/?keywords=Software%20Engineer&location=Taiwan&f_AL=true&f_TPR=r604800'
});
```

### Step 2: Inject Automation Helper
```javascript
// Inject scripts via MCP
mcp__chrome-devtools__evaluate_script({
  function: INJECT_SCRIPT // From linkedin_automation_controller.js
});
```

### Step 3: Get Job Cards
```javascript
mcp__chrome-devtools__evaluate_script({
  function: "() => window.getJobCards()"
});
```

### Step 4: Click on Job
```javascript
mcp__chrome-devtools__evaluate_script({
  function: "(index) => window.clickJobByIndex(index)",
  args: ["0"]
});
```

### Step 5: Click Easy Apply
```javascript
mcp__chrome-devtools__evaluate_script({
  function: "() => window.clickEasyApply()"
});
```

### Step 6: Complete Application Modal
```javascript
// Check modal state
mcp__chrome-devtools__evaluate_script({
  function: "() => window.getModalState()"
});

// Click next/submit
mcp__chrome-devtools__evaluate_script({
  function: "() => window.clickModalNext()"
});
```

### Step 7: Verify Success
```javascript
mcp__chrome-devtools__evaluate_script({
  function: "() => window.isSuccess()"
});
```

---

## Key Selectors

### Job Cards
```css
.job-card-container
.jobs-search-results__list-item
[data-job-id]
```

### Easy Apply Button
```css
button.jobs-apply-button
/* Or by text content: "Easy Apply" / "簡易應徵" */
```

### Application Modal
```css
.jobs-easy-apply-modal
[role="dialog"]
```

### Modal Buttons
- Next: `button` containing "Next"
- Submit: `button` containing "Submit application"
- Close: `button[aria-label="Dismiss"]`

---

## Keyboard Controls

When automation is running:
- **P** = Pause automation
- **R** = Resume automation
- **Q** = Quit automation

---

## Application Flow

```
1. Search Results Page
   ├── Get list of job cards
   ├── Filter out already applied jobs
   └── Click on first unapplied job

2. Job Details Panel
   ├── Find "Easy Apply" button
   ├── Click to open application modal
   └── If no Easy Apply, skip job

3. Application Modal (multi-step)
   ├── Contact Info → Click "Next"
   ├── Resume → Click "Next"
   ├── Questions (if any) → Fill & Click "Next"
   ├── Review → Click "Submit application"
   └── Success confirmation

4. After Application
   ├── Close success modal
   ├── Return to search results
   ├── Wait random delay (3-6 seconds)
   └── Process next job

5. Pagination
   ├── When page exhausted, click "Next" page
   └── Repeat until maxApplications reached
```

---

## Error Handling

### Common Issues

1. **Modal has required questions**
   - If `skipIfQuestionsRequired: true`, close modal and skip
   - Otherwise, try to fill default values

2. **Easy Apply button not found**
   - Job requires external application
   - Skip and move to next job

3. **Rate limiting**
   - Increase delay between applications
   - LinkedIn may temporarily limit applications

4. **Session timeout**
   - Re-authenticate in Chrome manually
   - Restart automation

---

## Safety Features

1. **Random delays** between applications (configurable)
2. **Pause/Resume/Quit** keyboard controls
3. **Skip already applied** jobs automatically
4. **Status indicator** shows current progress
5. **Max applications limit** to prevent over-application

---

## Logging Format

```
[Job 1/25] Software Engineer at Company Name
  Status: SUCCESS / SKIPPED / FAILED
  Reason: (if skipped/failed)
  Time: 2024-03-20 10:30:45
```

---

## Files Structure

```
linkedin/claude/
├── CLAUDE.md                          # This guide
├── linkedin_auto_apply.js             # Helper functions (browser-side)
├── linkedin_automation_controller.js  # Main controller & config
└── README.md                          # Quick start guide
```

---

## Quick Reference

### Start Automation (via MCP)
1. List Chrome pages: `mcp__chrome-devtools__list_pages`
2. Select LinkedIn tab: `mcp__chrome-devtools__select_page`
3. Navigate to search: `mcp__chrome-devtools__navigate_page`
4. Take snapshot: `mcp__chrome-devtools__take_snapshot`
5. Inject helpers: `mcp__chrome-devtools__evaluate_script`
6. Execute automation loop

### Check State
```javascript
mcp__chrome-devtools__evaluate_script({
  function: "() => window.automationState"
});
```

### Update Counter
```javascript
mcp__chrome-devtools__evaluate_script({
  function: "() => { window.automationState.applied++; return window.automationState; }"
});
```

---

## Chrome CDP Skill (Recommended Method)

The `chrome-cdp-skill` connects directly to your existing Chrome browser session - no separate browser instance, no re-login needed.

### Installation

```bash
# Clone to Claude skills directory
cd ~/.claude/skills
git clone https://github.com/pasky/chrome-cdp-skill.git chrome-cdp
```

### Prerequisites

1. **Node.js 22+** (required for built-in WebSocket)
   ```bash
   nvm install 22
   nvm use 22
   ```

2. **Enable Chrome Remote Debugging**
   - Open Chrome → `chrome://inspect/#remote-debugging`
   - Toggle the switch ON

3. **Log into LinkedIn in Chrome** before running automation

### CDP Commands

```bash
# Set up Node 22 (run once per terminal session)
source ~/.nvm/nvm.sh && nvm use 22

# List open Chrome tabs
~/.claude/skills/chrome-cdp/skills/chrome-cdp/scripts/cdp.mjs list

# Navigate to URL
~/.claude/skills/chrome-cdp/skills/chrome-cdp/scripts/cdp.mjs nav <target> "https://..."

# Take screenshot (saves to ~/.cache/cdp/)
~/.claude/skills/chrome-cdp/skills/chrome-cdp/scripts/cdp.mjs shot <target>

# Get accessibility tree snapshot
~/.claude/skills/chrome-cdp/skills/chrome-cdp/scripts/cdp.mjs snap <target>

# Click element by CSS selector
~/.claude/skills/chrome-cdp/skills/chrome-cdp/scripts/cdp.mjs click <target> "button.jobs-apply-button"

# Click at CSS pixel coordinates
~/.claude/skills/chrome-cdp/skills/chrome-cdp/scripts/cdp.mjs clickxy <target> <x> <y>

# Type text at focused element
~/.claude/skills/chrome-cdp/skills/chrome-cdp/scripts/cdp.mjs type <target> "text to type"

# Evaluate JavaScript in page context
~/.claude/skills/chrome-cdp/skills/chrome-cdp/scripts/cdp.mjs eval <target> "() => document.title"
```

**Note:** `<target>` is a unique prefix of the targetId from `list` command (e.g., `4BF87A` for a tab with ID `4BF87A9C`).

### LinkedIn Automation with CDP

```bash
# 1. List tabs to find LinkedIn
cdp.mjs list

# 2. Navigate to job search (with Easy Apply filter)
cdp.mjs nav 4BF87A "https://www.linkedin.com/jobs/search/?keywords=Software%20Engineer&location=Taiwan&f_AL=true&f_TPR=r604800&f_WT=2,3"

# 3. Take screenshot to verify
cdp.mjs shot 4BF87A

# 4. Click Easy Apply button
cdp.mjs click 4BF87A "button.jobs-apply-button"

# 5. Evaluate JS to interact with modal
cdp.mjs eval 4BF87A '(() => {
  const modal = document.querySelector(".jobs-easy-apply-modal");
  const nextBtn = modal?.querySelector("button");
  if (nextBtn && nextBtn.textContent.includes("Next")) {
    nextBtn.click();
    return { clicked: true };
  }
  return { clicked: false };
})()'
```

### Key Selectors (Verified 2026-03)

| Element | Selector |
|---------|----------|
| Easy Apply button | `button.jobs-apply-button` |
| Application modal | `.jobs-easy-apply-modal` |
| Modal close button | `button[aria-label="Dismiss"]` |
| Next button in modal | Button containing "Next" |
| Submit button | Button containing "Submit application" |
| Job cards | `[data-job-id]` |

### Common Issues

1. **WebSocket is not defined**
   - You're using Node.js < 22
   - Fix: `nvm use 22`

2. **Messaging popup blocking**
   - LinkedIn chat windows can overlay the page
   - Close them first or navigate to fresh URL with `&refresh=true`

3. **Modal not appearing after click**
   - Wait 2-3 seconds after clicking Easy Apply
   - Use `sleep 2` before next command

4. **Screenshot coordinates**
   - Screenshots have 2x DPR on Retina displays
   - CSS coordinates = Screenshot pixels ÷ 2

---

## Status: Ready for Testing
- Helper functions created
- MCP integration ready
- Chrome CDP skill installed and tested
- Configuration documented
