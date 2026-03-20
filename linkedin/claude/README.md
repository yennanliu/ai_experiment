# LinkedIn Auto Apply Automation

Automated job application for LinkedIn using Chrome DevTools Protocol (CDP).
Reuses existing browser session - no login handling required.

## Quick Start

### 1. Install chrome-cdp-skill

```bash
cd ~/.claude/skills
git clone https://github.com/pasky/chrome-cdp-skill.git chrome-cdp
```

### 2. Install Node.js 22+

```bash
nvm install 22
nvm use 22
```

### 3. Setup Chrome

```bash
# Open Chrome and enable remote debugging
# Navigate to: chrome://inspect/#remote-debugging
# Toggle the switch ON
```

### 4. Login to LinkedIn

Open LinkedIn in Chrome and make sure you're logged in.

### 5. Run Automation

```bash
# Set up environment
source ~/.nvm/nvm.sh && nvm use 22
CDP="~/.claude/skills/chrome-cdp/skills/chrome-cdp/scripts/cdp.mjs"

# List Chrome tabs to find target ID
$CDP list

# Navigate to job search (replace TARGET with your tab ID prefix)
$CDP nav TARGET "https://www.linkedin.com/jobs/search/?keywords=Software%20Engineer&location=China&f_AL=true&f_TPR=r604800&f_WT=2,3"

# Use the wrapper script for easy commands
./cdp.sh jobs TARGET      # List jobs
./cdp.sh apply TARGET     # Click Easy Apply
./cdp.sh next TARGET      # Click Next
./cdp.sh review TARGET    # Click Review
./cdp.sh submit TARGET    # Submit application
./cdp.sh close TARGET     # Close modal
```

## Wrapper Script (cdp.sh)

```bash
# Set target once
export LINKEDIN_TARGET=4BF87A

# Basic commands
./cdp.sh list              # List Chrome tabs
./cdp.sh nav               # Navigate to job search
./cdp.sh shot              # Take screenshot
./cdp.sh jobs              # List available jobs

# Application flow
./cdp.sh nextjob           # Click next unapplied job
./cdp.sh apply             # Click Easy Apply button
./cdp.sh next              # Click Next in modal
./cdp.sh review            # Click Review
./cdp.sh submit            # Submit application
./cdp.sh close             # Close success modal

# Handle questions
./cdp.sh yes               # Answer Yes to work auth

# Full automation
./cdp.sh auto              # Complete one application automatically
```

## Application Flow

| Step | Progress | Action |
|------|----------|--------|
| Contact Info | 0% | Pre-filled → Next |
| Resume | 25% | Select resume → Next |
| Work Experience | 50% | Pre-filled → Next |
| Education | 75% | Pre-filled → Review |
| Additional | (optional) | Message → Review |
| Review | 100% | Check → Submit |
| Success | Done | Dismiss modal |

## Key Selectors

| Element | Selector |
|---------|----------|
| Easy Apply button | `button.jobs-apply-button` |
| Application modal | `.jobs-easy-apply-modal` |
| Next button | `button[aria-label='Continue to next step']` |
| Review button | `button[aria-label='Review your application']` |
| Submit button | `button[aria-label='Submit application']` |
| Close button | `button[aria-label='Dismiss']` |
| Job cards | `[data-job-id]` |

## URL Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| `keywords` | Search terms | `Software%20Engineer` |
| `location` | Location | `China`, `Taiwan` |
| `f_AL` | Easy Apply only | `true` |
| `f_TPR` | Date posted | `r86400` (24h), `r604800` (week) |
| `f_WT` | Remote work | `1` (onsite), `2` (remote), `3` (hybrid) |

## Files

| File | Description |
|------|-------------|
| `cdp.sh` | Wrapper script for CDP commands |
| `linkedin_auto_apply.js` | Browser-side helper functions |
| `linkedin_automation_controller.js` | Main controller & configuration |
| `CLAUDE.md` | Detailed automation guide |
| `README.md` | This quick start guide |
| `README_zh-TW.md` | Traditional Chinese guide |

## Requirements

- Chrome browser with remote debugging enabled
- Node.js 22+ (for built-in WebSocket)
- LinkedIn account (already logged in)
- chrome-cdp-skill installed

## Troubleshooting

| Issue | Solution |
|-------|----------|
| WebSocket is not defined | Use Node.js 22+: `nvm use 22` |
| Messaging popup blocking | Navigate with `&refresh=true` |
| Modal not appearing | Wait 2-3 seconds after click |
| Additional questions | Use `./cdp.sh yes` to answer |

## Related

- 104.com.tw automation: See project CLAUDE.md
- chrome-cdp-skill: https://github.com/pasky/chrome-cdp-skill
