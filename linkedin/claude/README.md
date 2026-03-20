# LinkedIn Auto Apply Automation

Automated job application for LinkedIn using Chrome DevTools Protocol (CDP).
Reuses existing browser session - no login handling required.

## Quick Start with Claude Code

### 1. Prerequisites

```bash
# Install chrome-cdp-skill
cd ~/.claude/skills
git clone https://github.com/pasky/chrome-cdp-skill.git chrome-cdp

# Install Node.js 22+
nvm install 22
nvm use 22
```

### 2. Setup Chrome

1. Open Chrome browser
2. Navigate to `chrome://inspect/#remote-debugging`
3. Toggle the switch **ON**
4. Open LinkedIn and login

### 3. Run with Claude Code

Simply tell Claude Code what you want:

```
# Start Claude Code
claude

# Then tell it what to do:
> Apply to 10 software engineer jobs in China on LinkedIn

> Apply to 5 remote backend developer jobs in Taiwan

> Show me available jobs on LinkedIn for data scientist
```

Claude Code will automatically:
- Connect to your Chrome browser via CDP
- Navigate to LinkedIn job search
- Find and apply to jobs matching your criteria
- Handle the multi-step application flow
- Skip already-applied jobs

### 4. Example Prompts for Claude Code

```
# Basic job search and apply
> Apply to 10 software engineer jobs in China

# With specific filters
> Apply to remote Python developer jobs in Taiwan, past week only

# Check progress
> How many jobs have I applied to today?

# Manual control
> Show me the next unapplied job on LinkedIn
> Click Easy Apply on this job
> Submit the application
```

## Manual Usage (cdp.sh)

For manual control without Claude Code:

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
