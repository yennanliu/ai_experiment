# LinkedIn Auto Apply Automation

Automated job application for LinkedIn using Chrome DevTools Protocol (CDP).
Reuses existing browser session - no login handling required.

## Quick Start with Claude Code (Recommended)

### 1. Prerequisites

```bash
# Node.js is available if using Playwright MCP (built-in)
# Alternatively, install chrome-cdp-skill:
cd ~/.claude/skills
git clone https://github.com/pasky/chrome-cdp-skill.git chrome-cdp

# Install Node.js 22+ (for cdp-skill)
nvm install 22
nvm use 22
```

### 2. Setup Chrome

1. Open Chrome browser
2. Navigate to `chrome://inspect/#remote-debugging`
3. Toggle the switch **ON**
4. Open LinkedIn and login

### 3. Using Claude Code with linkedin_automation_controller.js

The main automation logic is in **`linkedin_automation_controller.js`** which provides:
- Configuration object for job search parameters
- Helper functions for DOM interaction (injected into browser)
- Status tracking and UI indicator
- Keyboard controls (P=Pause, R=Resume, Q=Quit)

**Simply tell Claude Code what you want:**

```
# Start Claude Code
claude

# Then tell it:
> Apply to 10 software engineer jobs in Germany

> Apply to 5 remote backend developer jobs in France

> How many jobs can I apply to in UK?
```

**Claude Code will automatically:**
- Use linkedin_automation_controller.js to build the search URL
- Navigate to LinkedIn job search with your filters
- Inject helper functions into the page
- Click through jobs and apply automatically
- Handle multi-step application forms
- Fill numeric fields with default values
- Track progress with status indicator
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
| **`linkedin_automation_controller.js`** | **Main automation controller** - Contains CONFIG object, URL builder, and INJECT_SCRIPT (required for Claude Code) |
| `linkedin_auto_apply.js` | Browser-side helper functions (legacy) |
| `cdp.sh` | Wrapper script for CDP commands |
| `CLAUDE.md` | Detailed automation guide with all steps and selectors |
| `README.md` | This quick start guide |
| `README_zh-TW.md` | Traditional Chinese guide |

### linkedin_automation_controller.js - Core Module

This file is **the main entry point** for LinkedIn automation. It exports:

```javascript
// Edit CONFIG to customize searches
CONFIG = {
  jobTitle: 'Software Engineer',
  locations: ['Germany', 'Remote'],
  datePosted: 'Past week',
  easyApplyOnly: true,
  remoteOptions: ['Remote', 'Hybrid'],
  maxApplications: 10,
  delayBetweenApps: { min: 3000, max: 6000 }
}

// Build search URL from CONFIG
buildSearchUrl(config)

// Inject helper functions into browser page
INJECT_SCRIPT

// Get/update automation state
CHECK_STATE_SCRIPT
getUpdateCounterScript(type)
```

**How Claude Code uses it:**
1. Reads CONFIG for search parameters
2. Calls `buildSearchUrl()` to generate LinkedIn search URL
3. Navigates to the URL
4. Injects INJECT_SCRIPT to add helper functions to page
5. Runs automation loop calling helper functions
6. Returns results without large console logs

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

## Job Filtering Strategy

**Only Easy Apply Jobs are Processed**

The automation automatically filters jobs and only applies to those with:
- ✅ "Easy Apply" button visible
- ✅ Not already applied to
- ✅ Still accepting applications

**Jobs that are SKIPPED:**
- ❌ Jobs requiring external application (no Easy Apply)
- ❌ Already applied positions
- ❌ "No longer accepting" positions
- ❌ Closed/archived listings

This filtering happens automatically:
1. Scan all visible job listings
2. Check each job card for Easy Apply availability
3. Only click/apply to qualifying jobs
4. Report skipped jobs in summary

**Result:** Higher success rate, zero wasted attempts, efficient automation.

---

## Handling Large Responses & Context Issues

**Problem:** Large automation scripts can generate 50-100KB responses, causing token limit issues.

**Solution:** Use **Batch Processing with Minimal Returns**

1. **Return only summary data** (not full logs)
   ```javascript
   // Instead of logging to console (large output)
   // Return structured summary
   return { applied: 9, failed: 0, jobs: [...] }
   ```

2. **Process jobs incrementally**
   - Apply to 5-10 jobs per batch
   - Return summary, not raw logs
   - Avoid storing full browser console output

3. **Split large automations**
   ```javascript
   // Bad: One giant loop returning all logs
   for (i = 0; i < 100; i++) { /* large loop */ }

   // Good: Smaller batches
   for (i = 0; i < 10; i++) { /* apply to 10 jobs */ }
   return summary
   // Then call again if needed
   ```

4. **Only read necessary parts of results**
   - Use `Grep` to search for specific patterns
   - Use `offset` and `limit` with `Read` tool
   - Don't read entire large files at once

5. **Example optimized approach:**
   ```javascript
   // Runs automation but returns only summary
   async function autoApplyBatch(maxJobs = 10) {
     const results = { applied: 0, failed: 0, jobs: [] };

     for (let i = 0; i < maxJobs; i++) {
       // ... apply to job ...
       results.jobs.push({ index: i, status: 'SUCCESS' });
     }

     return results; // Small, structured output only
   }
   ```

## Best Practices for Future Automations

### 1. Reference linkedin_automation_controller.js
Always start by asking Claude Code to use the existing controller:
```
> Apply to 10 backend engineer jobs in Singapore using linkedin_automation_controller.js

> Use the CONFIG in linkedin_automation_controller.js to search for remote data engineer jobs
```

### 2. Customize CONFIG for your search
The CONFIG object supports:
- `jobTitle`: Any job role
- `locations`: Multiple locations
- `datePosted`: 'Past week' / 'Past month' / 'Any time'
- `remoteOptions`: ['Remote'], ['Hybrid'], ['On-site'], or mix
- `maxApplications`: Limit number of applications
- `delayBetweenApps`: { min: 3000, max: 6000 } for safety

### 3. Only applies to jobs with "Easy Apply" button
- Automatically filters to jobs with Easy Apply enabled
- Skips jobs requiring external application
- Skips already-applied jobs
- Skips "no longer accepting" positions
- This avoids failures and wasted time

### 4. Keep automations small
- Apply to 10-20 jobs per run (maximum)
- Return only summary results
- Use pause/resume controls (P/R/Q) for real-time control

### 5. Monitor progress
Watch the blue automation status indicator (top-right corner):
- 🤖 RUNNING - Currently applying
- ⏸️  PAUSED - Press R to resume
- 🛑 STOPPED - Automation ended
- ✅ COMPLETED - Finished successfully

### 6. Reading large result files
If results are too large:
```bash
# Use grep to search for specific results
grep "SUCCESS\|FAILED" result_file.txt

# Use read with offset/limit
read file_path --offset 1 --limit 100
```

## Related

- 104.com.tw automation: See project CLAUDE.md
- chrome-cdp-skill: https://github.com/pasky/chrome-cdp-skill
