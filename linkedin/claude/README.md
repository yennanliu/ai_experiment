# LinkedIn Auto Apply Automation

Automated job application for LinkedIn using Chrome DevTools MCP tools.
Reuses existing browser session - no login handling required.

## Quick Start

### 1. Setup Chrome
```bash
# Open Chrome and enable remote debugging
# Navigate to: chrome://inspect/#remote-debugging
# Toggle the switch ON
```

### 2. Login to LinkedIn
Open LinkedIn in Chrome and make sure you're logged in.

### 3. Configure Search
Edit `linkedin_automation_controller.js`:
```javascript
const CONFIG = {
  jobTitle: 'Software Engineer',
  locations: ['Taiwan', 'Remote'],
  datePosted: 'Past week',
  easyApplyOnly: true,
  maxApplications: 50
};
```

### 4. Run via Claude Code

Use MCP tools to control the automation:

```javascript
// 1. List Chrome pages
mcp__chrome-devtools__list_pages()

// 2. Navigate to LinkedIn Jobs
mcp__chrome-devtools__navigate_page({
  type: 'url',
  url: 'https://www.linkedin.com/jobs/search/?keywords=Software%20Engineer&location=Taiwan&f_AL=true'
})

// 3. Take snapshot to see page
mcp__chrome-devtools__take_snapshot()

// 4. Inject automation helpers
mcp__chrome-devtools__evaluate_script({
  function: `...INJECT_SCRIPT from controller...`
})

// 5. Execute automation steps
mcp__chrome-devtools__evaluate_script({
  function: "() => window.clickEasyApply()"
})
```

## Keyboard Controls

While automation is running:
- **P** = Pause
- **R** = Resume
- **Q** = Quit

## Files

| File | Description |
|------|-------------|
| `linkedin_auto_apply.js` | Browser-side helper functions |
| `linkedin_automation_controller.js` | Main controller & configuration |
| `CLAUDE.md` | Detailed automation guide |
| `README.md` | This quick start guide |

## URL Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `keywords` | Search terms | `Software%20Engineer` |
| `location` | Location | `Taiwan` |
| `f_AL` | Easy Apply only | `true` |
| `f_TPR` | Date posted | `r604800` (past week) |
| `f_WT` | Remote work | `2` (Remote) |

## Safety Features

- Random delays between applications
- Skip already applied jobs
- Manual pause/resume/quit controls
- Max applications limit
- Visual status indicator

## Requirements

- Chrome browser with remote debugging enabled
- Node.js 22+ (for chrome-cdp-skill)
- LinkedIn account (already logged in)

## Related

- 104.com.tw automation: `../104/claude/`
- chrome-cdp-skill: https://github.com/yennanliu/chrome-cdp-skill
