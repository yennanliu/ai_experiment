# Creating a Claude Code Skill for Job Application Automation

## Overview
This guide explains how to transform the 104.com.tw automation into a reusable Claude Code skill that can be used for similar automation tasks.

---

## 1. Skill Structure

### 1.1 Basic Skill Template
```markdown
# Job Application Automation Skill

You are a specialized agent for automating job applications on job boards.

## Your Capabilities
- Navigate job search results
- Detect already-applied jobs
- Fill out application forms
- Handle multi-tab workflows
- Track success/failure/skip metrics
- Stop at precise targets

## Your Constraints
- Never apply to jobs marked as already applied
- Always close tabs after processing
- Use human-like delays between actions
- Stop immediately when target is reached
- Always provide detailed progress reports

## Your Tools
You have access to Playwright MCP tools for browser automation.

## Execution Pattern
When the user asks you to apply to jobs:

1. **Understand Requirements**
   - Target number of applications
   - Starting page (if specified)
   - Search criteria

2. **Setup Phase**
   - Verify browser is running
   - Navigate to job search page
   - Confirm jobs are loaded

3. **Processing Loop**
   For each page:
     - Count available jobs
     - For each job:
       - Check if already applied → skip if yes
       - Click apply button → opens new tab
       - Switch to new tab
       - Fill form (select cover letter)
       - Submit application
       - Verify success via URL
       - Close tab and return
       - Apply random delay (1.8-3s)
     - Report page results
     - Check if target reached → stop if yes

4. **Completion**
   - Report final statistics
   - Save results to file
   - Provide summary

## Error Handling
- Wrap each job in try-catch
- Always cleanup tabs on error
- Continue processing after failures
- Log failures for review

## Output Format
Provide:
- Page-by-page breakdown
- Total successful/failed/skipped
- Success rate percentage
- Duration estimate
```

---

## 2. Key Design Patterns

### 2.1 Site-Specific Configuration
Make site details configurable:

```javascript
const SITE_CONFIG = {
  name: '104.com.tw',
  baseUrl: 'https://www.104.com.tw/jobs/search/',
  selectors: {
    applyButton: '.apply-button__button',
    jobContainer: '[class*="job-list-container"]',
    jobTitle: 'a[href*="/job/"]',
    alreadyAppliedText: '已應徵',
    coverLetterDropdown: 'span:has-text("系統預設")',
    coverLetterOption: '.multiselect__option',
    submitButton: 'button:has-text("確認送出")',
    successUrlPattern: '/job/apply/done/'
  },
  delays: {
    afterClick: 1500,
    afterNavigation: 2000,
    betweenJobs: { min: 1800, max: 3000 },
    dropdownWait: 500,
    skipDelay: 400
  },
  authentication: {
    email: 'user@example.com',
    coverLetter: '自訂推薦信1'
  }
};
```

### 2.2 Abstraction Pattern
Create reusable functions:

```javascript
async function applyToJob(page, jobIndex, config) {
  // 1. Get job info
  const jobInfo = await getJobInfo(page, jobIndex, config);

  // 2. Check if already applied
  if (jobInfo.alreadyApplied) {
    await page.waitForTimeout(config.delays.skipDelay);
    return { status: 'skipped', reason: 'Already applied' };
  }

  // 3. Open application
  await clickApplyButton(page, jobIndex, config);

  // 4. Switch to new tab
  const newTab = await switchToNewestTab(page);

  // 5. Fill form
  await selectCoverLetter(newTab, config);

  // 6. Submit
  await submitApplication(newTab, config);

  // 7. Verify success
  const success = await verifySuccess(newTab, config);

  // 8. Cleanup
  await closeTabAndReturn(newTab, page);

  return success ?
    { status: 'success', job: jobInfo } :
    { status: 'failed', job: jobInfo };
}
```

### 2.3 State Management
Track state clearly:

```javascript
class AutomationSession {
  constructor(target, startPage = 1) {
    this.target = target;
    this.currentPage = startPage;
    this.results = {
      successful: 0,
      failed: 0,
      skipped: 0,
      pages: []
    };
    this.startTime = Date.now();
  }

  recordSuccess(job) {
    this.results.successful++;
  }

  recordFailure(job, error) {
    this.results.failed++;
  }

  recordSkip(job, reason) {
    this.results.skipped++;
  }

  isTargetReached() {
    return this.results.successful >= this.target;
  }

  getDuration() {
    return (Date.now() - this.startTime) / 1000 / 60; // minutes
  }

  getSuccessRate() {
    const total = this.results.successful + this.results.failed;
    return total > 0 ? (this.results.successful / total * 100).toFixed(1) : 0;
  }

  getSummary() {
    return {
      ...this.results,
      successRate: this.getSuccessRate(),
      duration: this.getDuration()
    };
  }
}
```

---

## 3. Skill Implementation Steps

### Step 1: Create Skill File
Create `.claude/skills/job-application-automation.md`:

```markdown
# Job Application Automation

Automate job applications on job boards like 104.com.tw.

## Usage
```
/apply-jobs <target> [start-page]
```

Examples:
- `/apply-jobs 50` - Apply to 50 jobs starting from page 1
- `/apply-jobs 200 10` - Apply to 200 jobs starting from page 10

## What I Do
1. Navigate through job search results
2. Skip jobs you've already applied to
3. Fill out application forms automatically
4. Track success/failure/skip metrics
5. Stop exactly when target is reached

## Requirements
- Must be logged into the job board
- Cover letter must be set up in your account
- Browser must be available

## Output
- Real-time progress updates
- Page-by-page statistics
- Final summary with success rate
- Results saved to file
```

### Step 2: Create Configuration File
Create `job-automation-config.json`:

```json
{
  "sites": {
    "104": {
      "name": "104.com.tw",
      "baseUrl": "https://www.104.com.tw/jobs/search/",
      "selectors": {
        "applyButton": ".apply-button__button",
        "jobContainer": "[class*=\"job-list-container\"]",
        "alreadyAppliedText": "已應徵"
      },
      "credentials": {
        "email": "your-email@example.com",
        "coverLetter": "自訂推薦信1"
      }
    }
  },
  "defaults": {
    "delayMin": 1800,
    "delayMax": 3000,
    "pageTimeout": 30000
  }
}
```

### Step 3: Create Core Automation Script
Create `job-automation-core.js`:

```javascript
/**
 * Core job application automation functions
 * Can be used across different job boards
 */

class JobAutomation {
  constructor(page, config) {
    this.page = page;
    this.config = config;
  }

  async processPage(pageNum) {
    const results = { successful: 0, failed: 0, skipped: 0 };

    await this.navigateToPage(pageNum);
    const jobCount = await this.countJobs();

    for (let i = 0; i < jobCount; i++) {
      try {
        const result = await this.processJob(i);
        results[result.status]++;
      } catch (error) {
        results.failed++;
        await this.cleanup();
      }

      await this.randomDelay();
    }

    return results;
  }

  async processJob(index) {
    const jobInfo = await this.getJobInfo(index);

    if (jobInfo.alreadyApplied) {
      return { status: 'skipped' };
    }

    await this.applyToJob(index);
    return { status: 'successful' };
  }

  // ... implementation details
}

module.exports = { JobAutomation };
```

---

## 4. Best Practices for Skills

### 4.1 Make It Reusable
**Pattern**: Separate site-specific details from core logic.

```javascript
// ✅ GOOD - Site-specific config separate
const config = loadConfig('104');
const automation = new JobAutomation(page, config);

// ❌ BAD - Hard-coded values
const applyButton = document.querySelector('.apply-button__button'); // Specific to 104
```

### 4.2 Clear User Communication
**Pattern**: Provide progress updates and clear instructions.

```markdown
## Starting Job Application Automation

Target: 50 applications
Starting: Page 1

Press Ctrl+C to stop at any time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Page 1: Processing 22 jobs
  Job 1/22: Software Engineer at TechCo
    ✅ SUCCESS
  Job 2/22: Backend Developer at StartupInc
    ⏭️  SKIPPED (already applied)
  ...

Page 1 Summary: ✅17 ❌2 ⏭️3
Progress: 17/50 (34%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4.3 Error Resilience
**Pattern**: Never let one error stop everything.

```javascript
for (let pageNum = start; pageNum <= max; pageNum++) {
  try {
    await processPage(pageNum);
  } catch (pageError) {
    console.error(`Page ${pageNum} failed: ${pageError.message}`);
    // Log but continue to next page
    continue;
  }

  for (let job of jobs) {
    try {
      await processJob(job);
    } catch (jobError) {
      console.error(`Job failed: ${jobError.message}`);
      // Log but continue to next job
      continue;
    }
  }
}
```

### 4.4 Provide Clear Results
**Pattern**: Always save results and provide summary.

```javascript
async function saveResults(results, filename) {
  const summary = {
    timestamp: new Date().toISOString(),
    target: results.target,
    successful: results.successful,
    failed: results.failed,
    skipped: results.skipped,
    successRate: results.getSuccessRate(),
    duration: results.getDuration(),
    pages: results.pages
  };

  await writeFile(filename, JSON.stringify(summary, null, 2));

  console.log('\n╔══════════════════════════════════════════════╗');
  console.log('║     AUTOMATION COMPLETE                      ║');
  console.log('╚══════════════════════════════════════════════╝');
  console.log(`✅ Successful: ${summary.successful}`);
  console.log(`❌ Failed: ${summary.failed}`);
  console.log(`⏭️  Skipped: ${summary.skipped}`);
  console.log(`🎯 Success Rate: ${summary.successRate}%`);
  console.log(`⏱️  Duration: ${summary.duration.toFixed(1)} minutes`);
  console.log(`\nResults saved to: ${filename}`);
}
```

---

## 5. Skill Parameters

### 5.1 Required Parameters
```javascript
{
  target: number,           // Number of jobs to apply to
  startPage: number = 1,    // Starting page number
  site: string = '104'      // Job board identifier
}
```

### 5.2 Optional Parameters
```javascript
{
  maxPages: number = 200,   // Maximum pages to process
  coverLetter: string,      // Override default cover letter
  keywords: string,         // Search keywords
  location: string,         // Job location filter
  dryRun: boolean = false   // Test mode (no actual applications)
}
```

### 5.3 Usage Examples
```bash
# Basic usage
/apply-jobs 50

# Specify starting page
/apply-jobs 100 10

# With all options
/apply-jobs --target=200 --start=1 --site=104 --keywords="軟體工程師"
```

---

## 6. Testing Strategy

### 6.1 Unit Tests
Test individual functions:

```javascript
describe('JobAutomation', () => {
  it('should detect already-applied jobs', async () => {
    const jobInfo = await automation.getJobInfo(0);
    expect(jobInfo.alreadyApplied).toBe(true);
  });

  it('should extract job title correctly', async () => {
    const jobInfo = await automation.getJobInfo(0);
    expect(jobInfo.title).toBe('Software Engineer');
  });
});
```

### 6.2 Integration Tests
Test complete flows:

```javascript
describe('Application Flow', () => {
  it('should complete full application', async () => {
    const result = await automation.applyToJob(0);
    expect(result.status).toBe('successful');
  });

  it('should handle tab management', async () => {
    await automation.applyToJob(0);
    const tabs = await page.context().pages();
    expect(tabs.length).toBe(1); // Should cleanup
  });
});
```

### 6.3 Dry Run Mode
Test without actually applying:

```javascript
if (config.dryRun) {
  console.log('[DRY RUN] Would apply to:', jobInfo.title);
  return { status: 'success', dryRun: true };
}
```

---

## 7. Documentation Template

### 7.1 README Structure
```markdown
# Job Application Automation Skill

## Overview
Automates job applications on supported job boards.

## Supported Sites
- ✅ 104.com.tw (Taiwan)
- 🚧 LinkedIn (coming soon)
- 🚧 Indeed (coming soon)

## Installation
1. Ensure Playwright MCP tools are available
2. Copy skill files to `.claude/skills/`
3. Configure credentials in `config.json`

## Usage
```bash
/apply-jobs <target> [options]
```

## Configuration
Edit `config.json`:
```json
{
  "credentials": {
    "email": "your-email@example.com",
    "coverLetter": "Your Cover Letter Name"
  }
}
```

## Examples
```bash
# Apply to 50 jobs
/apply-jobs 50

# Apply to 200 jobs starting from page 10
/apply-jobs 200 --start=10

# Dry run (test without applying)
/apply-jobs 10 --dry-run
```

## Results
Results are saved to `automation_results_TIMESTAMP.txt`

## Troubleshooting
**No jobs found**: Check login status
**High failure rate**: Check selectors (site may have changed)
**Browser crashes**: Reduce batch size

## Performance
- Average: 600-700 applications/hour
- Success rate: 90-95%
- Handles duplicates automatically

## Safety Features
- Skips already-applied jobs
- Human-like delays
- Error recovery
- Progress tracking
- Can stop at any time
```

---

## 8. Deployment Checklist

### 8.1 Pre-Deployment
- [ ] Test on small batch (10 jobs)
- [ ] Verify all selectors work
- [ ] Check login persistence
- [ ] Test error recovery
- [ ] Validate output format

### 8.2 Configuration
- [ ] Set correct credentials
- [ ] Configure cover letter name
- [ ] Set appropriate delays
- [ ] Define timeout values
- [ ] Set batch sizes

### 8.3 Documentation
- [ ] Write clear README
- [ ] Document all parameters
- [ ] Provide examples
- [ ] Add troubleshooting guide
- [ ] Include success metrics

### 8.4 Monitoring
- [ ] Log all actions
- [ ] Track success rates
- [ ] Save results to file
- [ ] Alert on failures
- [ ] Monitor performance

---

## 9. Advanced Features

### 9.1 Resume Capability
```javascript
// Save progress periodically
async function saveCheckpoint(session) {
  await writeFile('checkpoint.json', JSON.stringify({
    lastPage: session.currentPage,
    lastJob: session.currentJob,
    results: session.results,
    timestamp: Date.now()
  }));
}

// Resume from checkpoint
async function resume() {
  const checkpoint = JSON.parse(await readFile('checkpoint.json'));
  return new AutomationSession({
    ...checkpoint,
    startPage: checkpoint.lastPage,
    startJob: checkpoint.lastJob
  });
}
```

### 9.2 Multi-Site Support
```javascript
const sites = {
  '104': require('./sites/104-automation'),
  'linkedin': require('./sites/linkedin-automation'),
  'indeed': require('./sites/indeed-automation')
};

async function applyJobs(site, target) {
  const automation = new sites[site](page);
  return automation.run(target);
}
```

### 9.3 Intelligent Retry
```javascript
async function applyWithRetry(job, maxRetries = 2) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await applyToJob(job);
    } catch (error) {
      if (attempt === maxRetries) throw error;

      console.log(`Retry ${attempt}/${maxRetries} for ${job.title}`);
      await page.waitForTimeout(5000); // Wait before retry
    }
  }
}
```

---

## 10. Maintenance Guide

### 10.1 Regular Checks
- **Weekly**: Verify selectors still work
- **Monthly**: Check for UI changes
- **Quarterly**: Review and update documentation

### 10.2 Updating Selectors
```javascript
// When site updates, update config
const SELECTORS = {
  applyButton: '.apply-button__button', // Update this if changed
  // ... other selectors
};

// Test new selectors
async function validateSelectors(page) {
  for (const [name, selector] of Object.entries(SELECTORS)) {
    const element = await page.$(selector);
    if (!element) {
      console.error(`Selector broken: ${name} (${selector})`);
      return false;
    }
  }
  return true;
}
```

### 10.3 Version Control
```
job-automation/
├── README.md
├── CHANGELOG.md          # Track all changes
├── package.json          # Version number
├── src/
│   ├── core.js          # Core logic (rarely changes)
│   ├── sites/
│   │   ├── 104.js       # Site-specific (may change)
│   │   └── config.json  # Selectors (may change frequently)
│   └── utils.js
└── tests/
    └── *.test.js
```

---

## 11. Final Recommendations

### For Claude Code Skills:
1. **Keep it modular** - Separate concerns clearly
2. **Make it configurable** - Don't hard-code values
3. **Provide clear feedback** - Users should know what's happening
4. **Handle errors gracefully** - Never crash completely
5. **Document thoroughly** - Future you will thank present you
6. **Test extensively** - Start small, scale up
7. **Version everything** - Track changes
8. **Monitor performance** - Know when something breaks

### Success Metrics:
- ✅ 90%+ success rate
- ✅ <10% failure rate
- ✅ Clear progress tracking
- ✅ Automatic error recovery
- ✅ Stops at exact target
- ✅ Detailed results saved

### Red Flags:
- ❌ Hard-coded selectors
- ❌ No error handling
- ❌ Fixed delays only
- ❌ No progress visibility
- ❌ Crashes on first error
- ❌ No result tracking

---

## Conclusion

Creating a skill requires:
1. **Understanding** the domain deeply
2. **Abstracting** patterns into reusable components
3. **Documenting** everything clearly
4. **Testing** thoroughly
5. **Maintaining** regularly

This 104.com.tw automation demonstrates all these principles and achieved:
- 2,495 successful applications
- 92.5% success rate
- 5 successful runs
- 0 manual intervention

Follow these patterns and you can create reliable, maintainable skills for any automation task.
