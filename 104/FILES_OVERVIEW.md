# 104.com.tw Automation - Files Overview

## 📁 All Files

### Core Documentation
| File | Purpose | Use When |
|------|---------|----------|
| `README_104_AUTOMATION.md` | **START HERE** - Complete guide | First time setup |
| `CLAUDE.md` | Technical documentation | Need implementation details |
| `run_automation.md` | Detailed usage instructions | Running the automation |
| `FILES_OVERVIEW.md` | This file - Quick reference | Need to find something |

### Scripts
| File | Purpose | Use When |
|------|---------|----------|
| `test_automation.js` | Test on 3 jobs only | **First time / Testing** |
| `apply_single_job.js` | Apply to one job manually | Manual control needed |
| `104_auto_apply.js` | Full automation | Ready for batch processing |

### Generated Files
| File | Purpose | Created When |
|------|---------|--------------|
| `application_log.json` | Detailed results log | After automation runs |
| `.playwright-mcp/` | Screenshots & logs | During execution |

---

## 🚀 Quick Start Paths

### Path 1: First Time User (Recommended)
1. Read `README_104_AUTOMATION.md`
2. Review `CLAUDE.md` for technical details
3. Run `test_automation.js` on 3 jobs
4. If successful, run `104_auto_apply.js`

### Path 2: Experienced User
1. Configure `104_auto_apply.js`
2. Run full automation
3. Check `application_log.json`

### Path 3: Manual Control
1. Use `apply_single_job.js` functions
2. Execute step-by-step via browser console
3. Full control over each application

---

## 📖 File Details

### 1. README_104_AUTOMATION.md
**What:** Complete user guide
**Includes:**
- Quick start instructions
- Manual step-by-step guide
- Full automation setup
- Troubleshooting
- Best practices

**Read this if:** You're new to the system

---

### 2. CLAUDE.md
**What:** Technical documentation
**Includes:**
- Login process details
- Button selectors (`.apply-button__button`)
- Form interaction code
- Tab management
- Key learnings from testing

**Read this if:** You need to understand how it works or modify the code

---

### 3. run_automation.md
**What:** Detailed usage instructions
**Includes:**
- Configuration options
- Execution methods
- Monitoring tips
- Safety features
- Troubleshooting guide

**Read this if:** You're ready to run the automation

---

### 4. test_automation.js
**What:** Safe test script (3 jobs only)
**Features:**
- ✅ Limited to 3 jobs
- ✅ Detailed step-by-step logging
- ✅ Test summary
- ✅ Safe for first-time testing

**Use this when:**
- First time using the system
- Testing after code changes
- Verifying account setup
- Debugging issues

**How to use:**
```javascript
// Via Playwright MCP evaluate
await page.evaluate(() => {
  // Load script...
  // Then run test
});
```

---

### 5. apply_single_job.js
**What:** Manual control helper
**Features:**
- ✅ Apply to one job at a time
- ✅ Step-by-step functions
- ✅ List jobs helper
- ✅ Browser console compatible

**Use this when:**
- Want full manual control
- Applying to specific jobs
- Debugging a single job
- Testing without automation

**Functions:**
```javascript
apply104.listJobs()              // See all jobs
apply104.clickApply(0)           // Apply to job 0
apply104.selectCoverLetter()     // Select cover letter
apply104.submitApplication()     // Submit form
```

---

### 6. 104_auto_apply.js
**What:** Full automation script
**Features:**
- ✅ Multi-page processing
- ✅ Error handling
- ✅ JSON logging
- ✅ Skip already applied
- ✅ Random delays
- ✅ Configurable limits

**Use this when:**
- Ready for full automation
- Processing many jobs
- Want hands-off execution
- Need detailed logs

**Configuration:**
```javascript
const CONFIG = {
  searchUrl: 'YOUR_URL',
  coverLetter: '自訂推薦信1',
  maxPages: 10,
  delayBetweenJobs: { min: 2000, max: 4000 }
};
```

---

### 7. application_log.json
**What:** Execution results
**Created:** Automatically after automation runs
**Contains:**
- Start/end times
- Success/failure counts
- Job details
- Error messages
- Timestamps

**Example:**
```json
{
  "successful": 18,
  "failed": 2,
  "skipped": 2,
  "jobs": [...]
}
```

---

## 🎯 Recommended Workflow

### First Time Setup
```
1. Read README_104_AUTOMATION.md
   ↓
2. Review CLAUDE.md technical details
   ↓
3. Verify 104 account setup
   - Resume uploaded
   - Cover letter created ("自訂推薦信1")
   - Logged in
   ↓
4. Run test_automation.js (3 jobs)
   ↓
5. Check results
   ↓
6. If successful → Run 104_auto_apply.js
   If failed → Check CLAUDE.md for fixes
```

### Regular Usage
```
1. Configure 104_auto_apply.js
   ↓
2. Run automation
   ↓
3. Monitor console output
   ↓
4. Review application_log.json
   ↓
5. Check 104 account for confirmations
```

---

## 🔍 Which File for Which Question?

| Question | File to Check |
|----------|---------------|
| How do I get started? | `README_104_AUTOMATION.md` |
| What's the button selector? | `CLAUDE.md` |
| How to configure delays? | `run_automation.md` or `104_auto_apply.js` |
| How to test safely? | `test_automation.js` |
| How to apply manually? | `apply_single_job.js` |
| Why did it fail? | `application_log.json` |
| What's available? | `FILES_OVERVIEW.md` (this file) |

---

## ⚡ Quick Commands

### List all jobs on page
```javascript
const jobs = await page.evaluate(() => {
  const containers = document.querySelectorAll('[class*="job-list-container"]');
  return Array.from(containers).map((c, i) => ({
    index: i,
    title: c.querySelector('a[href*="/job/"]')?.textContent.trim()
  }));
});
console.table(jobs);
```

### Apply to job 0
```javascript
await page.evaluate(() => {
  document.querySelectorAll('.apply-button__button')[0].click();
});
```

### Check if logged in
```javascript
await page.evaluate(() => {
  return !!document.querySelector('[class*="personal"]');
});
```

---

## 📊 File Size Reference

| File | Lines | Purpose |
|------|-------|---------|
| 104_auto_apply.js | ~450 | Full automation |
| test_automation.js | ~300 | Testing |
| apply_single_job.js | ~150 | Manual control |
| README_104_AUTOMATION.md | ~400 | Documentation |
| CLAUDE.md | ~265 | Technical guide |
| run_automation.md | ~200 | Usage guide |

---

## 🎓 Learning Path

1. **Beginner:** Start with `README_104_AUTOMATION.md`
2. **Intermediate:** Use `test_automation.js` and review results
3. **Advanced:** Configure and run `104_auto_apply.js`
4. **Expert:** Modify scripts using `CLAUDE.md` as reference

---

## ✅ Pre-flight Checklist

Before running any automation:

- [ ] Read README_104_AUTOMATION.md
- [ ] Logged in to 104.com.tw
- [ ] Resume uploaded
- [ ] Cover letter "自訂推薦信1" exists
- [ ] Tested with test_automation.js
- [ ] Understood the process
- [ ] Ready to monitor execution
- [ ] Backup plan if issues occur

---

## 🆘 Emergency Stops

If something goes wrong:

1. **Stop execution:** Close browser or press Ctrl+C
2. **Check logs:** Open `application_log.json`
3. **Review console:** Look for error messages
4. **Check 104 account:** See which jobs were applied to
5. **Fix issue:** Use CLAUDE.md for technical help
6. **Resume:** Start from last successful page/job

---

## 📝 Notes

- All scripts are educational and for personal use
- Use responsibly and ethically
- Only apply to jobs you're qualified for
- Respect 104.com.tw terms of service
- Monitor execution to ensure quality applications

---

**Last Updated:** 2026-02-25
**Version:** 1.0
**Status:** ✅ Tested and Working
