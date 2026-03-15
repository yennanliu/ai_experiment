# 104.com.tw Job Application Automation

Complete automation system for applying to jobs on 104.com.tw using Playwright MCP tools.

## 📁 Project Files

```
.
├── CLAUDE.md                    # Technical documentation & implementation guide
├── 104_auto_apply.js           # Full automation script (Node.js/Playwright)
├── apply_single_job.js         # Single-job helper (browser console)
├── run_automation.md           # Detailed usage instructions
└── README_104_AUTOMATION.md    # This file
```

## 🎯 Quick Start

### Option 1: Manual Step-by-Step (Recommended for First Time)

1. **Navigate to job search page**
2. **List all jobs** to see what's available
3. **Apply to one job** to test
4. **Repeat** for more jobs

See detailed steps below in "Manual Execution Guide"

### Option 2: Full Automation

Use the `104_auto_apply.js` script to automate multiple pages of job applications.

---

## 📋 Prerequisites

### Account Setup
- ✅ 104.com.tw account (configured separately for security)
- ✅ Resume uploaded
- ✅ Cover letter created ("自訂推薦信1")
- ✅ Already logged in to 104.com.tw

### Technical Requirements
- Playwright MCP tools (available in your environment)
- Browser automation access
- Stable internet connection

---

## 🔧 Manual Execution Guide

### Step 1: Navigate to Job Search

Use Playwright MCP tools to navigate:
```javascript
await page.goto('https://www.104.com.tw/jobs/search/?area=6001001000,6001002000&jobsource=joblist_search&keyword=%20%20%20%20%E8%BB%9F%E9%AB%94%E5%B7%A5%E7%A8%8B%E5%B8%AB&order=15&page=1&remoteWork=1,2');
```

### Step 2: List Available Jobs

```javascript
await page.evaluate(() => {
  const containers = document.querySelectorAll('[class*="job-list-container"]');
  const jobs = [];

  containers.forEach((container, index) => {
    const titleLink = container.querySelector('a[href*="/job/"]');
    const title = titleLink ? titleLink.textContent.trim() : 'Unknown';

    jobs.push({
      index: index,
      title: title.substring(0, 60)
    });
  });

  console.table(jobs);
  return jobs;
});
```

### Step 3: Apply to a Job

**3.1 Click Apply Button (Job Index 0)**
```javascript
await page.evaluate(() => {
  const buttons = document.querySelectorAll('.apply-button__button');
  buttons[0].click(); // Apply to first job
});
```

**3.2 Switch to New Tab**
```javascript
// Get all pages/tabs
const pages = await context.pages();
const newTab = pages[pages.length - 1];
await newTab.bringToFront();
```

**3.3 Select Cover Letter**
```javascript
// Click dropdown
await newTab.evaluate(() => {
  const span = Array.from(document.querySelectorAll('span'))
    .find(el => el.textContent === '系統預設');
  if (span) span.parentElement.click();
});

await newTab.waitForTimeout(500);

// Select option
await newTab.evaluate(() => {
  const options = document.querySelectorAll('.multiselect__option');
  options.forEach(option => {
    if (option.textContent.trim() === '自訂推薦信1') {
      option.click();
    }
  });
});
```

**3.4 Submit Application**
```javascript
await newTab.evaluate(() => {
  const buttons = document.querySelectorAll('button');
  buttons.forEach(btn => {
    if (btn.textContent.includes('確認送出')) {
      btn.click();
    }
  });
});

await newTab.waitForTimeout(2000);
```

**3.5 Verify Success**
```javascript
const url = newTab.url();
console.log('Success:', url.includes('/job/apply/done/'));
```

**3.6 Close Tab and Return**
```javascript
await newTab.close();
await page.bringToFront();
```

### Step 4: Repeat for More Jobs

Change the index in step 3.1 to apply to different jobs:
- Job 0: `buttons[0].click()`
- Job 1: `buttons[1].click()`
- Job 2: `buttons[2].click()`
- etc.

---

## 🤖 Full Automation

### Using the Complete Script

The `104_auto_apply.js` provides full automation with:
- ✅ Multi-page support
- ✅ Error handling
- ✅ Logging to JSON file
- ✅ Skip already applied jobs
- ✅ Random delays between applications
- ✅ Success/failure tracking

### Configuration

Edit `104_auto_apply.js`:
```javascript
const CONFIG = {
  searchUrl: 'YOUR_SEARCH_URL',
  coverLetter: '自訂推薦信1',
  maxPages: 10,
  delayBetweenJobs: { min: 2000, max: 4000 }
};
```

### Expected Output

```
🚀 Starting 104.com.tw Job Application Automation...

========== PAGE 1 ==========
Found 22 jobs on this page

--- Job 1/22 ---
📋 Job: Software Engineer
🏢 Company: Tech Corp
📝 Selected cover letter: 自訂推薦信1
✅ SUCCESS: Application submitted
⏳ Waiting 3.2s...

--- Job 2/22 ---
📋 Job: Backend Developer
🏢 Company: Startup Inc
⏭️  SKIPPED: Already applied

...

========================================
           AUTOMATION SUMMARY
========================================
⏱️  Duration: 180s
📊 Total Attempted: 22
✅ Successful: 18
❌ Failed: 2
⏭️  Skipped: 2
========================================
```

---

## 📊 Logging & Results

### Log File: `application_log.json`

```json
{
  "startTime": "2026-02-25T07:00:00Z",
  "endTime": "2026-02-25T07:03:00Z",
  "duration": 180,
  "totalAttempted": 22,
  "successful": 18,
  "failed": 2,
  "skipped": 2,
  "jobs": [
    {
      "title": "Software Engineer",
      "company": "Tech Corp",
      "status": "SUCCESS",
      "timestamp": "2026-02-25T07:00:15Z"
    },
    {
      "title": "Backend Developer",
      "company": "Startup Inc",
      "status": "SKIPPED",
      "reason": "Already applied",
      "timestamp": "2026-02-25T07:00:20Z"
    }
  ]
}
```

---

## ⚠️ Important Notes

### What Works
- ✅ Standard 104.com.tw application forms
- ✅ Jobs with "應徵" button visible
- ✅ Jobs that accept your resume/cover letter
- ✅ Multiple pages of results

### Limitations
- ❌ Cannot handle CAPTCHA (solve manually)
- ❌ Cannot handle custom application forms
- ❌ Cannot handle jobs requiring additional info
- ❌ Requires browser to stay open
- ❌ Must be logged in before starting

### Safety Features
- Skips already applied jobs automatically
- Random delays (2-4 seconds) between applications
- Error handling for each job
- Detailed logging for review
- Maximum page limit to prevent runaway execution

---

## 🐛 Troubleshooting

### Problem: "Apply button not found"
**Cause:** Page structure changed or not logged in
**Solution:** Check if logged in, refresh page

### Problem: "Cover letter not found"
**Cause:** Cover letter name mismatch
**Solution:** Verify exact name in your 104 account settings

### Problem: Applications failing
**Cause:** Network issues, rate limiting, or account issues
**Solution:**
1. Check internet connection
2. Increase delays between jobs
3. Verify account status
4. Try manual application to test

### Problem: Script stops unexpectedly
**Cause:** JavaScript error or page navigation issue
**Solution:**
1. Check console for errors
2. Review `application_log.json` for last successful job
3. Resume from that page/job

---

## 📝 Best Practices

### Before Running
1. ✅ Test with 1-2 jobs manually first
2. ✅ Verify cover letter and resume are correct
3. ✅ Check search criteria match your skills
4. ✅ Ensure stable internet connection

### During Execution
1. 👀 Monitor first few applications
2. 📊 Check logs periodically
3. ⏸️ Stop if seeing errors
4. 🔄 Adjust delays if needed

### After Completion
1. 📁 Review `application_log.json`
2. ✅ Verify successful applications in 104 account
3. 📧 Check email for confirmation/interview invites
4. 🗑️ Clean up browser tabs

---

## ⚖️ Legal & Ethical

**Important Disclaimer:**
- This tool is for **educational purposes**
- Use **responsibly** and in accordance with 104.com.tw Terms of Service
- Only apply to jobs you're **genuinely interested** in and **qualified** for
- Do **not** spam applications
- Respect **rate limits** and **server load**
- Be **honest** in your applications

**Recommended Usage:**
- Apply to 10-20 jobs per session maximum
- Take breaks between sessions
- Review each job listing before applying
- Customize cover letter for different job types

---

## 📚 Additional Resources

- **CLAUDE.md** - Technical implementation details
- **run_automation.md** - Detailed usage instructions
- **104.com.tw Help** - https://www.104.com.tw/faq/104-jobbank

---

## 🎉 Success!

If everything works correctly, you should see:
- ✅ Applications submitted successfully
- 📧 Email confirmations from 104
- 📱 Interview invites (hopefully!)

Good luck with your job search! 🚀

---

## 📧 Questions?

Refer to:
1. CLAUDE.md for technical details
2. run_automation.md for usage instructions
3. Console output for real-time debugging
4. application_log.json for execution history
