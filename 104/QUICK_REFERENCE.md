# 104.com.tw Automation - Quick Reference Guide

## Critical Code Snippets

### 1. Apply Button Click
```javascript
// ✅ CORRECT
document.querySelectorAll('.apply-button__button')[index].click();

// ❌ WRONG
document.querySelectorAll('button')[index].click();
```

### 2. Tab Management
```javascript
// Click apply → new tab opens
await page.evaluate((i) => {
  document.querySelectorAll('.apply-button__button')[i].click();
}, jobIndex);

await page.waitForTimeout(1500);

// Get newest tab
const allPages = page.context().pages();
const newTab = allPages[allPages.length - 1];
await newTab.bringToFront();

// ALWAYS cleanup
await newTab.close();
await page.bringToFront();
```

### 3. Cover Letter Selection
```javascript
// Step 1: Open dropdown
await page.evaluate(() => {
  const span = Array.from(document.querySelectorAll('span'))
    .find(el => el.textContent === '系統預設' && el.tagName === 'SPAN');
  if (span?.parentElement) span.parentElement.click();
});
await page.waitForTimeout(500);

// Step 2: Select option
await page.evaluate(() => {
  document.querySelectorAll('.multiselect__option').forEach(opt => {
    if (opt.textContent.trim() === '自訂推薦信1') opt.click();
  });
});
```

### 4. Submit & Verify
```javascript
// Submit
await page.evaluate(() => {
  document.querySelectorAll('button').forEach(btn => {
    if (btn.textContent.includes('確認送出')) btn.click();
  });
});

await page.waitForTimeout(2000);

// Verify success
const finalUrl = page.url();
if (finalUrl.includes('/job/apply/done/')) {
  // SUCCESS
}
```

### 5. Skip Already Applied
```javascript
const jobInfo = await page.evaluate((index) => {
  const container = document.querySelectorAll('[class*="job-list-container"]')[index];
  return {
    alreadyApplied: container.textContent.includes('已應徵')
  };
}, jobIndex);

if (jobInfo.alreadyApplied) {
  await page.waitForTimeout(400);
  continue;
}
```

---

## Timing Cheatsheet

| Action | Delay | Reason |
|--------|-------|--------|
| After navigation | 2000-3000ms | Wait for page load |
| After button click | 1500ms | Wait for new tab |
| Between jobs | 1800-3000ms | Human-like |
| After dropdown open | 500ms | UI update |
| Skip (already applied) | 400ms | Quick skip |
| After submit | 2000ms | Form processing |

---

## Selectors Reference

```javascript
const SELECTORS = {
  // Job listing
  applyButton: '.apply-button__button',
  jobContainer: '[class*="job-list-container"]',
  jobTitle: 'a[href*="/job/"]',

  // Application form
  coverLetterDropdown: 'span:has-text("系統預設")',
  coverLetterOption: '.multiselect__option',
  submitButton: 'button:has-text("確認送出")',

  // Status checks
  alreadyAppliedText: '已應徵',
  successUrlPattern: '/job/apply/done/'
};
```

---

## Error Handling Template

```javascript
try {
  // Job processing logic
  await applyToJob(jobIndex);
  results.successful++;

} catch (error) {
  console.error(`❌ FAILED: ${error.message}`);
  results.failed++;

  // CRITICAL: Always cleanup
  try {
    const allPages = page.context().pages();
    if (allPages.length > 1) {
      await allPages[allPages.length - 1].close();
    }
    await page.bringToFront();
  } catch (cleanupError) {
    // Swallow cleanup errors
  }
}
```

---

## Progress Logging Template

```javascript
console.log(`\n╔════════════════════════════════════════╗`);
console.log(`║  PAGE ${pageNum} | Progress: ${count}/${target}  ║`);
console.log(`╚════════════════════════════════════════╝\n`);

console.log(`━━━━ Job ${index + 1}/${total} ━━━━`);
console.log(`📋 ${jobTitle}`);
console.log('✅ SUCCESS' || '❌ FAILED' || '⏭️  SKIPPED');

console.log(`\nPage ${pageNum}: ✅${success} ❌${failed} ⏭️${skipped}\n`);
```

---

## Results Tracking Structure

```javascript
const results = {
  totalSuccessful: 0,
  totalFailed: 0,
  totalSkipped: 0,
  pages: [
    {
      page: 1,
      successful: 17,
      failed: 2,
      skipped: 3
    }
  ]
};

// Success rate calculation
const successRate = (results.totalSuccessful /
  (results.totalSuccessful + results.totalFailed) * 100).toFixed(1);
```

---

## Target-Based Loop

```javascript
for (let pageNum = startPage; pageNum <= maxPages; pageNum++) {
  // Check target at page level
  if (results.totalSuccessful >= targetSuccessful) {
    console.log('🎯 Target reached!');
    break;
  }

  // Process page...

  for (let jobIndex = 0; jobIndex < jobCount; jobIndex++) {
    // Check target at job level too
    if (results.totalSuccessful >= targetSuccessful) break;

    // Process job...
    if (success) results.totalSuccessful++;
  }
}
```

---

## Browser Reset

```bash
# Kill Chrome processes
pkill -f "Google Chrome" || true

# Wait a moment
sleep 2

# Start fresh session
await page.goto(url);
```

---

## Performance Benchmarks

### Excellent
- Success rate: 95%+
- Failures: <5%
- Speed: 16+ jobs/page

### Good
- Success rate: 90-95%
- Failures: 5-10%
- Speed: 13-16 jobs/page

### Poor (investigate)
- Success rate: <90%
- Failures: >10%
- Speed: <13 jobs/page

---

## Troubleshooting Quick Fixes

### Problem: "No jobs found"
```javascript
// Solution: Click search button
await page.click('button:has-text("搜尋")');
await page.waitForTimeout(3000);
```

### Problem: "Apply button not found"
```javascript
// Check login status
const isLoggedIn = await page.evaluate(() => {
  return document.body.textContent.includes('***REMOVED***');
});
```

### Problem: "High failure rate"
```javascript
// Verify selectors still work
const applyButtons = await page.$$('.apply-button__button');
console.log(`Found ${applyButtons.length} apply buttons`);
```

### Problem: "Browser crashes"
```javascript
// Reduce batch size
const maxPages = 30; // Instead of 100+

// Or add more delays
await page.waitForTimeout(5000); // Between pages
```

---

## Command Templates

### Small Batch (10-20 jobs)
```javascript
await autoApply(page, { target: 20, startPage: 1 });
```

### Medium Batch (50-100 jobs)
```javascript
await autoApply(page, { target: 100, startPage: 1 });
```

### Large Batch (200+ jobs)
```javascript
await autoApply(page, { target: 500, startPage: 1, maxPages: 50 });
```

### Resume from Page X
```javascript
await autoApply(page, { target: 100, startPage: 51 });
```

---

## Configuration Values

```javascript
const CONFIG = {
  // Account (SET THESE SECURELY - DO NOT COMMIT!)
  email: process.env.JOB_EMAIL || '<YOUR_EMAIL>',
  coverLetter: '自訂推薦信1',

  // URLs
  baseUrl: 'https://www.104.com.tw/jobs/search/',
  searchUrl: '?area=6001001000,6001002000&keyword=%E8%BB%9F%E9%AB%94%E5%B7%A5%E7%A8%8B%E5%B8%AB&order=15',

  // Timing
  delayMin: 1800,
  delayMax: 3000,
  pageTimeout: 30000,

  // Limits
  maxPages: 200,
  maxRetries: 2
};
```

---

## Success Metrics Summary

### Achieved Results
- **Total**: 2,495 applications
- **Success Rate**: 92.5%
- **Best Run**: 96.2%
- **Speed**: 609 apps/hour

### Per-Run Performance
1. Run 1: 91.4% (427 jobs)
2. Run 2: 88.6% (895 jobs)
3. Run 3: 95.3% (223 jobs)
4. Run 4: 92.0% (900 jobs)
5. Run 5: 96.2% (50 jobs) ⭐ Best!

---

## File Structure

```
104/
├── LEARNINGS.md              # This file
├── CLAUDE_SKILL_GUIDE.md     # Skill creation guide
├── QUICK_REFERENCE.md        # Quick lookup
├── TOTAL_RESULTS.txt         # All results
├── automation_results_*.txt  # Individual run results
├── note.md                   # Technical notes
├── README_104_AUTOMATION.md  # User guide
└── 104_auto_apply_with_controls.js  # Working script
```

---

## Most Common Mistakes to Avoid

1. ❌ Using generic button selectors
2. ❌ Not waiting for dynamic content
3. ❌ Forgetting to close tabs
4. ❌ Hard-coding delays (use random)
5. ❌ Not checking already-applied status
6. ❌ Relying on page text for success
7. ❌ Not handling errors gracefully
8. ❌ Processing too many pages at once

---

## One-Liners

```javascript
// Count jobs on page
await page.evaluate(() => document.querySelectorAll('.apply-button__button').length);

// Check if logged in
await page.evaluate(() => document.body.textContent.includes('***REMOVED***'));

// Get total jobs
await page.evaluate(() => document.body.textContent.match(/共\s*(\d+)/)?.[1]);

// Random delay
await page.waitForTimeout(1800 + Math.random() * 1200);

// Close all extra tabs
const pages = page.context().pages();
for (let i = 1; i < pages.length; i++) await pages[i].close();
```

---

## Remember

- **Read before edit** - Always get job info first
- **Always cleanup** - Close tabs, handle errors
- **Be patient** - Wait for content to load
- **Track everything** - Log all results
- **Test incrementally** - Start small, scale up

---

## Quick Start

```javascript
// 1. Kill Chrome
pkill -f "Google Chrome"

// 2. Navigate to page 1
await page.goto('https://www.104.com.tw/jobs/search/...');

// 3. Click search if needed
await page.click('button:has-text("搜尋")');

// 4. Start automation
await autoApply(page, { target: 50, startPage: 1 });
```

---

## Support

For issues:
1. Check this guide first
2. Review LEARNINGS.md
3. Check selector validity
4. Verify login status
5. Try smaller batch
6. Check browser console
