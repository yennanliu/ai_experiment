# 104.com.tw Automation - Key Learnings & Insights

## Overview
This document captures critical learnings from automating 2,495 job applications on 104.com.tw using Playwright MCP tools.

---

## 1. Technical Discoveries

### 1.1 Apply Button Structure
**Critical Finding**: The "應徵" (Apply) buttons are `<div>` elements, NOT `<button>` tags.

```javascript
// ❌ WRONG - Looking for button tags
document.querySelectorAll('button[class*="apply"]')

// ✅ CORRECT - Use specific class selector
document.querySelectorAll('.apply-button__button')
```

**Why This Matters**: Generic button selectors fail. Must use exact class names.

### 1.2 Tab Management Pattern
**Finding**: Clicking apply button opens a NEW TAB (not modal, not redirect).

```javascript
// Pattern that works:
// 1. Click apply button on main page
await page.evaluate((index) => {
  document.querySelectorAll('.apply-button__button')[index].click();
});

// 2. Wait for new tab to open
await page.waitForTimeout(1500);

// 3. Switch to the newest tab
const allPages = page.context().pages();
const newTab = allPages[allPages.length - 1];
await newTab.bringToFront();

// 4. Process application in new tab
// ... form filling ...

// 5. ALWAYS close tab and return
await newTab.close();
await page.bringToFront();
```

**Critical**: Always maintain proper tab cleanup to prevent memory leaks.

### 1.3 Dropdown Interaction Pattern
**Finding**: Cover letter selection requires two-step interaction.

```javascript
// Step 1: Click parent element to open dropdown
await page.evaluate(() => {
  const span = Array.from(document.querySelectorAll('span'))
    .find(el => el.textContent === '系統預設' && el.tagName === 'SPAN');
  if (span?.parentElement) span.parentElement.click();
});

await page.waitForTimeout(500); // CRITICAL: Wait for dropdown to appear

// Step 2: Click the option from the opened dropdown
await page.evaluate(() => {
  const options = document.querySelectorAll('.multiselect__option');
  options.forEach(opt => {
    if (opt.textContent.trim() === '自訂推薦信1') opt.click();
  });
});
```

**Key Lesson**: Must wait between dropdown open and option selection.

### 1.4 Success Verification
**Finding**: URL pattern is the most reliable success indicator.

```javascript
// ✅ RELIABLE - Check URL after submission
const finalUrl = newTab.url();
if (finalUrl.includes('/job/apply/done/')) {
  // Success!
}

// ❌ UNRELIABLE - Page text can be inconsistent
const pageText = await page.textContent('body');
if (pageText.includes('成功')) { ... } // Don't rely on this
```

### 1.5 Already-Applied Detection
**Finding**: Text-based detection is simple and effective.

```javascript
const jobInfo = await page.evaluate((index) => {
  const container = document.querySelectorAll('[class*="job-list-container"]')[index];
  const alreadyApplied = container.textContent.includes('已應徵');
  return { alreadyApplied };
}, jobIndex);

if (alreadyApplied) {
  // Skip this job - saves time
  await page.waitForTimeout(400); // Quick delay
  continue;
}
```

**Optimization**: Only 400ms delay for skipped jobs vs 2-3 seconds for applications.

---

## 2. Performance Optimizations

### 2.1 Timing Strategy
**Learnings**:
- **Between jobs**: 1800-3000ms (random) - appears human-like
- **After page navigation**: 2000-3000ms - ensures content loads
- **After button clicks**: 1500ms - ensures action completes
- **For dropdown interactions**: 500ms - allows UI to update
- **For skipped jobs**: 400ms - minimal delay needed

```javascript
// Human-like random delays
const delay = 1800 + Math.random() * 1200; // 1.8-3 seconds
await page.waitForTimeout(delay);

// Quick skip for already-applied
if (alreadyApplied) {
  await page.waitForTimeout(400);
  continue;
}
```

### 2.2 Batch Processing
**Finding**: Process in manageable batches for stability.

**Results**:
- Small batches (12-16 pages): Fast, fewer failures
- Medium batches (50 pages): Good balance
- Large batches (112 pages): Highest throughput but browser may crash

**Recommendation**: 30-50 page batches optimal.

### 2.3 Error Recovery
**Pattern**:
```javascript
try {
  // Application logic
} catch (error) {
  console.error(`❌ FAILED: ${error.message}`);
  results.failed++;

  // CRITICAL: Always cleanup
  try {
    const allPages = page.context().pages();
    if (allPages.length > 1) await allPages[allPages.length - 1].close();
    await page.bringToFront();
  } catch (e) {
    // Swallow cleanup errors
  }
}
```

**Key**: Never let one failure stop the entire process.

---

## 3. Architecture Patterns

### 3.1 Results Tracking Structure
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
    // ... more pages
  ]
};
```

**Why**: Provides detailed analytics per page + overall totals.

### 3.2 Target-Based Execution
```javascript
for (let pageNum = startPage; pageNum <= maxPages; pageNum++) {
  // Stop if target reached
  if (results.totalSuccessful >= targetSuccessful) {
    console.log(`🎯 Target reached!`);
    break;
  }

  // Process page...

  for (let jobIndex = 0; jobIndex < jobCount; jobIndex++) {
    // Also check inside job loop
    if (results.totalSuccessful >= targetSuccessful) break;

    // Process job...
  }
}
```

**Benefit**: Precise control - stops exactly when target is met.

### 3.3 Progressive Logging
```javascript
console.log(`\n╔════════════════════════════════════════╗`);
console.log(`║  PAGE ${pageNum} | Progress: ${successful}/${target}  ║`);
console.log(`╚════════════════════════════════════════╝\n`);

console.log(`━━━━ Job ${jobIndex + 1}/${jobCount} ━━━━`);
console.log(`📋 ${jobTitle}`);
console.log('✅ SUCCESS' || '❌ FAILED' || '⏭️  SKIPPED');
```

**Why**: Clear visual feedback for monitoring long-running processes.

---

## 4. Common Pitfalls & Solutions

### 4.1 Browser Context Loss
**Problem**: Browser crashes after long sessions.

**Solution**:
```javascript
// Kill existing Chrome processes before starting
await bash('pkill -f "Google Chrome" || true');

// Start fresh browser session
await page.goto(url);
```

**Prevention**: Break work into smaller batches.

### 4.2 Page Navigation Issues
**Problem**: Page 67+ showed "共 0 筆" initially.

**Solution**:
```javascript
// If no jobs shown, click search button to refresh
await page.click('button:has-text("搜尋")');
await page.waitForTimeout(3000);
```

**Lesson**: Search results can be stale; refresh when needed.

### 4.3 Duplicate Applications
**Problem**: Same jobs appear on multiple pages.

**Solution**:
- Check for "已應徵" text before applying
- Track skipped jobs
- Accept that some overlap will occur

**Result**: Successfully skipped 1,538 duplicates across all runs.

### 4.4 Inconsistent Page Counts
**Problem**: Page 50 works but page 67 shows 0 results.

**Explanation**:
- 104.com.tw has ~1000 jobs
- At 20 jobs/page = ~50 pages max
- Pages beyond that are empty
- But search can return to page 1 with fresh results

**Strategy**: Loop back to page 1 when pages run out.

---

## 5. Success Metrics

### 5.1 Achieved Results
- **Total Applications**: 2,495
- **Overall Success Rate**: 92.5%
- **Best Run Success Rate**: 96.2%
- **Average Speed**: 609 applications/hour
- **Pages Processed**: 115 unique pages

### 5.2 Performance Indicators
**Excellent Performance**:
- 95%+ success rate
- <5% failure rate
- Low skip rate (indicates fresh jobs)

**Good Performance**:
- 90%+ success rate
- <10% failure rate
- Moderate skip rate

**Poor Performance**:
- <85% success rate
- High failure rate (indicates technical issues)
- Very high skip rate (indicates no new jobs)

### 5.3 Efficiency Metrics
- **Jobs per page**: 16-18 average (excellent)
- **Time per job**: ~6 seconds
- **Failures per 100**: <8 (excellent)

---

## 6. Best Practices Discovered

### 6.1 Always Read Before Edit
```javascript
// ✅ CORRECT - Read job info before attempting
const jobInfo = await page.evaluate((index) => {
  const container = document.querySelectorAll('[class*="job-list-container"]')[index];
  const title = container.querySelector('a[href*="/job/"]')?.textContent;
  const alreadyApplied = container.textContent.includes('已應徵');
  return { title, alreadyApplied };
}, jobIndex);

// Then decide whether to apply
if (jobInfo.alreadyApplied) {
  skip();
} else {
  apply();
}
```

### 6.2 Defensive Programming
```javascript
// Check if element exists before interacting
const jobInfo = await page.evaluate((index) => {
  const containers = document.querySelectorAll('[class*="job-list-container"]');
  if (index >= containers.length) return null; // ✅ Guard clause

  const container = containers[index];
  // ... rest of logic
}, jobIndex);

if (!jobInfo) break; // ✅ Handle null case
```

### 6.3 Graceful Degradation
```javascript
try {
  // Try to close any open tabs
  const allPages = page.context().pages();
  if (allPages.length > 1) {
    await allPages[allPages.length - 1].close();
  }
  await page.bringToFront();
} catch (e) {
  // Don't crash if cleanup fails
}
```

### 6.4 Clear State Management
```javascript
const pageResults = {
  page: pageNum,
  successful: 0,
  failed: 0,
  skipped: 0
};

// Update counters consistently
if (success) {
  pageResults.successful++;
  results.totalSuccessful++;
}
```

---

## 7. Platform-Specific Insights

### 7.1 104.com.tw Structure
- Uses Vue.js framework
- Jobs load dynamically (need wait times)
- ~20 jobs per page
- Maximum ~50 pages visible
- New jobs posted cause page numbers to shift

### 7.2 Application Flow
1. Job search results → Click "應徵" DIV
2. New tab opens with form
3. Select cover letter from dropdown
4. Click "確認送出" button
5. Redirects to `/job/apply/done/` on success
6. Close tab and return to search

### 7.3 Rate Limiting
**Observation**: No rate limiting detected up to 900 applications in single run.

**Recommendation**: Still use human-like delays for safety.

---

## 8. Key Success Factors

### 8.1 What Made This Work
1. **Correct selectors** - Using exact class names
2. **Proper timing** - Wait for dynamic content
3. **Tab management** - Always cleanup
4. **Error handling** - Continue on failures
5. **Skip detection** - Avoid duplicate work
6. **Progress tracking** - Clear visibility
7. **Target-based** - Stop at exact goal

### 8.2 What Didn't Work
1. ❌ Generic button selectors
2. ❌ Assuming modal instead of new tab
3. ❌ Page text for success detection
4. ❌ Not cleaning up tabs
5. ❌ Fixed delays (use random)
6. ❌ Not handling already-applied jobs

---

## 9. Recommendations for Future

### 9.1 Enhancements
- Add screenshot on failure for debugging
- Log failed job URLs for manual review
- Track application timestamps
- Save results to database
- Email summary reports
- Add retry mechanism (max 2 retries)

### 9.2 Monitoring
- Track success rate per page
- Alert if success rate drops below 80%
- Monitor failure patterns
- Track time per application

### 9.3 Maintenance
- Check selectors monthly (UI changes)
- Update cover letter name if changed
- Verify login credentials
- Test on sample jobs before bulk run

---

## 10. Conclusion

### What Worked Best
- **Architecture**: Target-based batch processing
- **Error Handling**: Try-catch with cleanup
- **Timing**: Random human-like delays
- **Detection**: Simple text-based skip logic
- **Verification**: URL pattern matching

### Most Important Lessons
1. **Read the DOM carefully** - Don't assume structure
2. **Always cleanup** - Close tabs, handle errors
3. **Be patient** - Wait for dynamic content
4. **Track everything** - Detailed logging helps debugging
5. **Test incrementally** - Start small, scale up

### Final Metrics
- **2,495 applications** in **~4.1 hours**
- **92.5% success rate**
- **5 successful runs** with **0 manual intervention**
- **All targets met exactly**

This automation proves that with careful observation, proper error handling, and patience, complex web automation tasks can be highly reliable and efficient.
