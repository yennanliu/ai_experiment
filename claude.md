# 104.com.tw Job Application Automation Guide

## Overview
Automated job application system for 104.com.tw using Playwright MCP tools.

## Login Process

### Step 1: Navigate to Job Search Page
```javascript
// Navigate to the search results page
await page.goto('https://www.104.com.tw/jobs/search/?page=6&keyword=++++%E8%BB%9F%E9%AB%94%E5%B7%A5%E7%A8%8B%E5%B8%AB&jobsource=joblist_search&order=15&remoteWork=1,2&area=6001001000,6001002000');
```

### Step 2: Login (if needed)
1. Click on "登入/註冊" button
2. Enter email in "Enter ID or Email" field
3. Click "Continue" button
4. Enter password in "Enter Password" field
5. Click "Login" button
6. Handle 2FA if prompted:
   - Enter 6-digit verification code sent to email
   - Or type the code manually via keyboard

**Credentials Used:**
- Email: ***REMOVED***
- Password: ***REMOVED***

---

## Job Collection Process

### Collect Job Links from Page
```javascript
// Wait for job listings to load
await page.waitForSelector('.job-list-item, article, [class*="job"]', { timeout: 5000 });

// Collect all job links
const jobLinks = await page.evaluate(() => {
  const links = [];
  const jobElements = document.querySelectorAll('a[href*="/job/"]');
  const seenUrls = new Set();

  jobElements.forEach((linkElement) => {
    const jobUrl = linkElement.href;

    // Skip duplicates
    if (seenUrls.has(jobUrl)) return;
    seenUrls.add(jobUrl);

    const container = linkElement.closest('article') || linkElement;
    const jobTitle = linkElement.textContent.trim();
    const containerText = container.textContent || '';

    // Filter out already applied or closed jobs
    const alreadyApplied = containerText.includes('今日已應徵') || containerText.includes('已應徵');
    const cantApply = containerText.includes('無法應徵') || containerText.includes('關閉職缺');

    if (jobUrl.includes('/job/') && !alreadyApplied && !cantApply) {
      links.push({
        url: jobUrl,
        title: jobTitle.substring(0, 100),
        status: 'pending'
      });
    }
  });

  return links;
});
```

---

## Application Process

### Step 1: Open Job Detail Page
```javascript
await page.goto(jobUrl, { waitUntil: 'networkidle', timeout: 30000 });
await page.waitForTimeout(2000);
```

### Step 2: Click Apply Button
The apply button is found using accessibility snapshot. Look for element with text "應徵":

```javascript
// Take snapshot first to get element reference
await mcp__playwright__browser_snapshot();

// Click using the ref from snapshot (e.g., ref=e128)
await mcp__playwright__browser_click({ ref: 'e128', element: '應徵 button' });
```

**Important:** The URL changes to include `?apply=form` parameter after clicking.

### Step 3: Select Cover Letter (自訂推薦信1)
```javascript
// Click to open dropdown and select cover letter
await page.evaluate(() => {
  // Open dropdown
  const dropdown = Array.from(document.querySelectorAll('div')).find(el =>
    el.textContent.includes('系統預設') && el.textContent.includes('自訂推薦信')
  );

  if (dropdown) {
    const clickableElement = dropdown.querySelector('.multiselect__select') ||
                             dropdown.querySelector('[class*="select"]') ||
                             dropdown;
    clickableElement.click();
  }
});

await page.waitForTimeout(1500);

// Select the cover letter option
await page.evaluate(() => {
  const option = Array.from(document.querySelectorAll('span, div, li')).find(el =>
    el.textContent.trim() === '自訂推薦信1'
  );
  if (option) {
    option.click();
  }
});
```

**Note:** "自動推薦信1" in the requirements actually refers to "自訂推薦信1" (Custom Cover Letter 1) in the UI.

### Step 4: Submit Application
```javascript
await page.evaluate(() => {
  const submitButton = Array.from(document.querySelectorAll('button')).find(el =>
    el.textContent.includes('確認送出')
  );
  if (submitButton) {
    submitButton.click();
  }
});

await page.waitForTimeout(3000);
```

**Success Indicator:** Page redirects to `/job/apply/done/?jobNo=XXXXX&jobsource=joblist_search`

---

## Complete Automation Workflow

### For Each Page:
1. **Collect job links** from current page (filtering out applied/closed jobs)
2. **For each job:**
   - Navigate to job detail page
   - Take snapshot to get apply button reference
   - Click apply button using the ref
   - Select cover letter "自訂推薦信1"
   - Click submit button "確認送出"
   - Log result (SUCCESS/SKIPPED/FAILED)
   - Add 2-4 second random delay for safety
3. **Navigate to next page** if available
4. **Repeat** until no more pages

### Safety Features:
- Random delay between applications (2-4 seconds)
- Error handling with try/catch
- Skip already applied jobs
- Check for apply button existence before proceeding
- Wait for page loads and form submissions

### Logging Format:
```
[Page 6]
Job Title – Company
Status: SUCCESS / SKIPPED / FAILED
Reason: (if skipped/failed)
```

---

## Key Learnings

1. **Login 2FA:** Must handle email verification for new devices
2. **Apply Button:** Found via accessibility snapshot, not direct selector
3. **Cover Letter Name:** "自動推薦信1" = "自訂推薦信1" in UI
4. **Dropdown Selection:** Requires JavaScript evaluate() for reliable clicking
5. **Success Check:** URL changes to `/job/apply/done/` after successful submission
6. **Page Structure:** Job listings use `a[href*="/job/"]` selector consistently

---

## Next Steps for Full Automation

1. Create a loop to process all 18 jobs found on page 6
2. Implement pagination to move to page 7, 8, etc.
3. Add persistent logging to JSON file to track applied jobs
4. Add daily application limit safeguard
5. Implement retry mechanism (max 2 retries per job)
6. Add screenshot on failure for debugging

---

## Sample Full Automation Script Structure

```javascript
// Main automation function
async function autoApplyJobs(startPage = 6) {
  let currentPage = startPage;
  let hasNextPage = true;

  while (hasNextPage) {
    console.log(`[Page ${currentPage}]`);

    // Collect jobs from current page
    const jobs = await collectJobLinks(page);
    console.log(`Found ${jobs.length} jobs to apply`);

    // Apply to each job
    for (const job of jobs) {
      try {
        const result = await applyToJob(page, job);
        console.log(`${job.title} - ${result.status}`);

        // Random delay
        const delay = 2000 + Math.random() * 2000;
        await page.waitForTimeout(delay);
      } catch (error) {
        console.log(`${job.title} - FAILED: ${error.message}`);
      }
    }

    // Check for next page and navigate
    hasNextPage = await goToNextPage(page);
    if (hasNextPage) {
      currentPage++;
    }
  }

  console.log('Automation complete!');
}
```

---

## Status: First Successful Application Completed ✓

Successfully applied to: **前端工程師（時薪制薪水無上限，時間彈性運用）** at 快組隊股份有限公司
