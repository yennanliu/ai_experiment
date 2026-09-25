# 104.com.tw Job Application Automation Guide

## Overview
Automated job application system for 104.com.tw using Playwright MCP tools.

## Login Process

### Step 1: Navigate to Job Search Page
```javascript
// Navigate to the search results page (the `page` param selects the results page)
await page.goto('https://www.104.com.tw/jobs/search/?page=1&keyword=++++%E8%BB%9F%E9%AB%94%E5%B7%A5%E7%A8%8B%E5%B8%AB&jobsource=joblist_search&order=15&remoteWork=1,2&area=6001001000,6001002000');
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

**Credentials:**
⚠️ **SECURITY WARNING:** Store credentials securely using environment variables or a secure credential manager. NEVER commit credentials to version control!

---

## Critical Technical Details

### Apply Button Structure
The "應徵" (Apply) buttons on the job search results page have unique characteristics:

- **Tag Type:** DIV element (not `<button>` or `<a>` tag)
- **CSS Selector:** `.apply-button__button`
- **Visual:** Shows envelope icon + "應徵" text
- **Count per page:** Typically 20-22 buttons (one per job listing)
- **HTML Structure:**
  ```html
  <div class="btn btn-sm apply-button__button button--gray btn-has-icon btn-outline-light btn-outline-light--apply">
    <i class="mr-3 apply-button__icon jb_icon_apply_line"></i>
    <span>應徵</span>
  </div>
  ```

### Form Popup Behavior
- **Trigger:** Clicking "應徵" button opens **NEW BROWSER TAB** (not a modal popup)
- **URL Pattern:** `https://www.104.com.tw/job/{jobId}?apply=form&jobsource=...`
- **Form Location:** Application form appears in the new tab immediately
- **Tab Management:** Must switch to new tab to interact with form

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

### Step 1: Click Apply Button on Job Listing
**IMPORTANT:** The "應徵" buttons are DIV elements (not button tags) on the job search results page.

```javascript
// Find and click "應徵" button using direct selector
await page.evaluate(() => {
  const applyButtons = document.querySelectorAll('.apply-button__button');

  if (applyButtons.length > 0) {
    // Click the desired button (e.g., first one)
    applyButtons[0].click();
  }
});
```

**Important:**
- **Selector:** `.apply-button__button`
- Clicking opens a **NEW TAB** with the job detail page
- New tab URL includes `?apply=form` parameter
- Application popup appears immediately in the new tab

### Step 2: Switch to New Tab
```javascript
// The apply form opens in a new tab; Steps 3-4 run on that tab, not on the search page
const pages = page.context().pages();
const applyTab = pages[pages.length - 1];
await applyTab.bringToFront();
```

### Step 3: Select Cover Letter (自訂推薦信1)
```javascript
// Click to open dropdown - find parent of "系統預設" span
await page.evaluate(() => {
  const elements = Array.from(document.querySelectorAll('*'));
  const systemDefault = elements.find(el => el.textContent === '系統預設' && el.tagName === 'SPAN');

  if (systemDefault && systemDefault.parentElement) {
    systemDefault.parentElement.click();
  }
});

await page.waitForTimeout(500);

// Select the cover letter option by clicking the multiselect option
await page.evaluate(() => {
  const options = document.querySelectorAll('.multiselect__option');

  options.forEach(option => {
    if (option.textContent.trim() === '自訂推薦信1') {
      option.click();
    }
  });
});
```

**Note:** The correct cover letter name is "自訂推薦信1" (Custom Cover Letter 1).

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
1. **Stay on job search results page** (don't navigate away)
2. **For each job:**
   - Click "應徵" DIV button (`.apply-button__button`) - opens new tab
   - Switch to the new tab
   - Wait for popup form to load
   - Select cover letter "自訂推薦信1" from dropdown
   - Click submit button "確認送出"
   - Verify success (URL contains `/job/apply/done/`)
   - Close the tab and return to search results
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
2. **Apply Button is a DIV:** Use selector `.apply-button__button` (not a button tag!)
3. **New Tab Behavior:** Clicking "應徵" opens a NEW TAB (not redirect on same page)
4. **Cover Letter Name:** Use "自訂推薦信1" (Custom Cover Letter 1)
5. **Dropdown Selection:**
   - Click parent of "系統預設" span to open dropdown
   - Select option using `.multiselect__option` selector
6. **Success Check:** URL changes to `/job/apply/done/` after successful submission
7. **Page Structure:** Job listings use `a[href*="/job/"]` selector consistently
8. **Tab Management:** Need to handle tab switching when automation opens new tabs

---

## Not Yet Implemented

1. Add persistent logging to JSON file to track applied jobs
2. Add daily application limit safeguard
3. Implement retry mechanism (max 2 retries per job)
4. Add screenshot on failure for debugging

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

