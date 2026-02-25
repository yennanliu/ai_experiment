# 104.com.tw Job Application Automation - Usage Guide

## Quick Start

### Prerequisites
- Already logged in to 104.com.tw (or have credentials ready)
- Prepared cover letter named "自訂推薦信1" in your 104 account
- Resume uploaded to your account

### Configuration

Edit `104_auto_apply.js` to customize:

```javascript
const CONFIG = {
  searchUrl: 'YOUR_SEARCH_URL',  // Your job search URL
  coverLetter: '自訂推薦信1',      // Your cover letter name
  maxPages: 10,                   // Max pages to process
  // ... other settings
};
```

### Running the Script

Since we're using Playwright MCP tools, you'll need to run it through the browser automation:

1. **Manual Execution (Recommended for Testing)**
   - Use the Playwright MCP browser tools
   - Navigate to the search page
   - Execute functions step by step

2. **Automated Execution**
   - Load the script through browser console
   - Or use a browser automation framework

## Step-by-Step Manual Process

### 1. Setup
```javascript
// Navigate to job search page
await page.goto('YOUR_SEARCH_URL');
```

### 2. Get Apply Buttons Count
```javascript
const count = await page.evaluate(() => {
  return document.querySelectorAll('.apply-button__button').length;
});
console.log(`Found ${count} jobs`);
```

### 3. Apply to Jobs (Loop)
```javascript
for (let i = 0; i < count; i++) {
  // Click apply button
  await page.evaluate((index) => {
    document.querySelectorAll('.apply-button__button')[index].click();
  }, i);

  // Switch to new tab
  const pages = await context.pages();
  const newTab = pages[pages.length - 1];
  await newTab.bringToFront();

  // Select cover letter
  await newTab.evaluate(() => {
    // Click dropdown
    const span = Array.from(document.querySelectorAll('span'))
      .find(el => el.textContent === '系統預設');
    if (span) span.parentElement.click();
  });

  await newTab.waitForTimeout(500);

  await newTab.evaluate(() => {
    // Select option
    const option = Array.from(document.querySelectorAll('.multiselect__option'))
      .find(el => el.textContent.trim() === '自訂推薦信1');
    if (option) option.click();
  });

  // Submit
  await newTab.evaluate(() => {
    const btn = Array.from(document.querySelectorAll('button'))
      .find(el => el.textContent.includes('確認送出'));
    if (btn) btn.click();
  });

  await newTab.waitForTimeout(2000);

  // Close tab
  await newTab.close();
  await page.bringToFront();

  // Delay
  await page.waitForTimeout(3000);
}
```

## Safety Features

### Built-in Protection
- ✅ Skips already applied jobs
- ✅ Random delays between applications (2-4 seconds)
- ✅ Error handling for each job
- ✅ Detailed logging
- ✅ Maximum page limit

### Manual Controls
- Stop execution at any time (Ctrl+C)
- Review results in `application_log.json`
- Adjust delays if rate limiting occurs

## Monitoring

### Real-time Console Output
```
🚀 Starting automation...

========== PAGE 1 ==========
Found 22 jobs on this page

--- Job 1/22 ---
📋 Job: Software Engineer
🏢 Company: ABC Company
✅ SUCCESS: Application submitted

--- Job 2/22 ---
📋 Job: Backend Developer
🏢 Company: XYZ Corp
⏭️  SKIPPED: Already applied
```

### Log File
Check `application_log.json` for detailed results:
```json
{
  "startTime": "2026-02-25T07:00:00Z",
  "totalAttempted": 22,
  "successful": 18,
  "failed": 2,
  "skipped": 2,
  "jobs": [...]
}
```

## Troubleshooting

### Issue: "Apply button not found"
**Solution:** Check if logged in, or if page structure changed

### Issue: "Cover letter not found"
**Solution:** Verify "自訂推薦信1" exists in your account settings

### Issue: Rate limiting / Too many requests
**Solution:** Increase delay between applications (3-5 seconds)

### Issue: Applications failing randomly
**Solution:**
- Check internet connection
- Verify account is not suspended
- Ensure resume and cover letter are valid

## Best Practices

1. **Test First**: Run on 1-2 jobs manually before full automation
2. **Monitor**: Watch the first few applications to ensure it's working
3. **Limit Daily Applications**: Don't apply to 100+ jobs per day
4. **Review Jobs**: Make sure your search criteria match your skills
5. **Backup**: Save logs for tracking which jobs you applied to

## Limitations

- Cannot handle CAPTCHA (must solve manually)
- Cannot handle custom application forms (only standard 104 forms)
- Requires stable internet connection
- Browser must stay open during execution

## Legal & Ethical Notes

⚠️ **Important:**
- Use responsibly and in accordance with 104.com.tw Terms of Service
- Only apply to jobs you're genuinely interested in
- Don't spam applications
- Respect rate limits
- This is for educational purposes

## Support

For issues or questions:
1. Check CLAUDE.md for technical details
2. Review console errors
3. Check application_log.json for failure reasons
