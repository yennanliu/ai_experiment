# 104 Automation - Working Script Documentation

**Date:** 2026-02-26
**Status:** ✅ Verified Working

---

## The Script That Works

### **`104_auto_apply_with_controls.js`** ✅

This is the **confirmed working script** for full automation flow on 104.com.tw.

---

## Why This Script?

### Key Features That Make It Work:

1. **Keyboard Controls**
   - Press `P` to pause automation
   - Press `R` to resume automation
   - Press `Q` to quit automation
   - Prevents interference with manual control

2. **Visual Status Indicator**
   - Shows real-time automation status on the page
   - Clear feedback (RUNNING / PAUSED / COMPLETED)

3. **Correct Technical Approach**
   - Stays on job search results page (doesn't navigate away)
   - Clicks `.apply-button__button` DIV elements by index
   - Opens new tabs for each application (not redirects)
   - Proper tab management (switch → process → close → return)

4. **Robust Error Handling**
   - Try/catch for each job
   - Cleanup on failures
   - Re-setup controls after navigation

---

## The Working Flow

```javascript
For each page (from startPage to maxPages):
  1. Navigate to search page
  2. Count jobs on page (via .apply-button__button)

  For each job:
    ✓ Check user control (pause/quit)
    ✓ Get job info (title, company)
    ✓ Skip if already applied
    ✓ Click .apply-button__button[index] → Opens new tab
    ✓ Switch to new tab
    ✓ Select cover letter "自訂推薦信1"
    ✓ Click submit "確認送出"
    ✓ Verify success (/job/apply/done/)
    ✓ Close tab, return to search page
    ✓ Random delay (2-4 seconds)

  3. Move to next page
```

---

## Why Not The Other Scripts?

### `104_auto_apply_complete.js` ❌
- **Problem:** Uses different approach - navigates directly to job URL
- **Issue:** Doesn't match the proven working method
- **Missing:** No keyboard controls for manual intervention

### `apply_single_job.js` ℹ️
- **Purpose:** Helper for manual testing only
- **Usage:** Not for full automation
- **Scope:** Single job at a time via browser console

---

## Technical Details

### Apply Button Structure
```html
<div class="btn btn-sm apply-button__button button--gray btn-has-icon btn-outline-light btn-outline-light--apply">
  <i class="mr-3 apply-button__icon jb_icon_apply_line"></i>
  <span>應徵</span>
</div>
```

### CSS Selector
```javascript
document.querySelectorAll('.apply-button__button')
```

### New Tab Behavior
- Clicking opens **NEW TAB** (not modal, not redirect on same page)
- URL pattern: `https://www.104.com.tw/job/{jobId}?apply=form&jobsource=...`
- Application form loads immediately in new tab

### Cover Letter
- **Name:** `自訂推薦信1` (Custom Cover Letter 1)
- **Dropdown:** Click parent of "系統預設" span
- **Selection:** Use `.multiselect__option` selector

### Success Verification
- URL changes to `/job/apply/done/` after successful submission

---

## Usage

### Configuration
```javascript
await autoApplyWithControls(page, startPage = 2, maxPages = 5)
```

### Parameters
- `page` - Playwright page object
- `startPage` - Starting page number (default: 2)
- `maxPages` - Maximum pages to process (default: 5)

### Example
```javascript
// Process pages 2-6 (5 pages total)
await autoApplyWithControls(page, 2, 5);
```

---

## Results Tracking

The script tracks:
- ✅ **Successful applications** - Count and list
- ⏭️ **Skipped jobs** - Already applied or no apply button
- ❌ **Failed applications** - Errors with reasons
- 📄 **Pages processed** - Summary per page

### Sample Output
```
╔══════════════════════════════════════════════╗
║     AUTOMATION COMPLETE                      ║
╚══════════════════════════════════════════════╝
✅ Successful: 45
❌ Failed: 3
⏭️  Skipped: 12
📄 Pages: 5
```

---

## Safety Features

1. **Manual Controls** - Can pause/resume/quit anytime
2. **Random Delays** - 2-4 seconds between jobs (human-like)
3. **Skip Already Applied** - Checks for "已應徵" status
4. **Error Recovery** - Try/catch per job, continues on failure
5. **Tab Cleanup** - Always closes tabs on error
6. **Control Re-setup** - Re-initializes keyboard controls after navigation

---

## Testing Status

**Last Verified:** 2026-02-25

### Test Results:
- ✅ Successfully applied to: **Analytical Engineer** at 保誠人壽保險_總公司
- ✅ Confirmed all steps working with correct selectors
- ✅ Keyboard controls functional
- ✅ Tab management working correctly
- ✅ Error handling tested and verified

---

## Known Issues

None currently. The script works as expected for standard 104.com.tw job applications.

### Limitations:
- Cannot handle CAPTCHA (must solve manually)
- Requires user to be logged in before starting
- Cannot handle custom application forms
- Requires browser to stay open during execution

---

## Credentials Used

- **Email:** <ask_me>
- **Password:** <ask_me>
- **Cover Letter:** 自訂推薦信1

---

## File Location

```
/Users/jerryliu/ai_experiment/104/104_auto_apply_with_controls.js
```

---

## Conclusion

**Use `104_auto_apply_with_controls.js` for all 104.com.tw job application automation.**

This script has been tested and verified to work with the complete flow documented in CLAUDE.md.

🚀 Ready for production use!
