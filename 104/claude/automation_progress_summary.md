# 104.com.tw Job Application Automation - Progress Summary

**Date:** 2026-02-28
**Current Status:** 44/100 applications completed

---

## Overall Results

### Total Applications: 44 ✅
- **Session 1** (Pages 2-7, with remote filter): 3 successful
- **Session 2** (Pages 8-33, with remote filter): 18 successful
- **Session 3** (Pages 1-100, no remote filter): 23 successful

### Statistics
- ✅ **Successful:** 44
- ❌ **Failed:** 99 (due to form errors, timeouts, etc.)
- ⏭️ **Skipped:** 2,401 (already applied)
- 📄 **Pages processed:** 152
- 📈 **Overall success rate:** 1.8% (due to 96%+ already applied)

---

## Search Parameters Used

### Completed Searches
1. **"軟體工程師" (Software Engineer)** - Taipei/New Taipei, with remote filter
   - Pages 2-33 processed
   - Result: Most jobs already applied to

2. **"軟體工程師" (Software Engineer)** - Taipei/New Taipei, NO remote filter
   - Pages 1-100 processed
   - Result: 96% already applied, reached end of listings

---

## Next Steps to Reach 100 Applications

### Remaining: 56 applications needed

### Recommended Strategy: Change Keywords
Since "軟體工程師" is exhausted, try these alternative keywords:

1. **"後端工程師" (Backend Engineer)**
   - URL: `https://www.104.com.tw/jobs/search/?area=6001001000,6001002000&keyword=%E5%BE%8C%E7%AB%AF%E5%B7%A5%E7%A8%8B%E5%B8%AB&order=15`

2. **"前端工程師" (Frontend Engineer)**
   - URL: `https://www.104.com.tw/jobs/search/?area=6001001000,6001002000&keyword=%E5%89%8D%E7%AB%AF%E5%B7%A5%E7%A8%8B%E5%B8%AB&order=15`

3. **"全端工程師" (Full-stack Engineer)**
   - URL: `https://www.104.com.tw/jobs/search/?area=6001001000,6001002000&keyword=%E5%85%A8%E7%AB%AF%E5%B7%A5%E7%A8%8B%E5%B8%AB&order=15`

4. **"開發工程師" (Development Engineer)**
   - URL: `https://www.104.com.tw/jobs/search/?area=6001001000,6001002000&keyword=%E9%96%8B%E7%99%BC%E5%B7%A5%E7%A8%8B%E5%B8%AB&order=15`

5. **"程式設計師" (Programmer)**
   - URL: `https://www.104.com.tw/jobs/search/?area=6001001000,6001002000&keyword=%E7%A8%8B%E5%BC%8F%E8%A8%AD%E8%A8%88%E5%B8%AB&order=15`

### Estimated Applications per Keyword
Assuming 10-15% success rate (85-90% already applied):
- Each keyword needs ~15-20 pages processed
- Total estimate: Process 5-10 pages per keyword across 5 keywords

---

## Automation Script

Use the script with different keywords:

```javascript
// Continue automation with new keywords
const keywords = [
  '後端工程師',
  '前端工程師',
  '全端工程師',
  '開發工程師',
  '程式設計師'
];

// For each keyword, process pages 1-20
// Stop when reaching 100 total applications
```

---

## Technical Notes

### Issues Encountered
- **High skip rate:** 96% of jobs already applied to after initial searches
- **Browser session timeout:** Long-running automations may need to be split
- **Page limit:** 104.com.tw shows max ~33 pages per search (660 jobs)

### Recommendations
1. **Run automation in shorter sessions** (10-20 pages at a time)
2. **Track progress** between sessions
3. **Use multiple keywords** to access fresh job listings
4. **Consider expanding location** if needed (add Hsinchu, Taichung, etc.)

---

## Files Generated

- `automation_results_run2.txt` - Initial results
- `automation_results_run3.txt` - Second session
- `automation_results_run4.txt` - Third session
- `automation_results_run5.txt` - Fourth session
- `automation_results_run6.txt` - Fifth session

---

## To Continue

Restart the browser automation and run:
1. Navigate to first new keyword search (後端工程師)
2. Run automation script for pages 1-50
3. Move to next keyword if needed
4. Repeat until reaching 100 total applications

**Current Count: 44/100** ⏱️ **Remaining: 56 applications**
