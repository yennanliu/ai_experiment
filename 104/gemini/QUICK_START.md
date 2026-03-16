# Quick Start - 104 Job Automation via Gemini CLI

## Setup (One Time)

1. **Chrome:** Go to `chrome://inspect/#remote-debugging` → Toggle ON
2. **Login:** Open 104.com.tw in Chrome and login
3. **Node:** `nvm install 22 && nvm use 22` (optional but recommended)

---

## Copy-Paste Prompts for Gemini CLI

### 🔍 List Your Chrome Tabs
```
Run: cd /Users/jerryliu/ai_experiment/104/gemini && ./cdp list
```

### 📸 View Page Content
```
Run: cd /Users/jerryliu/ai_experiment/104/gemini && ./cdp snap <TAB_ID>
```

### 🖱️ Click Apply Button
```
Run: cd /Users/jerryliu/ai_experiment/104/gemini && ./cdp eval <TAB_ID> "document.querySelector('.apply-button__button').click()"
```

### ✉️ Select Cover Letter & Submit
```
Run: cd /Users/jerryliu/ai_experiment/104/gemini && ./cdp eval <TAB_ID> "
const dd = document.querySelector('.multiselect'); if(dd) dd.click();
setTimeout(() => {
  [...document.querySelectorAll('.multiselect__option')].find(o => o.textContent.includes('自訂推薦信1'))?.click();
  setTimeout(() => [...document.querySelectorAll('button')].find(b => b.textContent.includes('確認送出'))?.click(), 500);
}, 500);
"
```

---

## Full Automation Prompt

Copy this entire block to Gemini CLI:

```
Help me apply to jobs on 104.com.tw using my existing Chrome browser.

Working directory: /Users/jerryliu/ai_experiment/104/gemini

Steps:
1. Run ./cdp list to find my 104.com.tw tab
2. Run ./cdp snap <id> to see the job listings
3. For each job, click the apply button with:
   ./cdp eval <id> "document.querySelector('.apply-button__button').click()"
4. After new tab opens, run ./cdp list to get the new tab ID
5. In the new tab, select cover letter and submit:
   ./cdp eval <new_id> "..." (use the select & submit snippet)
6. Repeat for remaining jobs

Key selectors:
- Apply: .apply-button__button
- Cover letter: .multiselect → .multiselect__option containing "自訂推薦信1"
- Submit: button containing "確認送出"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No DevToolsActivePort" | Enable debugging at `chrome://inspect/#remote-debugging` |
| Tab ID not found | Run `./cdp list` again for updated IDs |
| Command hangs | Run `./cdp stop` then retry |
