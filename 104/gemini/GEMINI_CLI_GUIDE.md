# Running 104 Job Automation via Gemini CLI

## Prerequisites

1. **Enable Remote Debugging in Chrome**
   - Open Chrome and go to: `chrome://inspect/#remote-debugging`
   - Toggle **ON** "Enable remote debugging"

2. **Login to 104.com.tw** in your Chrome browser

3. **Node.js 22+** (recommended)
   ```bash
   nvm install 22 && nvm use 22
   ```

---

## Quick Start Prompts for Gemini CLI

### Check Setup
```
Run this command to list my Chrome tabs:
cd /Users/jerryliu/ai_experiment/104/gemini && ./cdp list
```

### View a Page
```
Run: cd /Users/jerryliu/ai_experiment/104/gemini && ./cdp snap <TAB_ID>

Replace <TAB_ID> with the ID from the list command (e.g., 0A3B)
```

---

## Full Automation Prompts

### Option 1: Step-by-Step (Recommended for First Time)

Copy and paste this to Gemini CLI:

```
I want to apply to jobs on 104.com.tw using my Chrome browser.

Step 1: List my Chrome tabs
Run: cd /Users/jerryliu/ai_experiment/104/gemini && ./cdp list

Step 2: Find the 104.com.tw tab and tell me its ID

Step 3: Take a snapshot of that tab to see the job listings
Run: ./cdp snap <TAB_ID>

Then help me click the apply buttons one by one.
```

### Option 2: Automated Job Application

```
Help me apply to jobs on 104.com.tw using chrome-cdp commands.

Commands are in: /Users/jerryliu/ai_experiment/104/gemini/

1. First run: ./cdp list
2. Find the 104 job search tab
3. For each job with an apply button (.apply-button__button):
   - Click the apply button using: ./cdp eval <id> "document.querySelector('.apply-button__button').click()"
   - Wait 2 seconds, then run ./cdp list to find the new application tab
   - In the new tab, select cover letter "自訂推薦信1" and submit
   - Close the tab and continue to next job

Important selectors:
- Apply button: .apply-button__button
- Cover letter dropdown: .multiselect
- Submit button: button containing "確認送出"
```

---

## Common Commands Reference

Tell Gemini to run these commands:

| Task | Command |
|------|---------|
| List tabs | `./cdp list` |
| View page structure | `./cdp snap <id>` |
| Take screenshot | `./cdp shot <id>` |
| Navigate to URL | `./cdp nav <id> "https://..."` |
| Click element | `./cdp click <id> ".selector"` |
| Run JavaScript | `./cdp eval <id> "code"` |
| Type text | `./cdp type <id> "text"` |
| Open new tab | `./cdp open "https://..."` |

**Note:** Always run from `/Users/jerryliu/ai_experiment/104/gemini/`

---

## Example Conversation with Gemini CLI

**You:**
```
Run: cd /Users/jerryliu/ai_experiment/104/gemini && ./cdp list
```

**Gemini:** (shows list of tabs)
```
0A3B  https://www.104.com.tw/jobs/search/...  104人力銀行
1C2D  https://mail.google.com/...             Gmail
```

**You:**
```
Take a snapshot of the 104 tab:
Run: ./cdp snap 0A3B
```

**Gemini:** (shows accessibility tree with page elements)

**You:**
```
Click the first apply button:
Run: ./cdp eval 0A3B "document.querySelector('.apply-button__button').click()"
```

**Gemini:** (clicks the button, new tab opens)

**You:**
```
List tabs again to find the application form:
Run: ./cdp list
```

---

## JavaScript Snippets for Job Application

### Click Apply Button
```
./cdp eval <id> "document.querySelector('.apply-button__button').click()"
```

### Select Cover Letter
```
./cdp eval <id> "
const dropdown = document.querySelector('.multiselect');
if (dropdown) dropdown.click();
setTimeout(() => {
  const opts = document.querySelectorAll('.multiselect__option');
  for (const o of opts) {
    if (o.textContent.includes('自訂推薦信1')) { o.click(); break; }
  }
}, 500);
"
```

### Submit Application
```
./cdp eval <id> "
const btn = [...document.querySelectorAll('button')].find(b => b.textContent.includes('確認送出'));
if (btn) btn.click();
"
```

### Check if Already Applied
```
./cdp eval <id> "document.body.textContent.includes('已應徵') || document.body.textContent.includes('應徵成功')"
```

---

## Troubleshooting Prompts

### If "No DevToolsActivePort found"
```
Chrome remote debugging is not enabled.
Please open Chrome and go to: chrome://inspect/#remote-debugging
Then toggle ON "Enable remote debugging"
```

### If Tab Not Found
```
Run ./cdp list again to get updated tab IDs.
Tab IDs change when tabs are opened/closed.
```

### If Command Fails
```
Run: ./cdp stop
Then try the command again.
```

---

## Pro Tips for Gemini CLI

1. **Always start with `./cdp list`** to get current tab IDs

2. **Use `./cdp snap <id>`** before clicking to understand page structure

3. **Tab IDs are prefixes** - you only need enough characters to be unique (e.g., `0A3` instead of `0A3B5C7D`)

4. **New tabs get new IDs** - after clicking apply, run `./cdp list` again

5. **Commands are stateless** - each command runs independently, so always specify the full path or `cd` first

---

## One-Liner for Gemini CLI

```
cd /Users/jerryliu/ai_experiment/104/gemini && ./cdp list && echo "---" && echo "Use ./cdp snap <id> to view a tab, ./cdp eval <id> 'js' to run code"
```
