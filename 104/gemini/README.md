# 104 Job Auto-Apply via Chrome CDP

Connect to your **existing Chrome browser** to automate job applications on 104.com.tw. Reuses your logged-in session - no need to re-login or handle 2FA.

Uses [pasky/chrome-cdp-skill](https://github.com/pasky/chrome-cdp-skill).

---

## Setup

### 1. Enable Chrome Remote Debugging
Open Chrome → Go to `chrome://inspect/#remote-debugging` → Toggle **ON**

### 2. Login to 104.com.tw
Just login in your normal Chrome browser.

### 3. (Optional) Install Node.js 22+
```bash
nvm install 22 && nvm use 22
```

---

## Quick Start

```bash
cd /Users/jerryliu/ai_experiment/104/gemini

# List Chrome tabs
./cdp list

# View page structure
./cdp snap <TAB_ID>

# Click element
./cdp click <TAB_ID> ".apply-button__button"

# Run JavaScript
./cdp eval <TAB_ID> "document.title"
```

---

## Commands

| Command | Description |
|---------|-------------|
| `./cdp list` | List all Chrome tabs |
| `./cdp snap <id>` | Accessibility tree snapshot |
| `./cdp shot <id>` | Take screenshot |
| `./cdp html <id> [selector]` | Get HTML |
| `./cdp nav <id> <url>` | Navigate to URL |
| `./cdp click <id> <selector>` | Click element |
| `./cdp clickxy <id> <x> <y>` | Click at coordinates |
| `./cdp type <id> <text>` | Type text |
| `./cdp eval <id> <expr>` | Run JavaScript |
| `./cdp open [url]` | Open new tab |
| `./cdp stop` | Stop daemons |

---

## Gemini CLI Usage

### List Tabs
```
Run: cd /Users/jerryliu/ai_experiment/104/gemini && ./cdp list
```

### View Page
```
Run: ./cdp snap <TAB_ID>
```

### Click Apply Button
```
Run: ./cdp eval <TAB_ID> "document.querySelector('.apply-button__button').click()"
```

### Select Cover Letter & Submit
```
Run: ./cdp eval <TAB_ID> "
const dd = document.querySelector('.multiselect'); if(dd) dd.click();
setTimeout(() => {
  [...document.querySelectorAll('.multiselect__option')].find(o => o.textContent.includes('自訂推薦信1'))?.click();
  setTimeout(() => [...document.querySelectorAll('button')].find(b => b.textContent.includes('確認送出'))?.click(), 500);
}, 500);
"
```

### Full Automation Prompt
```
Help me apply to jobs on 104.com.tw using chrome-cdp.

Directory: /Users/jerryliu/ai_experiment/104/gemini

1. Run ./cdp list - find 104.com.tw tab
2. Run ./cdp snap <id> - see job listings
3. Click apply: ./cdp eval <id> "document.querySelector('.apply-button__button').click()"
4. Run ./cdp list - find new application tab
5. Select cover letter & submit in new tab
6. Repeat

Selectors:
- Apply button: .apply-button__button
- Cover letter: .multiselect → "自訂推薦信1"
- Submit: button with "確認送出"
```

---

## 104.com.tw Selectors

| Element | Selector |
|---------|----------|
| Apply button | `.apply-button__button` |
| Cover letter dropdown | `.multiselect` |
| Cover letter option | `.multiselect__option` containing "自訂推薦信1" |
| Submit button | `button` containing "確認送出" |
| Already applied | Text "已應徵" or "今日已應徵" |
| Success page | URL contains `/job/apply/done/` |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No DevToolsActivePort" | Enable at `chrome://inspect/#remote-debugging` |
| Tab ID not found | Run `./cdp list` for updated IDs |
| Command hangs | Run `./cdp stop` then retry |
| Node version warning | `nvm install 22 && nvm use 22` |

---

## Files

```
gemini/
├── cdp              # CLI wrapper
├── scripts/
│   └── cdp.mjs      # Chrome CDP skill
├── README.md        # This file
├── package.json
└── .env.example     # Credentials template (optional)
```

---

## Security

- Credentials in `.env` are optional (only for legacy script)
- Chrome CDP uses your existing browser session
- Never commit `.env` to git
