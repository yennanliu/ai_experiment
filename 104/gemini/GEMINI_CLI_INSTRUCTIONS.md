# 104.com.tw Auto Apply - Gemini CLI Instructions

## Quick Start (Recommended: Use Existing Browser)

This method reuses your existing Chrome session - no need to login again!

### Step 1: Start Chrome with Remote Debugging
```bash
cd /Users/jerryliu/ai_experiment/104/gemini
./start-chrome-debug.sh
```

Or manually:
```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
```

### Step 2: Login to 104.com.tw
Log into 104.com.tw in the Chrome window that opened.

### Step 3: Run Automation
```bash
node 104_auto_apply_gemini.js --use-existing --start-page 2 --max-pages 5
```

---

## Alternative: New Browser with Auto-Login

### 1. Setup Credentials
```bash
cd /Users/jerryliu/ai_experiment/104/gemini
cp .env.example .env
# Edit .env with your credentials
```

### 2. Install Dependencies
```bash
npm install
npx playwright install chromium
```

### 3. Run
```bash
node 104_auto_apply_gemini.js --start-page 2 --max-pages 5
```

---

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--use-existing` | **Connect to existing Chrome browser** | false |
| `--cdp-port <num>` | Chrome DevTools Protocol port | 9222 |
| `--start-page <num>` | Starting page number | 2 |
| `--max-pages <num>` | Number of pages to process | 5 |
| `--headless` | Run without visible browser | false |
| `--skip-login` | Skip automatic login attempt | false |
| `--help` | Show help message | - |

## Examples

```bash
# USE EXISTING CHROME (recommended - reuses your logged-in session)
node 104_auto_apply_gemini.js --use-existing --start-page 2 --max-pages 5

# Use existing Chrome on different port
node 104_auto_apply_gemini.js --use-existing --cdp-port 9223

# NEW BROWSER with auto-login
node 104_auto_apply_gemini.js --start-page 2 --max-pages 5

# Process 10 pages
node 104_auto_apply_gemini.js --use-existing --max-pages 10
```

## For Gemini CLI

Tell Gemini to run these commands:

**Using existing Chrome (recommended):**
```
1. Run: cd /Users/jerryliu/ai_experiment/104/gemini && ./start-chrome-debug.sh
2. Wait for me to login to 104.com.tw
3. Then run: node 104_auto_apply_gemini.js --use-existing --start-page 2 --max-pages 5
```

**Or one-liner (if Chrome is already running with debugging):**
```
Run: cd /Users/jerryliu/ai_experiment/104/gemini && node 104_auto_apply_gemini.js --use-existing
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `P` | Pause automation |
| `R` | Resume automation |
| `Q` | Quit automation |

## Environment Variables (.env file)

Only needed if NOT using `--use-existing`:

| Variable | Description | Required |
|----------|-------------|----------|
| `JOB_EMAIL` | Your 104.com.tw login email | Yes |
| `JOB_PASSWORD` | Your 104.com.tw password | Yes |
| `COVER_LETTER` | Cover letter name to use | No (default: 自訂推薦信1) |

## How --use-existing Works

1. **You start Chrome** with remote debugging enabled
2. **You log in** to 104.com.tw in that browser
3. **Script connects** to your existing browser via Chrome DevTools Protocol
4. **Script automates** job applications using your existing session
5. **Browser stays open** after script finishes - your session is preserved

Benefits:
- No need to store passwords in .env file
- No need to deal with 2FA every time
- Reuses your existing cookies and session
- Browser stays open after automation

## Troubleshooting

### "Failed to connect to existing Chrome"
Make sure Chrome is running with remote debugging:
```bash
# Close all Chrome windows first, then:
./start-chrome-debug.sh
# or
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
```

### "Not logged in"
Log into 104.com.tw in the Chrome window before running the script.

### "Browser not installed" (for new browser mode)
```bash
npx playwright install chromium
```

## File Locations

```
/Users/jerryliu/ai_experiment/104/gemini/
├── 104_auto_apply_gemini.js   # Main script
├── start-chrome-debug.sh      # Helper to start Chrome with debugging
├── .env                       # Your credentials (optional with --use-existing)
├── .env.example               # Template for credentials
├── package.json               # Dependencies
└── GEMINI_CLI_INSTRUCTIONS.md # This file
```
