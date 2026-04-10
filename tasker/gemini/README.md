# Tasker.com.tw Automation Script

Automated proposal submission for Tasker (出任務) using Puppeteer.

## Setup

```bash
npm install
```

## Usage

### Method 1: With Auto-Login (via Environment Variables)

```bash
TASKER_ACCOUNT=0963335868 TASKER_PASSWORD=your_password node tasker_apply.js
```

### Method 2: Manual Login

```bash
node tasker_apply.js
```

The script will pause at the login page and wait for you to log in manually. Once logged in, the automation continues.

### Test Run

A helper script is included (not committed to git):

```bash
bash test_run.sh  # Uses environment variables from the script
```

## Features

- ✅ Automatic login via environment variables (credentials not stored in code)
- ✅ Detects and handles manual login fallback
- ✅ Collects all available cases from search results
- ✅ Processes proposal forms
- ✅ Random delays to avoid detection
- ✅ Detailed logging to `automation_log.txt`
- ✅ Form field detection and inspection

## Current Status

**Phase 1: Exploration & Setup** ✅
- Login flow implementation
- Case list detection and collection
- Form field inspection

**Phase 2: Form Automation** (Next)
- Budget input field identification
- Automatic price injection
- Submit button automation

## Output

Results are logged to:
- Console: Real-time status with emoji indicators
- `automation_log.txt`: Detailed log file with timestamps

## Security

⚠️ **IMPORTANT**: 
- Credentials are passed via environment variables, **NOT hardcoded** in source
- `test_run.sh` is in `.gitignore` to prevent accidental credential commits
- Never commit `.env` files or credentials to version control
- All sensitive files are protected by the project's `.gitignore`

## Debugging

If something goes wrong:
1. Check `automation_log.txt` for detailed error messages
2. The browser stays open for manual inspection
3. Check console output for emoji-prefixed status messages

## Browser Control

The script runs with `headless: false` so you can watch the automation in action and manually intervene if needed.
