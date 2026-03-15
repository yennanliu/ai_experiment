# 104.com.tw Auto Apply - Gemini CLI Instructions

## Quick Start

### 1. Setup Credentials
```bash
cd /Users/jerryliu/ai_experiment/104

# Copy example env file
cp .env.example .env

# Edit .env with your credentials
# JOB_EMAIL=your_email@example.com
# JOB_PASSWORD=your_password
# COVER_LETTER=自訂推薦信1
```

### 2. Install Dependencies
```bash
npm install
npx playwright install chromium
```

### 3. Run the Script
```bash
node 104_auto_apply_gemini.js --start-page 2 --max-pages 5
```

## Environment Variables (.env file)

| Variable | Description | Required |
|----------|-------------|----------|
| `JOB_EMAIL` | Your 104.com.tw login email | Yes |
| `JOB_PASSWORD` | Your 104.com.tw password | Yes |
| `COVER_LETTER` | Cover letter name to use | No (default: 自訂推薦信1) |

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--start-page <num>` | Starting page number | 2 |
| `--max-pages <num>` | Number of pages to process | 5 |
| `--headless` | Run without visible browser | false |
| `--skip-login` | Skip automatic login attempt | false |
| `--help` | Show help message | - |

## Examples

```bash
# Process pages 2-6 (5 pages) with auto-login
node 104_auto_apply_gemini.js --start-page 2 --max-pages 5

# Process pages 5-14 (10 pages)
node 104_auto_apply_gemini.js --start-page 5 --max-pages 10

# Skip automatic login (login manually)
node 104_auto_apply_gemini.js --skip-login

# Run in headless mode (no browser window)
node 104_auto_apply_gemini.js --headless
```

## Keyboard Controls

While the automation is running, you can control it with these keys:

| Key | Action |
|-----|--------|
| `P` | Pause automation |
| `R` | Resume automation |
| `Q` | Quit automation |

## For Gemini CLI

When using Gemini CLI, you can run this automation with:

```
Run the 104 job application automation script:
cd /Users/jerryliu/ai_experiment/104 && node 104_auto_apply_gemini.js --start-page 2 --max-pages 5
```

Or ask Gemini to:
1. Navigate to the script directory
2. Run the script with your desired parameters

## Login Flow

The script handles login automatically:

1. **Auto-Login**: Uses credentials from `.env` file
2. **2FA Support**: Waits 60 seconds for manual 2FA if required
3. **Fallback**: If auto-login fails, waits 30 seconds for manual login

## How It Works

1. **Login** (if not logged in)
   - Uses credentials from `.env` file
   - Handles email/password entry
   - Waits for 2FA if needed
2. **Opens browser** to 104.com.tw job search page
3. **For each job** on the page:
   - Clicks the "應徵" (Apply) button
   - Opens new tab with application form
   - Selects cover letter from `.env` (default: 自訂推薦信1)
   - Submits the application
   - Verifies success
   - Closes tab and returns to search page
4. **Moves to next page** and repeats
5. **Shows final summary** of results

## Output

The script shows real-time progress:
```
🔐 Starting login process...
📧 Entering email...
➡️ Clicking continue...
🔑 Entering password...
🚪 Clicking login...
✅ Login successful!

╔════════════════════════════════════════╗
║         PAGE 2                         ║
╚════════════════════════════════════════╝

Found 20 jobs

━━━━ Job 1/20 ━━━━
📋 Software Engineer
🏢 ABC Company
✅ SUCCESS

━━━━ Job 2/20 ━━━━
📋 Backend Developer
🏢 XYZ Corp
⏭️ SKIPPED (already applied)
```

## Troubleshooting

### "Credentials not found"
Make sure you have a `.env` file with:
```
JOB_EMAIL=your_email@example.com
JOB_PASSWORD=your_password
```

### "Browser not installed"
```bash
npx playwright install chromium
```

### "2FA required"
The script will wait 60 seconds - enter the verification code manually in the browser.

### "Login failed"
The script will wait 30 seconds for manual login in the browser window.

### "Timeout errors"
- Check your internet connection
- Try running with fewer pages: `--max-pages 2`

## Security Notes

- **Never commit** your `.env` file to version control
- The `.env` file is in `.gitignore` by default
- Credentials are only used locally

## File Locations

```
/Users/jerryliu/ai_experiment/104/
├── 104_auto_apply_gemini.js  # Main script
├── .env                       # Your credentials (create from .env.example)
├── .env.example               # Template for credentials
├── package.json               # Dependencies
└── GEMINI_CLI_INSTRUCTIONS.md # This file
```
