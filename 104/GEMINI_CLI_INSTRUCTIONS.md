# 104.com.tw Auto Apply - Gemini CLI Instructions

## Quick Start

### 1. Install Dependencies
```bash
cd /Users/jerryliu/ai_experiment/104
npm install playwright
npx playwright install chromium
```

### 2. Run the Script
```bash
node 104_auto_apply_gemini.js --start-page 2 --max-pages 5
```

## Usage

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--start-page <num>` | Starting page number | 2 |
| `--max-pages <num>` | Number of pages to process | 5 |
| `--headless` | Run without visible browser | false |
| `--help` | Show help message | - |

### Examples

```bash
# Process pages 2-6 (5 pages)
node 104_auto_apply_gemini.js --start-page 2 --max-pages 5

# Process pages 5-14 (10 pages)
node 104_auto_apply_gemini.js --start-page 5 --max-pages 10

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
Run the 104 job application automation script with:
node /Users/jerryliu/ai_experiment/104/104_auto_apply_gemini.js --start-page 2 --max-pages 5
```

Or ask Gemini to:
1. Navigate to the script directory
2. Run the script with your desired parameters

## Prerequisites

1. **Node.js** must be installed
2. **Playwright** must be installed (`npm install playwright`)
3. **You must be logged into 104.com.tw** - the script will wait 30 seconds for manual login if needed

## How It Works

1. Opens browser to 104.com.tw job search page
2. For each job on the page:
   - Clicks the "應徵" (Apply) button
   - Opens new tab with application form
   - Selects cover letter "自訂推薦信1"
   - Submits the application
   - Verifies success
   - Closes tab and returns to search page
3. Moves to the next page and repeats
4. Shows final summary of results

## Output

The script shows real-time progress:
```
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

### "Browser not installed"
```bash
npx playwright install chromium
```

### "Not logged in"
The script will pause for 30 seconds - log in manually in the browser window.

### "Timeout errors"
- Check your internet connection
- Try running with fewer pages: `--max-pages 2`

## File Location

```
/Users/jerryliu/ai_experiment/104/104_auto_apply_gemini.js
```
