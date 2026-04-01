# Tasker automation (Puppeteer)

Implements the workflow described in `tasker/flow.txt`:

- Open cases list
- Open first case
- Click "我要提案" (opens new tab)
- Fill min/max price within budget range
- Click "送出提案"

## Setup

```bash
cd cursor
npm i
```

## Run

```bash
cd cursor
npm run tasker:apply
```

## Options

- **HEADLESS**: set `HEADLESS=1` to run headless (default: headed)
- **SLOW_MO_MS**: add slow-mo (e.g. `SLOW_MO_MS=50`)
- **START_URL**: override URL (default: `https://www.tasker.com.tw/cases?selected_tags=1`)

