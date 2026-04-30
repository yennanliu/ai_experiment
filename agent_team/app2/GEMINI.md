# News Scraping Agent Team

This project uses a 3-agent news scraping team. Use the agents in order for any non-trivial scraping task:

## Team

| Agent | Trigger | Role |
|-------|---------|------|
| `@project-leader` | New scraping task | Reads the target and produces a numbered scraping plan |
| `@scraper` | After plan is confirmed | Executes the plan, fetches pages, saves structured output |
| `@reviewer` | After scraper finishes | Reviews data quality, correctness, and ethical compliance |

## Workflow

```
User task → @project-leader → (user confirms plan) → @scraper → @reviewer → done
```

## Output

Scraped data is saved to `output_gemini/` as JSON or CSV.

## When to skip agents

- Re-running an existing scraper with no changes: skip `@project-leader`, go straight to `@scraper` + `@reviewer`.
- Read-only questions about the data: answer directly without delegating.

## General rules

- Respect robots.txt and add delays between requests.
- Never collect data beyond what the plan specifies.
- Never hardcode credentials — use environment variables.
- If a site blocks scraping, stop and report; do not attempt to circumvent.
