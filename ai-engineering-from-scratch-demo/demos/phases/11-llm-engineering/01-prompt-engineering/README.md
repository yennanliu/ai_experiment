# Prompt Engineering — the lesson's patterns against a real Claude

**Lesson:** [11-llm-engineering / 01-prompt-engineering](https://yennj12.js.org/ai-engineering-from-scratch/lesson.html?path=phases%2F11-llm-engineering%2F01-prompt-engineering)
**Tier:** T2 (replayed from a cassette by default) · **Install:** `uv sync --extra llm`

## What it proves

The lesson scores prompts with `score_response()` — and feeds it
`simulate_llm_call()`, which seeds a hash with the prompt text. So the numbers
it prints measure the simulator, not the prompt.

This demo runs the lesson's own `build_prompt` / `format_anthropic_request` /
`score_response` over responses a real model produced, and prints the real and
simulated scores side by side under identical criteria.

One thing only a real call can tell you also falls out: the lesson's
`format_anthropic_request()` sets `temperature`, which current Claude models
reject with a 400. `adapt_request()` shows the fix.

## Run

```bash
uv run demo run phases/11-llm-engineering/01-prompt-engineering   # replay, free, offline
DEMO_MODE=live uv run demo run phases/11-llm-engineering/01-prompt-engineering
```

Replay needs no key and no network. Live mode needs `ANTHROPIC_API_KEY`,
re-records `cassettes/prompt-patterns.json`, and prints what the run cost.
Until someone records the tape once, the demo skips cleanly and says so.
