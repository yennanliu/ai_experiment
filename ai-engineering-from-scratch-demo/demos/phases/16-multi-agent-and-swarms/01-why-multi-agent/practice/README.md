<!-- generated:start -->
# 16-multi-agent-and-swarms / 01-why-multi-agent

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/01-why-multi-agent/) · upstream spec
`phases/16-multi-agent-and-swarms/01-why-multi-agent/docs/en.md`

```bash
uv run demo practice run 01-why-multi-agent --ex 1
uv run demo explain 01-why-multi-agent --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/01-why-multi-agent
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a fourth specialist: a "tester" agent that receives code from the coder and review feedba… | code | T0 | `ex01_the_fourth_specialist_costs_five_times_what_the_split_saves.py` |
| 2 | Modify the pipeline so the reviewer can send feedback back to the coder for a revision loop (… | code | T0 | `ex02_the_loop_has_no_way_to_stop_early_because_nothing_can_approve.py` |
| 3 | Convert the sequential pipeline into a fan-out: run the researcher and a "requirements analyz… | code | T0 | `ex03_the_fan_out_already_exists_and_throws_the_review_away.py` |
<!-- generated:end -->

## Answers

The lesson ships no Python. `code/single_vs_multi.ts` is the whole
implementation, so each solution reads it as text, lifts the prompts and the
task string out of it verbatim, and ports `fakeLLMCall`'s arithmetic:

```
tokens = floor((systemPrompt.length + userMessage.length) / 4) + 500
output = `[Response to: ${userMessage.slice(0, 80)}...]`
```

Both lines matter more than they look. The `+ 500` is a flat charge per call,
and the `slice(0, 80)` caps every message at 98 characters forever.

### 1 — the fourth specialist costs five times what the split saves

Priced on the lesson's own task, `Build a rate limiter middleware for an
Express.js API`:

| arm | calls | tokens | flat part | variable part |
|---|---:|---:|---:|---:|
| single agent | 3 | 1745 | 1500 | 245 |
| pipeline | 3 | 1652 | 1500 | 152 |
| pipeline + tester | 4 | 2238 | 2000 | 238 |

The lesson's headline comparison is 1745 against 1652 — the pipeline wins by
**93 tokens, 5.3%**. Adding the tester this exercise asks for costs **493**,
so the pipeline ends **28% worse** than the single agent it was introduced to
beat. The advantage being demonstrated is smaller than one fifth of the price
of one more agent, which is the real lesson about when to split work up.

It is smaller still than it looks: **86%** of the single agent's total and
**91%** of the pipeline's is the flat `+ 500`. The context pollution the demo
dramatises is 245 tokens against 152.

Two things are worth saying about the arms being compared. The single agent's
system prompt numbers four steps — research, write, review, **write tests** —
and `singleAgentApproach` contains three `await fakeLLMCall` lines. It is
charged for the tester's job in every prompt and never does it. And the third
printed metric, tool calls, is `Math.floor(Math.random() * 5) + 1`: uniform on
1..5, independent of task, prompt and architecture. It compares three draws
against three, and four once the tester lands, so it moves with agent count
and nothing else.

### 2 — the loop has no way to stop early because nothing can approve

The revision loop runs coder and reviewer alternately behind a round counter.
Each round is two calls and costs exactly **1140** tokens:

| rounds | calls | tokens | vs single agent |
|---:|---:|---:|---:|
| 0 | 3 | 1652 | −5% |
| 1 | 5 | 2792 | +60% |
| 2 | 7 | 3932 | +125% |

"Max 2 rounds" reads like a safety cap, which implies something else could
stop the loop sooner. Nothing can. `LLMResponse` declares three fields —
`output`, `tokens`, `calls` — and none carries a verdict, and `fakeLLMCall`
has exactly one output expression, a constant template of its own input. There
is no review to write a stopping predicate over, so the cap is the entire
stopping rule and the loop always runs to it.

The same `slice(0, 80)` explains the flat per-round price. Every message in
every round is **98** characters, so round two's review is the same size as
round zero's and a revision costs the same whether there is one line of code
under review or a thousand. Context-window pressure is the thing this lesson
exists to teach, and its simulator truncates it away.

### 3 — the fan-out already exists, and throws the review away

Written independently, the fan-out is four calls and **2238** tokens against
the pipeline's three and **1652** — **586 more, 35%** — for **three**
sequential `setTimeout(50)` ticks either way, because the parallel pair still
has to finish before the coder can start. Parallelism buys an extra opinion
here and no wall clock at all.

Then diff it against `multiAgentFanOut`, which is this exercise, already in
the file forty lines above the exercise list. It makes **four** agent calls and
performs **three** `messages.push`. The one never pushed is the reviewer's:
it is handed `codeResult.content` directly and its result is dropped. Since
`content` is built by mapping over `messages`, **550 tokens** are spent,
counted into the total the demo prints, and absent from what the function
returns.

The two arms are not even plumbed alike. The pipeline routes every hop through
`messages.filter((m) => m.to === ...)` — two filters for three calls. The
fan-out uses one filter for four calls. So the printed comparison is between
an architecture and a shortcut through it.
