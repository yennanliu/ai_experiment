# Skill Factory — Building an Uber-Style Agent Skill Platform in `skill_factory/`

> Source: Uber Engineering, *"How Uber Evals 1 Million Agent Skill Invocations a Month"* (Nov 23, 2026)
> — X article by @UberEng, full post at <https://www.uber.com/us/en/blog/optimizing-ubers-agent-skills/>.
> Companion post: Uber Engineering, *"Running a Software Factory Efficiently at Uber Scale"* (Aug 27, 2026)
> — <https://www.uber.com/us/en/blog/efficient-software-factory/>.
> Spec for the skill format: <https://agentskills.io>.
>
> 中文版：[DESIGN.zh-TW.md](./DESIGN.zh-TW.md)

---

## 0. TL;DR

Uber treats agent skills as **software products**: they live in Git, they have owners and tiers, they are
generated from a spec, linted, evaluated (with and without the skill), measured for cost per call, and fed
by a stream of bug reports filed *by the agents themselves*.

This doc describes a scaled-down version we can build in `skill_factory/` for a single developer or a small
team using Claude Code (plus optionally Codex / OpenCode / Gemini CLI). It reuses what is already in this
repo — `~/.claude/skills/lc-add`, `~/.claude/skills/lc-coach`, `claude_code/skills_poc/.claude/skills/*`,
and the 104.com.tw automation guide in `../CLAUDE.md` — as the first skills to bring under management.

The goal is not 5,000 skills. The goal is that **every skill we keep is proven to help, costs a known
amount, and gets better over time without us noticing a problem first.**

---

## 1. What Uber built (summary)

| Area | What Uber does | Key number / detail |
|---|---|---|
| Scale | Shared skills marketplace across Claude Code, OpenCode, Codex and an internal agent | 1 skill (Oct 2025) → 5,000+ (Oct 2026); >1M invocations / month |
| Format | Agent Skills spec: a directory with `SKILL.md` + optional scripts + references | 3-stage loading: name+description (~100 tokens) always; body (<500 lines) on use; scripts/refs on demand |
| Distribution | A Git repo **is** the marketplace | Ownership, review and history for free |
| Tiers | Always-on (~100), Core (platform-reviewed), Domain (team-owned, 1 peer approval + CI) | |
| Launcher | `aifx` CLI: starts each agent, installs skills/plugins/MCP, 6-layer config merge | company → team → repo → dir → user → local; later layers only add |
| Authoring | **Skill Workshop**: write a *spec* (goal, outputs, required/forbidden tools, target agents/models); it compiles `SKILL.md` + test prompts | Recompiles when a model or agent changes |
| Discovery | `find-skills` skill; ranking by evals + run counts; quality score | |
| Telemetry | OpenTelemetry events from agent hooks; no prompt text, no file contents | 12 measures per skill (users, retention, reach, co-use, cost, context, latency, errors, tools, versions, agent, installs) |
| Cost / call | Session-cost snapshot after the call minus snapshot before; daily total ÷ calls | |
| Evals | ~5% golden (fixed I/O + tool mocks), ~80% real prompts graded on **trajectory**, ~15% author-judged | Evals passing 70–90% are more useful than 100% |
| Ablation | Replay real prompts with/without the skill (and with parts removed); blind judge | Biggest wins: skills that unlock things the agent can't find (internal docs, warehouse queries) |
| Clean room | Run evals in remote sandboxes (Minion, AES, devpods) | Caught hidden laptop dependencies |
| Linter | Rules distilled from good skills | Numbered steps, one objective each, verification per step, error-recovery table |
| Cost levers | Code Mode (scripts instead of MCP tool dumps), cheaper subagent workers, run scripts at load time, move polling off the main session | MCP defaults loaded 50–70k tokens; tracing skill −46%; search skill −32% |
| Feedback | Always-on `log-papercut` skill — agents file structured friction reports | Thousands filed in 3 months, hundreds fixed |
| Human signal | PRs that get review comments / rejected feed back as eval cases | |

### Lessons Uber states
1. Store skills in Git.
2. Write a spec before the skill, so intent survives model changes.
3. Test with **and without** the skill; some skills should be retired.
4. Measure cost per skill.
5. Clean-room evals catch laptop assumptions.
6. Let agents report friction and route reports to owners.

### 1.1 The bigger picture: Uber's "software factory" (Aug 2026 post)

The skills post is one slice of a larger system. The earlier post explains the **factory** skills run in
and how Uber keeps it cheap while usage grows.

| Area | What Uber does | Key number / detail |
|---|---|---|
| Adoption | Local + cloud agents across the SDLC | >70% of PRs attributed to agents; weekly active users ×7 and requests ×9.4 (Feb→Aug 2026) while total spend stayed flat since April |
| Cost equation | Spend = users × sessions/user × **turns/session × requests/turn × tokens/request** × price/token | Optimize the middle three; the first two are adoption you *want* |
| Managed agents | Agents started by systems, not people: code review (uReview), self-healing CI, end-to-end PRs, alert triage, debugging, maintenance | Each one gets target outcome metrics → an eval benchmark → the cheapest model that meets it (Pareto) |
| Model choice | Uber SWE Benchmark built from thousands of real PRs | Pick models per agent by benchmark, not by habit |
| Subagent default | Main model plans and checks; subagents on a cheaper model do well-defined work | Called "the most impactful lever" |
| Tools | One MCP gateway (>1,000 servers); MCP tools exposed as **CLI commands**; tool search; code-mode plugin | Preloading >100 tools = 50–70k tokens/turn; code mode saved 55–71% on simple queries, >90% on bulk work |
| Grounding | AI Context Graph (24M nodes, 80M edges, 30+ systems) | Grounded run: 38 s and correct. Ungrounded: 20 min, 3 errors, wrong. "An ungrounded agent fails slowly rather than cheaply." |
| Context & cache | Auto-compact at 400k tokens even on 1M models; reasoning effort default Medium; 1-hour cache TTL for interactive sessions, 5-minute for subagents | Cache reads cost 0.1× input |
| Visibility | Live cost in the status line; session analyzer flags **16 anti-patterns**; Slack nudges at 50/80/100% of expected spend | No hard caps — visibility and nudges instead |
| Measurement | Hold the model fixed when measuring a gain; outcome metrics per agent | Cost per merged PR / review / alert, with revert rate, F1, MTTR as quality |

Anti-patterns the session analyzer looks for (examples from the post): using Opus where Sonnet would do,
big MCP payloads (e.g. 40 KB) staying in context, cache expired after a break, ~100k tokens of instructions
and tools preloaded before the first prompt.

**What this adds to our design:** skills are only one lever. We also want (a) per-*session* cost analysis,
not just per-skill, (b) a live cost readout, (c) a "managed agent" path where a skill runs headless on a
schedule with its own outcome metric, and (d) a grounding/context layer so skills don't burn turns searching.

---

## 2. Mapping Uber → `skill_factory`

| Uber component | Our version | Size of effort |
|---|---|---|
| Skills marketplace repo | `skill_factory/skills/` in this repo, published as a Claude Code plugin marketplace | S |
| Tiers | `tier:` field in `spec.yaml` — `always-on`, `core`, `domain`, `experimental` | S |
| `aifx` launcher | `sf` — a small Python CLI (`sf install`, `sf new`, `sf lint`, `sf eval`, `sf report`) | M |
| 6-layer config | 3 layers: `skill_factory/profiles/default.yaml` → repo `.skillfactory.yaml` → `~/.skillfactory.local.yaml` | S |
| Skill Workshop | `sf new` + `sf compile`: spec → `SKILL.md` + eval prompts, driven by Claude itself | M |
| OTel telemetry | Claude Code hooks → local JSONL (phase 1), optional OTel collector (phase 3) | S → M |
| Dashboard | `sf report` → static HTML (or a published artifact) | M |
| Eval harness | `claude -p --output-format stream-json` runner + trajectory grader + LLM judge | L |
| Clean room | Docker container or git worktree with an empty `$HOME` | M |
| Linter | `sf lint` — deterministic rules + optional LLM review | S |
| `log-papercut` | An always-on skill that appends to `papercuts/inbox.jsonl` | S |
| `find-skills` | A skill that searches `skills/*/spec.yaml` and the usage stats | S |
| Status-line cost counter | Claude Code `statusLine` script showing session cost | S |
| Session analyzer (16 anti-patterns) | `sf analyze` over transcripts — see section 9.1 | M |
| Managed agents | Skills run headless via `claude -p` on cron / CI (e.g. nightly 104 run, PR review) with an outcome metric | M |
| MCP gateway + CLI projection | Wrap the MCP tools we use (Playwright, etc.) as scripts under `skills/*/scripts/` | S–M |
| AI Context Graph | A `context/` folder of repo maps + a `lookup` script, loaded on demand (not a graph DB) | M |
| Uber SWE Benchmark | Our eval suites, re-run across models to pick the cheapest that passes | (reuses §7) |

---

## 3. Repository layout

```
skill_factory/
├── README.md
├── docs/
│   ├── DESIGN.en.md            ← this file
│   └── DESIGN.zh-TW.md
├── .claude-plugin/
│   └── marketplace.json        ← makes the repo installable as a Claude Code plugin marketplace
├── plugins/
│   ├── always-on/              ← plugin = bundle of skills, one per tier
│   │   ├── .claude-plugin/plugin.json
│   │   └── skills/ -> ../../skills (symlinked or generated)
│   ├── core/
│   └── domain-<name>/
├── skills/
│   ├── lc-add/
│   │   ├── spec.yaml           ← source of truth (human-written)
│   │   ├── SKILL.md            ← compiled from spec (checked in, reviewed)
│   │   ├── scripts/            ← deterministic code ("Code Mode")
│   │   ├── references/         ← loaded on demand
│   │   └── evals/
│   │       ├── golden/*.yaml   ← fixed input → expected output, tool mocks
│   │       ├── real/*.yaml     ← real prompts + expected trajectory
│   │       └── judged/*.yaml   ← author-graded set for calibrating the LLM judge
│   ├── lc-coach/
│   ├── log-papercut/
│   ├── find-skills/
│   └── job-apply-104/          ← the 104.com.tw flow from ../CLAUDE.md, as a skill
├── hooks/
│   ├── telemetry.py            ← PostToolUse / SessionStart / Stop hook → events.jsonl
│   └── settings.snippet.json
├── sf/                         ← the CLI (Python, uv-managed)
│   ├── cli.py
│   ├── compile.py
│   ├── lint.py
│   ├── eval/
│   │   ├── runner.py
│   │   ├── trajectory.py
│   │   ├── judge.py
│   │   └── ablation.py
│   └── report.py
├── profiles/
│   └── default.yaml
├── papercuts/
│   └── inbox.jsonl             ← git-ignored, or committed after triage
├── telemetry/                  ← git-ignored raw events
└── .github/workflows/ci.yml    ← lint + golden evals on every PR
```

---

## 4. The skill spec (write this first)

Uber's lesson #2: the spec, not the prose, is the source of truth. When the model changes, recompile.

```yaml
# skills/lc-add/spec.yaml
name: lc-add
version: 1.3.0
tier: domain            # always-on | core | domain | experimental
owner: jim
domain: leetcode
goal: >
  File a LeetCode solution into the repo the way existing ones are filed.
triggers:               # becomes the description (~100 tokens, always loaded)
  - "add LC <number> to <folder>"
  - "file this solution and update the README"
outputs:
  - A Python file in the house layout (docstring, V0 with IDEA block, time/space line)
  - A README row inserted in sorted position
tools:
  required: [Read, Write, Edit, Bash]
  forbidden: [WebSearch]      # slug lookup must use the local script
  scripts:
    - scripts/find_slug.py    # deterministic: runs instead of a model turn
targets:
  agents: [claude-code, codex]
  models: [claude-sonnet-5-5, claude-opus-5-5]
budget:
  max_body_lines: 500
  target_cost_usd: 0.15       # per invocation, used by `sf report` alerts
evals:
  min_pass_rate: 0.7
  max_pass_rate: 0.95         # >95% means the eval is probably too easy
```

`sf compile skills/lc-add` sends the spec + the linter rules + 2–3 exemplary skills to Claude and writes:
- `SKILL.md` (frontmatter `name`/`description` + numbered steps),
- `evals/real/*.yaml` draft prompts the author then edits.

The compiled `SKILL.md` is checked in and reviewed like code. Recompile is a PR, never a silent change.

---

## 5. Distribution & install (`sf`, our `aifx`)

### 5.1 Claude Code plugin marketplace
Claude Code can install plugins from a Git-hosted marketplace (`.claude-plugin/marketplace.json`), and each
plugin can ship skills, hooks and commands. Using that means **no custom sync code** for Claude Code:

```bash
/plugin marketplace add <git-url-or-path-to>/skill_factory
/plugin install always-on@skill-factory
/plugin install core@skill-factory
```

> Verify the current marketplace/plugin schema against the Claude Code docs before implementing; it has
> changed across versions.

### 5.2 `sf install` for other agents
For Codex / OpenCode / Gemini CLI, `sf install --agent codex` copies or symlinks `skills/<name>/` into
that agent's skill directory and writes any agent-specific frontmatter. This is the "handles differences
per agent" job `aifx` does.

### 5.3 Layered profiles
```yaml
# profiles/default.yaml          (layer 1: "company")
plugins: [always-on]
# <repo>/.skillfactory.yaml       (layer 2: repo)
plugins: [core, domain-leetcode]
# ~/.skillfactory.local.yaml      (layer 3: user/local)
plugins: [experimental]
```
Rule copied from Uber: **later layers can only add**, never remove. This keeps "always-on" actually always on.

---

## 6. Linter (`sf lint`)

Deterministic checks, fast enough for a pre-commit hook. Start with:

| Rule | Why |
|---|---|
| Frontmatter has `name` and `description`; description ≤ ~100 tokens and says *when* to use it | Stage-1 loading cost and trigger accuracy |
| Body ≤ 500 lines | Stage-2 budget |
| Steps are numbered; each step has one objective | Uber linter rule |
| Each step ends with a verification ("confirm X exists", "check exit code") | Uber linter rule |
| Commands are concrete (a code block), not "run the tests" | Uber linter rule |
| Has an error-recovery table (`symptom → action`) | Uber linter rule |
| No `.mcp.json` inside a skill/plugin | Uber blocked these — 50–70k token tool dumps |
| No secrets / credentials (regex + `gitleaks`) | `../CLAUDE.md` already warns about this |
| No shared temp-file paths (`/tmp/commit_msg.txt`) — use `mktemp` | Real Uber papercut: concurrent sessions overwrote each other |
| Scripts referenced in `SKILL.md` exist and are executable | Broken-link class of bugs |
| `spec.yaml` and `SKILL.md` are in sync (hash of spec stored in SKILL.md comment) | Detects hand-edits that bypass the spec |

Optional `sf lint --llm`: a Claude review against the same rules for the fuzzy ones.

---

## 7. Evaluation

### 7.1 Three suites per skill (Uber's 5 / 80 / 15 mix)

**Golden (~5%)** — fixed input, exact expected output, tools mocked. Cheap; run on every PR.
```yaml
# evals/golden/two-sum.yaml
prompt: "add LC 1 to Hash_table"
fixtures: { repo: fixtures/mini_repo }
mocks:
  scripts/find_slug.py: { stdout: "two-sum" }
expect:
  files_created: ["Hash_table/two-sum.py"]
  readme_contains: "| 1 | [Two Sum]"
```

**Trajectory (~80%)** — real prompts (taken from your own history), graded on *what the agent did*:
```yaml
# evals/real/contest-q1.yaml
prompt: "file yesterday's contest Q1, it's in scratch.py"
expect_trajectory:
  must_call_skill: lc-add
  ordered:
    - { tool: Read, path_glob: "**/scratch.py" }
    - { tool: Bash, cmd_regex: "find_slug.py" }
    - { tool: Write }
    - { tool: Bash, cmd_regex: "python .*\\.py" }   # smoke test
    - { tool: Edit, path_glob: "README.md" }
  forbidden: [WebSearch]
  max_turns: 12
```

**Judged (~15%)** — open-ended outputs (e.g. `lc-coach` verdicts). The author grades a set by hand; that
set calibrates an LLM judge, and the judge's agreement with the author is itself tracked.

### 7.2 Runner
Claude Code's headless mode streams every tool call, so we can grade trajectory without instrumenting the agent:

```bash
claude -p "$PROMPT" \
  --output-format stream-json --verbose \
  --model claude-sonnet-5-5 \
  --plugin-dir ./plugins/core \
  --max-turns 20 \
  > runs/$SKILL/$CASE/$RUN.jsonl
```

`sf/eval/runner.py` runs each case **N times (default 5)** because agents are non-deterministic, then
`trajectory.py` parses `tool_use` events (including the `Skill` tool call itself) and scores each run.
Report the **pass rate**, not pass/fail. Target 70–90%: at 100% the eval is likely blind to real changes.

### 7.3 Ablation (does the skill earn its tokens?)
`sf eval --ablate lc-add` replays the real prompts three ways:
1. with the skill,
2. without the skill,
3. with one section removed at a time (e.g. no error table).

A **blind judge** (shuffled A/B, model doesn't know which is which) picks the better answer; we also record
cost, turns and tool errors per arm. Decision rule:

| Result | Action |
|---|---|
| Wins and costs ≤ baseline | Keep |
| Wins but costs much more | Optimize (section 9) |
| Ties | Shorten or retire — models got better |
| Loses | Retire |
| A section's removal doesn't hurt | Delete that section |

### 7.4 Clean room
Run evals in a container with an empty `$HOME`, no `~/.claude`, and only the fixture repo:
```bash
docker run --rm -e ANTHROPIC_API_KEY \
  -v "$PWD/fixtures/mini_repo:/work" -v "$PWD/plugins:/plugins:ro" \
  skill-factory-eval:latest \
  claude -p "$PROMPT" --plugin-dir /plugins/core --output-format stream-json --verbose
```
This is what catches "works on my laptop" skills: hard-coded `/Users/jliu/...` paths, a logged-in `gh`,
a cached file, macOS-only `sed -i ''`.

### 7.5 CI
`.github/workflows/ci.yml` on every PR touching `skills/**`:
1. `sf lint` (blocking),
2. golden evals (blocking),
3. trajectory evals for changed skills, N=3 (non-blocking, posted as a PR comment with pass-rate delta vs `main`).

---

## 8. Telemetry & cost per skill

### 8.1 Collecting events (phase 1: local, no infra)
Claude Code hooks receive a JSON payload on stdin for events such as `SessionStart`, `PreToolUse`,
`PostToolUse`, `UserPromptSubmit` and `Stop`. A skill invocation shows up as a tool call to `Skill`.

```json
// hooks/settings.snippet.json  → merge into ~/.claude/settings.json or ship in the plugin
{
  "hooks": {
    "PostToolUse": [{ "matcher": "Skill", "hooks": [{ "type": "command", "command": "python3 $SF_HOME/hooks/telemetry.py skill" }] }],
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "python3 $SF_HOME/hooks/telemetry.py session_start" }] }],
    "Stop":         [{ "hooks": [{ "type": "command", "command": "python3 $SF_HOME/hooks/telemetry.py stop" }] }]
  }
}
```

`telemetry.py` appends one line to `telemetry/events.jsonl`:
```json
{"ts":"2026-11-24T10:02:11Z","event":"skill","session":"…","skill":"lc-add","version":"1.3.0",
 "trigger":"model|user","agent":"claude-code","model":"claude-sonnet-5-5","cwd_hash":"…"}
```
Privacy rule (from Uber): **no prompt text, no file contents**. Hash the cwd.

### 8.2 Cost per invocation
Uber: cost = session cost snapshot *after* the call − snapshot *before*. Two ways to get session cost:
- **OTel**: set `CLAUDE_CODE_ENABLE_TELEMETRY=1` with an OTel exporter; Claude Code emits per-API-request
  cost/token events. Run a local collector (or just the console/file exporter) and join on session id + time.
- **Transcript**: the `Stop` hook gets `transcript_path`; sum token usage per assistant message between the
  skill call and the next skill call / session end, multiply by model price.

`sf report` then computes per skill, per day: invocations, distinct sessions, cost/call (mean & p90),
turns/call, tool errors/call, model-vs-user trigger ratio, versions in use, and co-use (skills seen in the
same session).

### 8.3 The 12 measures, scaled down
| Uber measure | How we get it |
|---|---|
| users / reach / installs | distinct `cwd_hash` or machine (single user: distinct repos) |
| retention | skill used in ≥2 different weeks |
| co-use | skills in the same session |
| cost, context, latency | OTel or transcript |
| errors, tools | `PostToolUse` with error results in the skill window |
| versions, agent | event fields |

---

## 9. Cost levers (apply after measuring, measure again after)

1. **Code over LLM ("Code Mode").** Any step that is deterministic (slug lookup, README sort, form-filling
   selectors in the 104 flow) becomes a script in `scripts/`. The model calls one command instead of
   reasoning through it. Never ship an MCP server just to get a few tools — wrap the call in a script.
2. **Cheaper workers.** Inside a skill, dispatch a subagent for bulk/mechanical work on a cheaper model
   (e.g. Haiku 5.5 for "collect job links on this page"). Uber saw −46% on one skill and 0% on another —
   so measure each change.
3. **Run scripts at load time.** If a skill always starts by running `git status` or listing a folder,
   pre-compute it and inject the result, saving a whole model turn (Uber: 4 → 3 turns, −32%).
4. **Get polling off the main session.** Waiting on CI, a deploy, or the 104 "apply done" page should
   happen in a background agent or a script loop, not as repeated turns in your expensive session.

5. **Pick the model by benchmark.** Run each skill's eval suite on Haiku / Sonnet / Opus and set the
   cheapest model that clears `min_pass_rate` as the skill's default (`targets.models[0]`). Re-run when a
   new model ships. Hold the model fixed when you measure any other optimization.
6. **Ground before searching.** If evals show the agent spending turns on `grep`/`find`/web search, add
   a `references/` file or a lookup script with the answer. Uber's grounded run was 38 s vs 20 min.
7. **Tune context and cache.** Keep always-loaded text small (descriptions, CLAUDE.md, always-on MCP).
   Note that `../CLAUDE.md` loads the full 104.com.tw guide into *every* session under `ai_experiment/` —
   moving it into the `job-apply-104` skill is itself a cost win.

### 9.1 Session analyzer (`sf analyze`)
Per-skill numbers miss waste that happens between skills. `sf analyze` reads the transcripts in
`~/.claude/projects/*/` (the JSONL the CLI already writes) and flags anti-patterns, starting with the
ones Uber named:

| Anti-pattern | Detection |
|---|---|
| Expensive model on easy work | Opus turns with short outputs and only Read/Grep tool calls |
| Large tool result kept in context | Any tool result > 20 KB that stays in context for many later turns |
| Cache expired after a break | Gap between turns > cache TTL followed by a large cache write |
| Heavy preloaded context | First-request input tokens > 30k before the user's first prompt |
| Polling in the main session | ≥3 near-identical Bash calls (`gh pr checks`, `sleep`) in a row |
| Search thrash | > 8 Grep/Glob/find calls before the first Edit |
| Retry loops | Same failing command run ≥ 3 times |

Output: a weekly report ranking sessions by cost with the anti-patterns found and a suggested fix (often
"make this a skill" or "move this to a script").

### 9.2 Live cost readout
A Claude Code `statusLine` command receives session info (including cost) as JSON on stdin and prints one
line. Show `$cost · model · active skill`. Uber's approach: no hard caps, just make the cost visible and
nudge (e.g. a notification at 50/80/100% of your own daily budget).

### 9.3 From skills to managed agents
Once a skill passes its evals reliably, run it **without a human starting it**:
```bash
# cron / GitHub Actions
claude -p "/job-apply-104 page=1 dry_run=false max=20" \
  --model claude-haiku-5-5 --output-format stream-json --verbose > runs/$(date +%F).jsonl
```
Each managed agent gets an outcome metric in its spec (`outcome: cost per successful application`,
`quality: applications later withdrawn / rejected-as-spam`), and its runs feed telemetry and papercuts
like any other session. Keep a human checkpoint for anything with outside effects.

---

## 10. Papercuts: let the agent file the bugs

An **always-on** skill `log-papercut`:

```markdown
---
name: log-papercut
description: Use when a skill, script or tool behaved unexpectedly, was confusing, or needed a workaround. Files a short structured report. Use it even if you recovered.
---
1. Write one JSON line to `$SF_HOME/papercuts/inbox.jsonl` with:
   skill, skill_version, tried, expected, happened, suggested_fix, severity (low|med|high).
2. Do not include secrets, prompt text, or file contents beyond a ≤3-line error excerpt.
3. Verify the line was appended (`tail -n1`), then continue the user's task.
```

Weekly loop (`sf papercuts`): group by skill → triage → root cause → fix → **add the case to that skill's
evals** → PR → measure for regressions → close. Every fixed papercut becomes a regression test.

Uber's examples are good lint rules to steal: shared temp files, seconds-vs-milliseconds, silent wrong
values for missing keys, waiting forever on CI that doesn't exist.

**Human signal**: when you reject or heavily edit an agent's output that a skill produced, that prompt
becomes an eval case (`sf eval add --from-session <id>`).

---

## 11. Discovery: `find-skills`

A small skill that, given a task description, searches `skills/*/spec.yaml` (goal, triggers, domain) and
ranks by `sf report` data: eval pass rate × usage × recency, minus cost. It answers "is there already a
skill for this?" before someone writes a duplicate. Quality score:

```
quality = 0.4·eval_pass_rate_norm + 0.3·ablation_win_rate + 0.2·retention + 0.1·(1 − cost_vs_target)
```

---

## 12. Roadmap

| Phase | Deliverable | Done when |
|---|---|---|
| **0 — Bootstrap** (½ day) | Repo layout, `marketplace.json`, move `lc-add`, `lc-coach`, `skills_poc/*` in; write `spec.yaml` for each | `/plugin install` works from this repo |
| **1 — Lint + telemetry** (1–2 days) | `sf lint`, pre-commit hook, `hooks/telemetry.py`, `log-papercut` | Every skill call lands in `events.jsonl` |
| **2 — Evals** (3–5 days) | Runner, trajectory grader, golden suite for each skill, 5–10 real prompts each | Pass rates reported for every skill |
| **3 — Cost + report** (2–3 days) | Cost per call via OTel or transcript, `sf report` HTML | One page shows cost/call and pass rate per skill |
| **4 — Spec compiler** (2–3 days) | `sf new`, `sf compile`, spec↔SKILL hash check | New skills start from a spec |
| **5 — Ablation + clean room** (3–5 days) | Docker eval image, `sf eval --ablate`, blind judge | Every core skill has a keep/shorten/retire verdict |
| **6 — CI + discovery** (2 days) | GitHub Actions, `find-skills`, quality score | PRs show pass-rate deltas |
| **7 — Factory efficiency** (2–4 days) | Status-line cost, `sf analyze` session analyzer, per-skill model benchmark | Weekly report of top-cost sessions + anti-patterns |
| **8 — Managed agents** (2–3 days) | First scheduled headless skill with an outcome metric | One skill runs on a schedule and reports cost per outcome |
| **Later** | A/B tests between skill versions, per-model compilation, task router | — |

### First skills to onboard
| Skill | Why it's a good first candidate |
|---|---|
| `lc-add` | Clear, checkable outputs → easy golden evals |
| `lc-coach` | Open-ended → exercises the judged suite and judge calibration |
| `python-code-reviewer`, `fastapi-master` (from `claude_code/skills_poc`) | Good ablation targets: do they beat the base model now? |
| `job-apply-104` (from `../CLAUDE.md`) | Browser automation → tests Code Mode (move selectors into a Playwright script), polling-off-main-session, and clean-room (needs login; must never contain credentials) |
| `log-papercut`, `find-skills` | Platform skills, always-on |

---

## 13. Risks & open questions

| Risk | Mitigation |
|---|---|
| Eval cost (N runs × cases × models) | Golden on every PR; trajectory only for changed skills; nightly full run on a cheaper model |
| Non-determinism makes deltas noisy | Report pass rate with N≥5 and a confidence interval; don't block on small deltas |
| LLM judge bias | Blind A/B order, calibrate on the author-graded set, track judge–author agreement |
| Telemetry leaking private data | No prompts/file contents; hash paths; `telemetry/` git-ignored |
| Skills with side effects (the 104 flow submits real applications) | Evals run against mocks/fixtures or a dry-run flag; never against the live site in CI |
| Agent/format drift (Claude Code, Codex change their skill schema) | Spec is the source of truth; `sf compile` per agent; pin versions in CI |
| Rare, high-stakes workflows are hard to eval (Uber's open problem too) | Human checkpoint steps inside the skill; log every run for later review |

---

## 14. Glossary

- **Skill** — a directory with `SKILL.md` (+ scripts, references) that an agent loads when relevant.
- **Plugin** — a bundle of skills (and hooks/commands) installed together; we use one per tier.
- **Trajectory** — the ordered list of tool calls an agent made; graded instead of (or in addition to) the final answer.
- **Ablation** — re-running with a skill or part of a skill removed to measure what it contributes.
- **Papercut** — a small, structured friction report filed by the agent.
- **Clean room** — an eval environment with none of the developer's local state.
