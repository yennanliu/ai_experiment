# Harness Engineering Guide — Notes

Source: https://rar.design/posts/harness-engineering-guide  
Published: early 2026

---

## What it is

Harness Engineering is the third evolution in AI engineering practice:

```
2022  Prompt Engineering   →  how to speak to the model once
2024  Context Engineering  →  what to put into the context window
2026  Harness Engineering  →  model-external infrastructure for reliable, multi-session work
```

The name comes from the literal meaning of "harness": equipment that directs a powerful but difficult-to-control force. Applied to AI: *all the model-external things that let models do useful work reliably over time*.

---

## The core problem it solves

Long-running agents fail predictably in four ways:

1. **Memory discontinuity** — context resets between sessions; prior work is invisible
2. **Phantom completion** — agents claim tasks are done without verifying
3. **Quality drift** — no external evaluation creates inflated self-assessments
4. **Invisible knowledge** — rules in Google Docs, Slack, or human heads don't exist for the agent

The key principle: *"From the agent's perspective, anything not in-context during execution doesn't exist."*

---

## Evidence: the two landmark case studies

### OpenAI (Feb 2026)
- 3 engineers, 5 months, ~1 million lines of agent-written code
- **Zero human-written lines**
- Method: all context (architecture decisions, API rationales, team consensus) was converted to repository files; background agents scanned for rule violations and auto-generated repair PRs

### Anthropic — Solo vs. Harness comparison
Same prompt: *"Build a 2D retro game editor with level editing, sprite editing, entity behavior, playable test mode"*

| | Solo Agent | Full Harness |
|---|---|---|
| Time | 20 min | 6 hours |
| Cost | $9 | $200 |
| Result | Appears functional; character won't respond to input; core wiring severed — **invisible failure** | Actually playable; sprite editor works; built-in Claude assistant; core physics loop functional |

Same harness later produced a full DAW (composition, reverb, AI-assisted composing) in 3h 50m for $124.

**The lesson:** cosmetic similarity between harness and non-harness output hides fundamental differences in actual functionality.

---

## Five components

### 1. Repository Structure & Readability
All standing instructions live as repo files: `AGENTS.md`, `CLAUDE.md`, `SKILL.md` (naming flexible, format fixed).

> "If rules aren't in files, they don't exist for the agent."

### 2. Architectural Constraints
Core principles expressed as mechanically verifiable rules. Background agents scan for violations and auto-generate repair PRs. Framing: "small continuous payments prevent massive technical debt."

### 3. Tools & Model Context Protocol (MCP)
MCP acts as the expansion slot — Playwright for UI testing, Puppeteer for browser interaction, database MCPs for queries.

> "An agent without tools is an engineer without hands."

### 4. Context Management
Three mechanisms: **compaction**, **context reset**, **handoff artifacts** (progress files written after each session so the next session resumes cleanly).

Note on model capability: Claude Sonnet 4.5 shows "context anxiety" (premature completion claims); Opus 4.5/4.6 eliminates this, so some harness elements expire as models improve.

### 5. Evaluation Loops
The most overlooked and highest-impact component. Single agents self-evaluate leniently. A dedicated evaluator agent, designed specifically to find problems, produces honest scores and drives measurable quality improvement.

---

## Anthropic's three-agent architecture

```
PLANNER
  Takes a one-line prompt
  Outputs a complete spec (e.g. 200 words → 16 features × 10 sprints)

GENERATOR
  Implements one feature per session
  Uses git for versioning
  Self-assesses before handoff

EVALUATOR
  Uses Playwright MCP to actually test the output
  Clicks UI, verifies API responses, checks DB state
  Contracts with Generator on success criteria upfront
```

Earlier two-agent version:
- **Initializer** — runs once, produces feature JSON (200+ items) + init scripts + initial commits
- **Coding agent** — each session reads the same JSON, git logs, and progress files before starting

Feature lists stored as JSON rather than markdown because *"models are less likely to randomly modify JSON files."*

---

## Minimum viable harness

Three elements resolve 80% of friction:

1. **Rule file** — `SKILL.md` / `CLAUDE.md` / `AGENTS.md` with stable format
2. **Progress file** — updated after each session for continuity
3. **Checking mechanism** — a separate agent that critiques the output

Starting investment: ~30 minutes defining standards for a recurring workflow yields permanent quality gains.

---

## Harness decay

Harness elements encode hidden assumptions about model capability (*"model can't do X"*). Stronger models invalidate those assumptions:

- Sonnet 4.5 required context reset → Opus 4.6 doesn't
- Harness degrades gracefully but needs periodic audit

Validation method: run the same task with and without the harness. If the difference is negligible, discard that element. If it's dramatic, it's a keeper.

---

## The mindset shift

| Before | After |
|---|---|
| Explain rules each interaction | Write rules to files; agent self-reads |
| Ask agent to self-verify | Deploy a separate agent to find flaws |
| Complete everything in one prompt | Split across sessions with progress notes |
| "What's the strongest model?" | "What does my infrastructure around the model look like?" |

Model rankings become outdated every 90 days. Infrastructure compounds as a lasting asset.

---

## Relation to existing tools

All of these are harness components already in the wild:
- Anthropic Skill system
- Claude Code plugins / `CLAUDE.md`
- Cursor's `.cursorrules`
- OpenAI's `AGENTS.md`

Common trait: pre-written rules loaded automatically at execution, eliminating repetitive re-explanation.
