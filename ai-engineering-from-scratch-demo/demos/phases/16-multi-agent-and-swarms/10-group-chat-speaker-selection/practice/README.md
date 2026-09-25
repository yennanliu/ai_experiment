<!-- generated:start -->
# 16-multi-agent-and-swarms / 10-group-chat-speaker-selection

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/10-group-chat-speaker-selection/) · upstream spec
`phases/16-multi-agent-and-swarms/10-group-chat-speaker-selection/docs/en.md`

```bash
uv run demo practice run 10-group-chat-speaker-selection --ex 1
uv run demo explain 10-group-chat-speaker-selection --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/10-group-chat-speaker-selection
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Compare the conversation under round-robin vs LLM-selected. Which agent d… | code | T0 | `ex01_round_robin_ships_the_bug_because_the_manager_speaks_between_review_and_fix.py` |
| 2 | Add a "max-speaks-per-agent" rule in the selector. How does it affect the transcript? | code | T0 | `ex02_a_speak_cap_either_changes_nothing_or_blocks_the_fix.py` |
| 3 | Implement a goal-reached termination: stop when the reviewer returns "approved." How often do… | code | T0 | `ex03_the_goal_check_already_exists_one_turn_late_inside_the_manager.py` |
| 4 | Read the AutoGen stable docs on GroupChat (https://microsoft.github.io/autogen/stable/user-gu… | explain | T0 | prose, below |
| 5 | Read the AG2 repo (https://github.com/ag2ai/ag2) and compare its v0.2 GroupChat to the v0.4 e… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

### 1 — round-robin ships the bug because the manager speaks between review and fix

**The coder dominates round-robin, 2 of 4 turns. Nobody dominates the
LLM-style selector, 2/2/2.** The demo prints "Round-robin gives every agent
an equal turn" directly under the counts {coder: 2, reviewer: 1, manager: 1}.

Who dominates by count matters less than what each run ships:

| turn | round-robin | LLM-style |
|---:|---|---|
| 1 | coder: `a - b` (buggy) | manager: continue |
| 2 | reviewer: bug, please fix | coder: `a - b` (buggy) |
| 3 | manager: continue working | reviewer: bug, please fix |
| 4 | **coder: TERMINATE** | coder: `a + b` |
| 5 | | reviewer: approved |
| 6 | | manager: TERMINATE |

**Round-robin ends with the bug unfixed.** `coder_policy` reads only the
*last* non-coder message in its three-message window, and that is the
manager's "continue working", which contains neither "review" nor "fix". The
fix request is still inside the window, but two messages back, so the coder
never reads it.

**Whether round-robin works is decided by team order alone.** Over the 6
orders of the same three agents, 3 ship the bug. They are exactly the orders
where the manager follows the reviewer in the cycle, and the shipped `AGENTS`
dict is one of them. The 3 working orders need 8, 9 and 10 turns to reach
TERMINATE, so under the demo's `max_rounds=8` only one of them finishes. Each
also wastes a turn after approval: the coder re-sends "revised code" because
"review: approved" contains "review".

### 2 — a speak cap either changes nothing or blocks the fix

The rule: if the selector's pick has already spoken `cap` times, the next
agent in team order under the cap speaks instead. When no agent is under the
cap, the selector returns None, and `run_groupchat` already treats None as
the end. It runs over 7 conversations, the LLM-style selector plus the 6
round-robin orders:

| cap | conversations shipping `a + b` |
|---|---:|
| 1 | 0 |
| 2 | 4 |
| 3 | 4 |
| none | 4 |

**At the transcripts' own maximum the cap changes nothing, and below it the
fix never happens.** No agent speaks more than twice in either demo run, so
cap 2 leaves both transcripts identical word for word. Cap 1 ends all 7
conversations after 3 turns with `a - b` shipped, because the fix needs the
coder twice. The task's minimum *is* the imbalance the rule targets, so no
cap balances turns and still lets it finish.

The cap cannot rescue the 3 failing round-robin orders either. They fail on
the coder's second turn, which every cap ≥ 2 allows.

At cap 2, 3 round-robin orders end **by exhaustion**: every agent has spoken
twice, the selector returns None, and the transcript stops on "continue
working" or "revised code" with no TERMINATE. `run_groupchat` cannot tell a
selector's silence from completion.

### 3 — the goal check already exists, one turn late, inside the manager

**It triggers in 4 of 7 conversations, and strictly before the cap in 3.**
The check is a selector wrapper that returns None after a reviewer approval.
Under `max_rounds=8`:

| conversation | without the check | with it |
|---|---:|---:|
| LLM-style | 6 | 5 |
| round-robin, 3 working orders | 8, 9, 10 (2 cut off by the cap) | 6, 7, 8 |
| round-robin, 3 failing orders (incl. the demo's) | 4–6, bug shipped | never fires |

**The manager already is this check.** `manager_policy` returns TERMINATE
exactly when an approval is in the pool, which is the exercise's condition.
The LLM-style selector always gives the manager the turn after an approval,
so the new check saves exactly one turn there: the manager's.

Under round-robin it matters more. It brings the two orders the cap was
cutting off inside it, and removes the coder's wasted post-approval turn. But
its 3 misses are exactly the 3 conversations that ship `a - b`. A termination
rule reads the outcome; it cannot create one.

### 4 — the default selector is an LLM, and it excludes the last speaker

*Draws on "The ConversableAgent API".*

On the linked page, the `GroupChatManager` is a class the reader writes
themselves in the Core API. The page describes it as the agent "which
manages the group chat and selects the next agent to speak using an LLM". The
manager builds a prompt from the participants' role descriptions and the
conversation history, asks the model for a name, matches that name against
the participants, and sends the winner a `RequestToSpeak`.

The detail the lesson leaves out is that the manager also "always picks a
different participant to speak next, by keeping track of the previous
speaker". The previous speaker's topic is filtered out of the candidate list
before the prompt is built. So the default is not "LLM picks anyone"; it is
"LLM picks anyone but the last speaker". That is the same shape as this
module's `llm_style_selector`, which never returns the speaker it just
handed off from.

In the v0.2 line (now AG2 Classic), the equivalent default is
`GroupChat.speaker_selection_method = "auto"`, with `"manual"`, `"random"`,
`"round_robin"` or a callable as the alternatives, and `max_round = 10`.

### 5 — v0.4 adds asynchrony and distribution, not a better selector

*Draws on "Lineage: forks and mergers".*

First, a correction to the question. The AG2 repository no longer contains a
GroupChat to compare. AG2 v1.0 moved the original AutoGen-derived framework,
"GroupChat / swarms, nested- and sequential-chat patterns", into a separate
`ag2-classic` repository. In its place is a `Hub` with a write-ahead log and
agents talking over typed channels. And "v0.4" is Microsoft's AutoGen
rewrite, not an AG2 version.

With that fixed, the comparison is AG2 Classic's GroupChat against AutoGen
v0.4. The v0.2-style `run_chat` loop is sequential: one speaker per round for
`range(max_round)` rounds, with a synchronous selector call in between. v0.4
is described as "an asynchronous, event-driven architecture" in which "agents
communicate through asynchronous messages", plus distributed agent networks
and cross-language (Python and .NET) interoperability.

**The concrete property added is composability across process boundaries.**
An agent becomes an addressable message handler rather than an object in the
manager's list, so it can run elsewhere, in another language, or inside
another team. This is not throughput *within* a group chat: speaker
selection is still one decision per turn, and the Core API's own group-chat
pattern above still waits for one `RequestToSpeak` at a time. Nor is it fault
tolerance in the durable sense; the announcement claims observability
(OpenTelemetry, message tracing), not delivery guarantees. It is AG2's v1.0
Hub, with its write-ahead log, that makes the durability claim.
