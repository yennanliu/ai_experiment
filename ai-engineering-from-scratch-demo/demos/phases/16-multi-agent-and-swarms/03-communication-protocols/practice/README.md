<!-- generated:start -->
# 16-multi-agent-and-swarms / 03-communication-protocols

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/03-communication-protocols/) · upstream spec
`phases/16-multi-agent-and-swarms/03-communication-protocols/docs/en.md`

```bash
uv run demo practice run 03-communication-protocols --ex 1
uv run demo explain 03-communication-protocols --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/03-communication-protocols
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Multi-hop task delegation. Extend the `TaskManager` so an agent handler can delegate subtasks… | code | T0 | `ex01_the_task_is_finished_before_anyone_can_subscribe.py` |
| 2 | Streaming audit trail. Modify the `AuditableRunner` to support streaming mode. Instead of wai… | code | T0 | `ex02_there_is_nothing_to_stream_because_nothing_is_appended.py` |
| 3 | DID rotation. Add key rotation to the `IdentityRegistry`. An agent should be able to publish… | code | T0 | `ex03_both_verification_methods_hold_the_same_key.py` |
| 4 | Protocol negotiation. Implement ANP's meta-protocol concept. Two agents exchange `protocolNeg… | code | T0 | `ex04_the_three_round_cap_is_never_reached.py` |
| 5 | Rate-limited discovery. Add a `RateLimitedRegistry` wrapper that caches Agent Card lookups wi… | code | T0 | `ex05_the_cache_removes_the_scans_and_not_the_stampede.py` |
<!-- generated:end -->

## Answers

This lesson ships 744 lines of TypeScript and no Python, so each solution
ports the part it needs and reads the rest of `code/main.ts` as text. One
porting detail matters throughout: an un-awaited `async` function in
JavaScript still runs **synchronously to its first `await`**. Several of the
findings below follow from that single fact, and a port that schedules the
body instead of starting it would hide all of them.

### 1 — the task is finished before anyone can subscribe

**The multi-hop works, and it is now a plain `await`.** The researcher sends
`search` and `summarize`, awaits both together, and merges their artifacts:
three tasks in the manager, researcher `completed`, two merged artifacts.

That is because upstream changed the bug this solution used to report.
[#357](https://github.com/rohitg00/ai-engineering-from-scratch/pull/357) (in
the fork via merge `31f21f1d`) turned the single call site into
`await this.processTask(...)` — called in **1** place, awaited in **1**.
Before, `sendMessage` returned a task already reading `working` while the
handler was still running; now its promise resolves with a task reading
**`completed`**, so a delegating handler needs no polling.

| | `sendMessage` resolves with | listener at first moment hears |
|---|---|---:|
| before #357 (un-awaited) | `working` | 1 of 2 status updates |
| now (awaited) | **`completed`** | **0 of 4** events |

The fix moves the blind spot rather than closing it. The caller still learns
the task id only from what `sendMessage` returns, and that is now after the
task is terminal: `submitted` and `working` are both written and neither is
ever held by a caller, and a subscriber hears **none** of the researcher's
four events (working, two artifacts, completed). The streaming API has no
stream left to observe.

The lesson page did not follow the code. `docs/en.md` still calls
`this.processTask(` in **1** place and awaits it in **0**, so a reader typing
along builds the old, returns-early behaviour.

And a delegated task still cannot be told from a root one. `Task` declares
five fields — `id`, `contextId`, `status`, `artifacts`, `history` — and none
names a parent. `createTask` mints a fresh random `contextId` whenever the
caller does not thread one through, so a subtask that forgets becomes a root,
and the audit trail exercise 2 is about has no edge to record.

### 2 — there is nothing to stream because nothing is appended

"Yield updates in real-time as trajectory entries are added" describes
something the runner never does.

| handler contract | snapshots for a 6-step run | trajectory lengths seen |
|---|---:|---|
| shipped: `Promise<{output, trajectory}>` | **2** | 0 → 6 |
| re-typed: async generator | **8** | 0,1,2,3,4,5,6,6 |

`entry.trajectory` is **assigned** in one place, from `result.trajectory`,
*after* `await handler(input)` has already returned — and **pushed** in one
other, the error path, which records a single synthetic entry. In a successful
run the field goes from empty to complete in one statement. There is no
intermediate state to observe, so the exercise cannot be done inside
`AuditableRunner` at all; the handler contract has to change with it.

Two smaller things a streaming consumer would trip on. `AuditEntry` declares
five statuses and `"awaiting"` appears exactly once in the module — in that
union. No code path assigns it, so a consumer written against the type has a
branch that can never run. And `entry.status = "failed"` appears twice while
`entry.completedAt` is set twice: on success and in the `catch`. A run that
fails because no handler is registered never gets an end time, so a duration
is a number for one kind of failure and `undefined` for the other.

### 3 — both verification methods hold the same key

Rotation is a new document, a `previousDid` edge, and a deadline the verifier
reads. Inside the window a registry that walks the chain accepts **2 of 2**
signatures — retired key and new. After it, **1 of 2**: the old document still
resolves, and the window is a policy rather than a fact about storage.

Now read the document being rotated. `createIdentity` computes `publicKeyDer`
**once**, from the Ed25519 signing key, and writes it into **two** entries:

| id | declared type | key |
|---|---|---|
| `#key-1` | `Ed25519VerificationKey2020` | the signing key |
| `#key-x25519-1` | `X25519KeyAgreementKey2019` | **the same signing key** |

The key-agreement slot holds a signing key and contradicts itself in its own
`type` field. So "publish a new DID document with updated keys" is one
decision wearing two names: rotating changes both at once, and a verifier has
no way to learn which of the two was replaced.

The human-authorization control is dead in the same document.
`humanAuthorization` is built as `[]` in the only constructor and pushed to
**zero** times anywhere, so `requiresHumanAuth` is false for both key ids —
the operation class ANP defines to require a person cannot be expressed by any
document this module produces.

The sharpest one is in the gateway. `delegateTask` verifies the signature over
`message.id`, a `crypto.randomUUID()` minted at construction. Keeping the id
and rewriting the body from *"please summarise the Q3 filing"* to *"wire the
balance to account 9912"* leaves the signature valid. What is signed is the
envelope's serial number.

### 4 — the three-round cap is never reached

Implemented as one candidate each per round. Then swept over every pair of
preference lists across four formats — all 64 orderings of every non-empty
subset, on both sides:

| | pairs | outcome |
|---|---:|---|
| share a format | **3964** | all agree — 3796 in 1 round, 168 in 2 |
| share none | **132** | never agree, at any cap |
| take 3 rounds | **0** | — |

The maximum is **two**. The specification's one free parameter is a cap that
is one higher than the worst case that exists, so moving it to 3 changes the
outcome for zero of 4096 pairs. What separates agreement from timeout is
whether the capability sets intersect — and a one-shot exchange of whole sets
settles all 4096 in a single round *and* reports impossibility immediately
rather than after three silences. It is better on both axes.

The last clause of the exercise — "the agreed format determines which
`TaskManager` or `AuditableRunner` they use" — has nothing to attach to.
`capabilities.streaming` is declared three times and read zero. `delegateTask`
has zero branches on any capability. And it does not choose between the two
runners at all: it calls `taskManager.sendMessage(...)` and then
`auditRunner.run(...)`, both of which execute the target agent. The
port counts **two executions per delegation**, through two registries sharing
no state, returning a `task` and an `audit` that describe different runs of
the same request.

### 5 — the cache removes the scans and not the stampede

A herd of 100 agents each calling `discoverBySkillTag` once:

| | tag comparisons |
|---|---:|
| shipped registry | **10000** |
| TTL cache (1 miss, 99 hits) | **100** |

`[...this.cards.values()].filter(...)` rebuilds and rescans the whole map on
every call, so the wrapper is worth 100×. That is the measurement the exercise
asks for, and it measures the wrong thing.

`discoverAndDelegate` selects `candidates[0]` — no ranking, Map insertion
order. With ten agents carrying the tag, all one hundred callers pick the
**same one**. The rate limiter protects the registry, which was never at risk;
the elected agent takes a hundred delegations either way. Caching a fan-out
does not spread it, and a thundering herd is a problem about concentration.

The TTL buys its reduction with a blind window. An agent that registers one
tick after the first lookup is invisible until the entry expires: over a
30-tick TTL the herd sees **10** candidates while the registry holds **11**,
and the newcomer appears in **zero** cached results. The startup case this
exercise simulates is exactly the case a startup cache gets wrong.

One last inconsistency worth knowing before wrapping either method:
`discoverBySkillTag` reads only `skill.tags`, while `discoverByInputMode`
reads `card.defaultInputModes` **or** `skill.inputModes`. Two methods, two
different notions of what a card offers, and the registry exposes both without
saying which one `discoverAndDelegate` uses.
