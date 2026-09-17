<!-- generated:start -->
# 11-llm-engineering / 09-function-calling

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/09-function-calling/) · upstream spec
`phases/11-llm-engineering/09-function-calling/docs/en.md`

```bash
uv run demo practice run 09-function-calling --ex 1
uv run demo explain 09-function-calling --ex 1
uv run pytest demos/phases/11-llm-engineering/09-function-calling
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a 6th tool: database query. Implement a simulated SQL tool with an in-memory table. The t… | code | T0 | `ex01_the_sixth_tool_is_registered_and_unreachable.py` |
| 2 | Implement retry with error feedback. When a tool call fails (e.g., city not found), feed the… | code | T0 | `ex02_no_query_can_make_any_tool_return_an_error.py` |
| 3 | Build a multi-step agent. Some queries require chaining tool calls: "Read the config file and… | code | T0 | `ex03_the_loop_ends_its_body_with_an_unconditional_break.py` |
| 4 | Measure tool selection accuracy. Create 30 test queries with expected tool names. Run your de… | code | T0 | `ex04_the_if_chain_order_is_the_whole_classifier.py` |
| 5 | Implement tool call caching. If the same tool is called with identical arguments within 60 se… | code | T0 | `ex05_the_key_the_exercise_specifies_cannot_hold_the_tool_it_asks_for.py` |
<!-- generated:end -->

## Answers

The lesson is pure stdlib, so all five exercises are **T0** and run in CI. The
five tools are well built — each validates its own inputs and returns a
structured error with a code. What every exercise runs into is the seam above
them: `simulate_model_decision` is a fixed if-chain over 21 keyword substrings,
it ignores both of its other two parameters, and every branch ends in a fallback
that is guaranteed to succeed.

### 1 — the sixth tool registers, and can never be called

```text
TOOL_REGISTRY            5  ->  6
query_db("users", [["age", ">", 30]])   ->  2 rows
database-shaped queries routed to it    ->  0 of 8
```

**ANSWER: registered, described, exposed, unreachable.** `run_function_calling_loop`
rebuilds `tool_definitions` from the registry on every call — and hands all six
to a decision function that never reads its `tools` parameter.

**FINDING: the execution path never validates.** `validate_tool_arguments` is
called **once** in the whole module, inside `run_demo`; `execute_tool_call` runs
`func(**args)` directly. A tool that does not check its own arguments is not
checked.

**FINDING: the schema expresses one allowlist and not the other.**

| invalid payload | validator | function body |
|---|---|---|
| `table: "secrets"` | ❌ caught (enum) | ❌ `TABLE` |
| `table: "users; DROP TABLE users"` | ❌ caught (enum) | ❌ `TABLE` |
| `filters: [["age", "LIKE", 30]]` | ✅ passes | ❌ `OPERATOR` |
| `filters: [["age", ";DROP", 30]]` | ✅ passes | ❌ `OPERATOR` |
| `filters: [{"anything": 1}]` | ✅ passes | ❌ `FILTER` |
| `filters: [["age", ">"]]` | ✅ passes | ❌ `FILTER` |

`filters: []` and `filters: [{"anything": 1}]` are both just arrays to a
validator that reads `type`, `required` and `enum`.

**CONTROL: one branch** — `("table", "query", "rows", "where")` at the front of
the chain — routes **8 of 8**.

### 2 — no query can make any tool return an error

```text
15 queries  ->  14 tool calls  ->  0 errors
```

**ANSWER: the retry path is unreachable from the decision function.**

**MECHANISM: every branch ends in a value guaranteed to succeed.**

| query | selected argument | why it cannot fail |
|---|---|---|
| "the weather in Atlantis" | `city: "Tokyo"` | fallback `cities = ["tokyo"]` |
| "the weather in paris" | `city: "Tokyo"` | paris is not one of the five cities |
| "what is love" | `expression: "0"` | evaluates to 0.0 |
| "open the door" | `path: "README.md"` | exists |
| any search miss | — | returns `{"results": [], "total": 0}` |

**FINDING: the retry could not use the feedback anyway.**
`simulate_model_decision(user_message, tools, conversation_history)` mentions
`conversation_history` exactly once — in its own signature. Calling it again
with the error appended returns byte-identical calls on all 15 queries, so a
3-retry budget spends three identical executions.

**CONTROL: the tools are retry-ready.** Called directly with bad arguments they
return `CITY_NOT_FOUND`, `NOT_FOUND`, `FORBIDDEN`, `SECURITY_VIOLATION` and a
division-by-zero message. Every piece exists except a decision function that
reads its third argument.

### 3 — the loop ends its body with an unconditional break

```python
        all_tool_results.extend(results)
        break                                  # <- ends the for body, always

    return {..., "iterations": iteration + 1 if tool_calls else 0}
```

**ANSWER: `max_iterations` is dead and `iterations` takes two values** — 1 for a
query that produces a call, 0 for one that does not. Removing the break does not
help: the decision is re-asked with `user_message`, not with the conversation.

**ANSWER: the exercise's own chaining query routes to one tool, and the wrong
one.** "Read the config file … then search the web …" hits the `search`/`find`
branch before `read`/`file`, calls `web_search` with the whole sentence, gets
**0** results, and never reads the config file.

**FINDING: a working chain needs one decision per clause plus the result.**
Splitting on "then" and substituting the previous result gives:

```text
step 1   read_file    data/config.json   ->  {"model": "gpt-4o", ...}
step 2   web_search   "gpt-4o"           ->  the pricing search
```

**FINDING: the 10-iteration cap is load-bearing.** With the break removed and
the decision unchanged, the loop runs the full cap — 10 identical `web_search`
executions.

### 4 — the if-chain order is the whole classifier

**ANSWER: 25 of 30, 83.3%.**

| expected | selected | count | why |
|---|---|---:|---|
| `web_search` | `get_weather` | 2 | "weather" is a weather keyword and that branch is first |
| *(none)* | `calculator` | 2 | "what is" and "how much" are calculator keywords |
| `read_file` | `web_search` | 1 | "find" is a search keyword and search is tested first |

**FINDING: the calculator cannot decline.**

```python
return [{"name": "calculator", "arguments": {"expression": "0"}}]
```

"what is the capital of France" → a tool call that **succeeds** and returns
`{"result": 0.0, "expression": "0"}`.

**MECHANISM: 21 keywords, 5 branches, no scoring.** First hit wins outright; 5
of the 7 queries that should decline do.

**CONTROL: make a branch bind before it fires** — a city in `WEATHER_DB`, an
arithmetic expression, a known path, a `SEARCH_DB` key — and accuracy goes to
**30 of 30** with the same keywords in the same order. The ordering was never
the problem; the unconditional fallbacks were. (One more fix is needed: the
lesson compares `"README"` against a lowercased message, so `README.md` is
reachable only through the fallback.)

### 5 — the cache key cannot hold the tool exercise 1 asks for

```text
20 calls, 5 hits, 25%, 15 distinct keys
  of the 5 hits:  2 x the Tokyo fallback, README.md, the fixed run_code literal
```

**ANSWER: the hit rate measures the decision function.** Every hit is a query
the router collapsed onto an argument it had already produced — not a user
asking the same thing twice.

**FINDING: the specified key raises on a database tool.**

```python
frozenset({"table": "users", "filters": [["age", ">", 30]]}.items())
# TypeError: unhashable type: 'list'
```

Exercise 1 asks for a tool whose arguments exercise 5's key cannot hold.

**FINDING: the 60-second window is never tested by its own fixture.** All 20
queries run in under a second. Driving the same conversation through an injected
clock at +61 s per query takes the hits **5 → 0**: the branch works and the
fixture cannot reach it.

**FINDING: the key cannot tell which tools are safe to cache.** Read
`README.md`, mutate `FILE_SYSTEM`, read again — you get the stale content.
`run_code` has the same shape: arbitrary code behind an argument key.

**CONTROL: `json.dumps(args, sort_keys=True)`** gives the same 5 hits over the
same 15 keys, and additionally keys the database tool without raising.
