"""Exercise 3 — the loop ends its body with an unconditional break, and the query routes once.

    **Build a multi-step agent.** Some queries require chaining tool calls:
    "Read the config file and tell me what model is configured, then search the
    web for that model's pricing." Implement a loop that runs until the model
    decides no more tools are needed, passing accumulated results into each
    decision step. Limit to 10 iterations to prevent infinite loops.

Reading of the exercise: the loop already exists as
`run_function_calling_loop(user_message, max_iterations=5)`, so the exercise is
read as *make it iterate*, and the chaining query is the exercise's own. The
agent built here passes accumulated results into the next decision, which is
the one thing the shipped loop does not do.

**ANSWER: the shipped loop cannot iterate.** Its body ends with an
unconditional `break`, so `max_iterations` is dead and `iterations` is 1 for any
query that produces a call and 0 for any that does not -- never 2, whatever the
query. Removing the break does not help either: the decision is re-asked with
`user_message`, not with the conversation, so it returns the identical calls
until the cap.

**ANSWER: the exercise's own chaining query routes to one tool, and the wrong
one.** "Read the config file and tell me what model is configured, then search
the web for that model's pricing" hits the `search`/`find` branch before the
`read`/`file` branch, because the if-chain returns on first match. It calls
`web_search` with the whole sentence as the query, gets 0 results, and the
config file is never read.

**FINDING: a working chain needs one decision per clause, plus the result.** The
if-chain returns on first match, so a single sentence can reach only one branch.
Splitting the query on "then" and substituting the previous result into the next
clause gives `read_file` on `data/config.json`, which yields
`"model": "gpt-4o"`, and then `web_search` for that model. Two steps, then a
stop.

**FINDING: the 10-iteration cap the exercise asks for is the only thing standing
between the loop and a non-terminating run.** With the break removed and the
decision left as shipped, the loop runs the full cap on every query that
produces a call -- 10 identical `web_search` executions for the chaining query.

**FINDING: `iterations` is reported from the loop variable.**
`"iterations": iteration + 1 if tool_calls else 0` reads the `for` index, which
the break pins at 0. The field cannot distinguish a one-step answer from a
ten-step one, which is exactly what the exercise asks to measure.

Structure: `unbroken` is the shipped loop with the break removed, `chained` the
agent that decomposes and feeds results forward, `follow_up` the one field it
carries between steps, and `CHAIN` the exercise's own query.
"""

from __future__ import annotations

import inspect
import json

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "09-function-calling"
CHAIN = ("Read the config file and tell me what model is configured, "
         "then search the web for that model's pricing")
CAP = 10


def unbroken(ref, message, cap=CAP):
    """The shipped loop with the trailing `break` removed and nothing else changed."""
    executed = []
    for _ in range(cap):
        calls = ref.simulate_model_decision(message, [], [])
        if not calls:
            break
        executed += [ref.execute_tool_call(call)["tool"] for call in calls]
    return executed


def chained(ref, message, cap=CAP):
    """The agent the exercise asks for: one clause per step, results fed forward.

    The if-chain returns on first match, so a single sentence can only reach one
    branch. Splitting on "then" gives each clause its own decision, and the
    previous result is substituted into the next clause.
    """
    clauses = [c.strip() for c in message.split("then") if c.strip()]
    steps, carried = [], ""
    for clause in clauses[:cap]:
        calls = ref.simulate_model_decision(f"{clause} {carried}".strip(), [], [])
        if not calls:
            break
        outcome = ref.execute_tool_call(calls[0])
        steps.append({"tool": calls[0]["name"], "args": calls[0]["arguments"],
                      "result": outcome["result"]})
        carried = follow_up(outcome["result"]) or carried
    return steps


def follow_up(result):
    """What the next step should look for: the model name the config file names."""
    content = result.get("content", "") if isinstance(result, dict) else ""
    try:
        return json.loads(content).get("model", "")
    except (ValueError, TypeError):
        return ""


def solve():
    ref = parity.load_reference(PHASE, LESSON, "function_calling")
    ref.register_all_tools()
    source = inspect.getsource(ref.run_function_calling_loop)
    shipped = ref.run_function_calling_loop(CHAIN, max_iterations=CAP)
    steps = chained(ref, CHAIN)
    return {
        "trailing_break": "        break\n\n    return" in source,
        "reasks_with_message": "simulate_model_decision(user_message" in source,
        "iterations": shipped["iterations"],
        "shipped_tools": [r["tool"] for r in shipped["tool_results"]],
        "shipped_results": [r["result"] for r in shipped["tool_results"]],
        "no_call_iterations": ref.run_function_calling_loop("hello there", CAP)["iterations"],
        "unbroken": len(unbroken(ref, CHAIN)),
        "unbroken_tools": sorted(set(unbroken(ref, CHAIN))),
        "chain_tools": [s["tool"] for s in steps],
        "chain_model": follow_up(steps[0]["result"]) if steps else "",
        "chain_query": steps[1]["args"]["query"] if len(steps) > 1 else "",
        "cap": CAP,
    }


def verify(result):
    shipped = result["shipped_results"]
    return [
        practice.Check(
            "ANSWER: the shipped loop cannot iterate",
            all([result["trailing_break"], result["reasks_with_message"],
                 result["iterations"] == 1, result["no_call_iterations"] == 0]),
            f"the body ends with an unconditional break, so `iterations` is "
            f"{result['iterations']} for a query that produces a call and "
            f"{result['no_call_iterations']} for one that does not -- never 2, whatever the "
            "query. And the decision is re-asked with `user_message` rather than the "
            "conversation, so removing the break would not help either",
        ),
        practice.Check(
            "ANSWER: the chaining query routes to one tool, and the wrong one",
            all([result["shipped_tools"] == ["web_search"], shipped[0]["total"] == 0]),
            f"the exercise's own query hits the search/find branch before the read/file "
            f"branch, because the if-chain returns on first match: it calls "
            f"{result['shipped_tools']} with the whole sentence and gets "
            f"{shipped[0]['total']} results. The config file is never read",
        ),
        practice.Check(
            "FINDING: a working chain needs 2 iterations, fed by the results",
            all([result["chain_tools"] == ["read_file", "web_search"],
                 result["chain_model"] == "gpt-4o"]),
            f"one decision per clause, with the previous result substituted into the next, "
            f"gives {result['chain_tools']} -- the config file yielding model "
            f"{result['chain_model']!r} and the search then running on "
            f"{result['chain_query']!r}. Two steps and then a stop",
        ),
        practice.Check(
            "FINDING: the cap is the only thing between the loop and a non-terminating run",
            all([result["unbroken"] == result["cap"],
                 result["unbroken_tools"] == ["web_search"]]),
            f"with the break removed and the decision left as shipped, the loop runs the "
            f"full cap: {result['unbroken']} executions of "
            f"{result['unbroken_tools']} for the chaining query. The 10-iteration limit the "
            "exercise asks for is load-bearing, not a safety margin",
        ),
        practice.Check(
            "FINDING: `iterations` is reported from the loop variable",
            all([result["iterations"] == 1, result["no_call_iterations"] == 0]),
            "`\"iterations\": iteration + 1 if tool_calls else 0` reads the `for` index, "
            "which the break pins at 0. The field takes two values and cannot distinguish "
            "a one-step answer from a ten-step one -- which is what the exercise asks to "
            "measure",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
