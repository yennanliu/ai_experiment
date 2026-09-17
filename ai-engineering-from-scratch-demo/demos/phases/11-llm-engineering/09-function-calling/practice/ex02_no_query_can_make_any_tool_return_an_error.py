"""Exercise 2 — no query can make any tool fail, and the decision function cannot read the feedback.

    **Implement retry with error feedback.** When a tool call fails (e.g., city
    not found), feed the error message back to the model decision function and
    let it correct its arguments. Track how many retries each call takes. Set a
    maximum of 3 retries per tool call.

Reading of the exercise: the retry loop is built exactly as specified -- catch
the error, append it to the history, ask the decision function again, up to 3
retries -- and run over the 30 labelled queries of exercise 4 so the error rate
is measured rather than assumed.

**ANSWER: there is nothing to retry.** Fifteen queries produce 14 tool calls
and 0 of them return an error. The retry path is unreachable from the decision
function, so the exercise's own example -- "city not found" -- cannot occur.

**MECHANISM: the weather branch cannot emit a city that is not in the database.**
It scans `WEATHER_DB` for a substring match, then scans the words for a
capitalised match, and then falls back to `cities = ["tokyo"]`. So "weather in
Atlantis" and "the weather in paris" both return Tokyo's weather -- paris is not
one of the five cities -- with no error and no indication that the question was
not answered.

**MECHANISM: every other branch has the same shape.** `read_file` falls back to
`README.md`, which exists; the calculator falls back to `expression: "0"`, which
evaluates; `web_search` returns `{"results": [], "total": 0}` rather than an
error; `run_code` sends a fixed literal. Each of the five branches ends in a
value that is guaranteed to succeed.

**FINDING: the retry could not use the feedback anyway.**
`simulate_model_decision(user_message, tools, conversation_history)` mentions
`conversation_history` once -- in its own signature. Calling it again with the
error appended returns byte-identical calls, so a 3-retry budget spends three
identical executions and reports 3 retries for a call that never changed.

**CONTROL: the tools are retry-ready; the router is not.** Called directly with
bad arguments, all five return structured errors with codes -- CITY_NOT_FOUND,
NOT_FOUND, FORBIDDEN, SECURITY_VIOLATION and a division-by-zero message. Every
piece the exercise needs exists except a decision function that reads its third
argument.

Structure: `retry` is the loop the exercise describes, `errors_from` counts what
the router can actually provoke, and `DIRECT` is the bad-argument control.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "09-function-calling"
MAX_RETRIES = 3
QUERIES = ["what is the weather in Tokyo", "weather in Atlantis", "temperature in london",
           "the weather in paris", "calculate 12 * 7", "what is 2+2",
           "search for python function calling", "look up weather API", "read README.md",
           "open data/config.json", "cat the config", "run this code",
           "what is the capital of France", "find the config file", "tell me a joke"]
DIRECT = [("get_weather", {"city": "Atlantis"}), ("read_file", {"path": "nope.txt"}),
          ("read_file", {"path": "/etc/passwd"}), ("run_code", {"code": "import os"}),
          ("calculator", {"expression": "1/0"})]


def failed(result):
    return isinstance(result, dict) and bool(result.get("error"))


def retry(ref, query):
    """The loop the exercise asks for: feed the error back, up to 3 retries."""
    history, attempts = [{"role": "user", "content": query}], []
    calls = ref.simulate_model_decision(query, [], history)
    for call in calls:
        for attempt in range(MAX_RETRIES + 1):
            outcome = ref.execute_tool_call(call)
            attempts.append({"tool": call["name"], "attempt": attempt,
                             "error": failed(outcome["result"])})
            if not failed(outcome["result"]):
                break
            history.append({"role": "tool", "content": str(outcome["result"])})
            corrected = ref.simulate_model_decision(query, [], history)
            call = corrected[0] if corrected else call
    return attempts


def errors_from(ref, queries):
    calls = executed = errors = 0
    for query in queries:
        for call in ref.simulate_model_decision(query, [], []):
            calls += 1
            outcome = ref.execute_tool_call(call)
            executed += 1
            errors += failed(outcome["result"])
    return {"calls": calls, "executed": executed, "errors": errors}


def weather_for(ref, city):
    calls = ref.simulate_model_decision(f"the weather in {city}", [], [])
    return calls[0]["arguments"]["city"] if calls else None


def stubborn(ref, query):
    """Does feeding the error back change the call? Ask twice, with and without."""
    plain = ref.simulate_model_decision(query, [], [])
    fed = ref.simulate_model_decision(query, [], [{"role": "tool", "content": "error"}])
    return plain == fed


def solve():
    ref = parity.load_reference(PHASE, LESSON, "function_calling")
    ref.register_all_tools()
    body = inspect.getsource(ref.simulate_model_decision)
    return {
        **errors_from(ref, QUERIES), "queries": len(QUERIES),
        "history_mentions": body.count("conversation_history"),
        "identical": all(stubborn(ref, q) for q in QUERIES),
        "cities": {city: weather_for(ref, city) for city in ("Atlantis", "paris", "london")},
        "fallbacks": {"calculator": ref.simulate_model_decision("what is love", [], []),
                      "read_file": ref.simulate_model_decision("open the door", [], []),
                      "search": ref.execute_tool_call(
                          {"name": "web_search", "arguments": {"query": "zzz"}})["result"]},
        "direct": [ref.execute_tool_call({"name": name, "arguments": args})["result"]
                   for name, args in DIRECT],
        "attempts": retry(ref, "the weather in paris"),
    }


def verify(result):
    direct, cities = result["direct"], result["cities"]
    codes = sorted({r.get("code", "no code") for r in direct})
    return [
        practice.Check(
            "ANSWER: there is nothing to retry",
            all([result["errors"] == 0, result["executed"] == 14]),
            f"{result['queries']} queries produce {result['executed']} tool calls and "
            f"{result['errors']} of them return an error. The retry path is unreachable "
            "from the decision function, so the exercise's own example -- city not found -- "
            "cannot occur",
        ),
        practice.Check(
            "MECHANISM: the weather branch cannot emit a city outside the database",
            all([cities["Atlantis"] == "Tokyo", cities["paris"] == "Tokyo",
                 cities["london"] == "London"]),
            f"asking for the weather in Atlantis, paris and london gives "
            f"{cities}. The branch scans WEATHER_DB for a substring, then the words for a "
            "capitalised match, then falls back to ['tokyo']. paris is not one of the five "
            "cities, so a real question is answered with the wrong city and no error",
        ),
        practice.Check(
            "MECHANISM: every other branch ends in a value guaranteed to succeed",
            all([result["fallbacks"]["calculator"][0]["arguments"]["expression"] == "0",
                 result["fallbacks"]["read_file"][0]["arguments"]["path"] == "README.md",
                 not failed(result["fallbacks"]["search"])]),
            f"the calculator falls back to expression '0', which evaluates; read_file to "
            f"README.md, which exists; web_search returns "
            f"{result['fallbacks']['search']} rather than an error; run_code sends a fixed "
            "literal. Five branches, five guaranteed successes",
        ),
        practice.Check(
            "FINDING: the retry could not use the feedback anyway",
            all([result["history_mentions"] == 1, result["identical"]]),
            f"`simulate_model_decision` mentions conversation_history "
            f"{result['history_mentions']} time -- in its own signature. Calling it again "
            f"with the error appended returns byte-identical calls on all "
            f"{result['queries']} queries, so a {MAX_RETRIES}-retry budget spends three "
            "identical executions on a call that never changed",
        ),
        practice.Check(
            "CONTROL: the tools are retry-ready and the router is not",
            all([len(codes) == 5, all(failed(r) for r in direct)]),
            f"called directly with bad arguments, all five tools return structured errors: "
            f"{codes}. Every piece the exercise needs exists except a decision function "
            f"that reads its third argument -- and the retry loop, run on a query whose "
            f"call cannot fail, records {len(result['attempts'])} attempt and stops",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
