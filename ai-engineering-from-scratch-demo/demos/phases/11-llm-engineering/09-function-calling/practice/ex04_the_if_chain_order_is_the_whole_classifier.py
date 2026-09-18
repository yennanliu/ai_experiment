"""Exercise 4 — the if-chain order is the whole classifier, and 25 of 30 is its ceiling.

    **Measure tool selection accuracy.** Create 30 test queries with expected
    tool names. Run your decision function on all 30 and measure what
    percentage of the time it selects the correct tool. Identify which queries
    cause the most confusion between tools.

Reading of the exercise: the 30 queries are labelled with the tool that should
answer them, including 8 that should select no tool, because `return []` is a
reachable outcome and a classifier that cannot decline is not measurable.
Accuracy is exact match on the first selected tool.

**ANSWER: 25 of 30, 83.3%.** Five queries are wrong, and all five for the same
reason: `simulate_model_decision` is an if-chain that returns on the
first branch whose keyword appears anywhere in the message.

**FINDING: the confusions are a function of branch order, not of similarity.**
Two `web_search` queries go to `get_weather` -- "look up weather API" and
"google the weather API" -- because the weather branch is first and "weather" is
one of its keywords. Two queries that should select nothing go to `calculator`,
because "what is" and "how much" are its keywords. And "find the config file",
which should read the file, goes to `web_search` because search is tested
before read.

**FINDING: the calculator cannot decline.** Its branch ends
`return [{"name": "calculator", "arguments": {"expression": "0"}}]`, so "what is
the capital of France" produces a tool call that succeeds and returns 0. A
question with no arithmetic in it is answered with a number.

**MECHANISM: there are 21 keywords and no scoring.** The five branches test 21
substrings in a fixed order, and the first hit wins outright -- a message
containing keywords from three branches is routed by which branch was written
first, not by how many of each it contains.

**CONTROL: make a branch bind before it fires.** Requiring a city from
`WEATHER_DB`, an arithmetic expression, a known path or a `SEARCH_DB` key before
the branch wins takes accuracy from 25 to 30 of 30, with the same keywords in
the same order -- the ordering was never the problem, the unconditional
fallbacks were. One more fix is needed to get there: the lesson's own path
matcher compares `"README"` against a lowercased message, so `README.md` is
reachable only through the fallback.

Structure: `LABELLED` is the 30-query set, `scored` is the control classifier,
and `confusions` is the (expected, selected) tally.
"""

from __future__ import annotations

import collections

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "09-function-calling"
KEYWORDS = {
    "get_weather": ("weather", "temperature", "forecast"),
    "calculator": ("calculate", "compute", "math", "what is", "how much"),
    "web_search": ("search", "find", "look up", "google"),
    "read_file": ("read", "file", "open", "cat", "show"),
    "run_code": ("run", "execute", "code", "python"),
}
LABELLED = [
    ("what is the weather in Tokyo", "get_weather"), ("weather in new york", "get_weather"),
    ("temperature in london", "get_weather"), ("forecast for sydney", "get_weather"),
    ("what is the forecast in san francisco", "get_weather"),
    ("calculate 12 * 7", "calculator"), ("compute 45/9", "calculator"),
    ("what is 2+2", "calculator"), ("how much is 100-37", "calculator"),
    ("math: 6*6", "calculator"),
    ("search for python function calling", "web_search"),
    ("find the MCP protocol", "web_search"), ("look up weather API", "web_search"),
    ("google the weather API", "web_search"),
    ("read README.md", "read_file"), ("open data/config.json", "read_file"),
    ("show me data/users.csv", "read_file"), ("cat the config", "read_file"),
    ("read the users file", "read_file"),
    ("run this code", "run_code"), ("execute the script", "run_code"),
    ("python snippet", "run_code"),
    ("what is the capital of France", None), ("find the config file", "read_file"),
    ("how much does it cost", None), ("tell me a joke", None),
    ("who won the world cup", None), ("explain quantum tunnelling", None),
    ("summarise this document", None), ("translate hello", None),
]


def selected(ref, query):
    calls = ref.simulate_model_decision(query, [], [])
    return calls[0]["name"] if calls else None


def binders(ref):
    """Could a branch fill its arguments from the message? One test per branch."""
    return {
        "get_weather": lambda m: any(city in m for city in ref.WEATHER_DB),
        "calculator": lambda m: any(c in m for c in "+-*/") and any(c.isdigit() for c in m),
        "read_file": lambda m: any(p.split("/")[-1].split(".")[0].lower() in m
                                   for p in ref.FILE_SYSTEM),
        "web_search": lambda m: any(k.lower() in m for k in ref.SEARCH_DB),
        "run_code": lambda m: True,
    }


def scored(ref, query, bind=None):
    """The control: a branch fires only when it can bind an argument from the message."""
    message, bind = query.lower(), bind or binders(ref)
    for name, words in KEYWORDS.items():
        if any(word in message for word in words) and bind[name](message):
            return name
    return None


def accuracy(picks):
    return sum(pick == want for pick, (_, want) in zip(picks, LABELLED))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "function_calling")
    ref.register_all_tools()
    picks = [selected(ref, query) for query, _ in LABELLED]
    bind = binders(ref)
    control = [scored(ref, query, bind) for query, _ in LABELLED]
    capital = ref.simulate_model_decision("what is the capital of France", [], [])
    return {
        "queries": len(LABELLED), "accuracy": accuracy(picks),
        "control_accuracy": accuracy(control), **failures(picks),
        "declined": sum(pick is None for pick in picks),
        "should_decline": sum(want is None for _, want in LABELLED),
        "keywords": sum(len(words) for words in KEYWORDS.values()),
        "branches": len(KEYWORDS), "capital": capital,
        "capital_result": ref.execute_tool_call(capital[0])["result"],
    }


def failures(picks):
    wrong = [(q, want, pick) for (q, want), pick in zip(LABELLED, picks) if pick != want]
    return {"confusions": collections.Counter((want, pick) for _, want, pick in wrong),
            "wrong": [(q[:30], want, pick) for q, want, pick in wrong]}


def verify(result):
    confusions = result["confusions"]
    return [
        practice.Check(
            "ANSWER: 25 of 30, 83.3%",
            all([result["accuracy"] == 25, result["queries"] == 30]),
            f"{result['accuracy']} of {result['queries']} queries select the expected tool, "
            f"{100 * result['accuracy'] / result['queries']:.1f}%. The "
            f"{result['queries'] - result['accuracy']} failures are "
            f"{result['wrong']}",
        ),
        practice.Check(
            "FINDING: the confusions are a function of branch order",
            all([confusions[("web_search", "get_weather")] == 2,
                 confusions[(None, "calculator")] == 2,
                 confusions[("read_file", "web_search")] == 1]),
            f"the (expected, selected) tally is {dict(confusions)}. Two web_search queries "
            "go to get_weather because the weather branch is first and 'weather' is one of "
            "its keywords; two that should decline go to calculator because 'what is' and "
            "'how much' are its keywords; and 'find the config file' goes to web_search "
            "because search is tested before read. Nothing here is about similarity",
        ),
        practice.Check(
            "FINDING: the calculator cannot decline",
            all([result["capital"][0]["arguments"]["expression"] == "0",
                 result["capital_result"].get("result") == 0]),
            f"the calculator branch ends `return [{{'name': 'calculator', 'arguments': "
            f"{{'expression': '0'}}}}]`, so 'what is the capital of France' produces "
            f"{result['capital'][0]['arguments']} and the tool returns "
            f"{result['capital_result']}. A question with no arithmetic is answered with a "
            "number, successfully",
        ),
        practice.Check(
            "MECHANISM: 21 keywords, five branches, no scoring",
            all([result["keywords"] == 21, result["branches"] == 5,
                 result["declined"] < result["should_decline"]]),
            f"the five branches test {result['keywords']} substrings in a fixed order and "
            f"the first hit wins outright. {result['declined']} of the "
            f"{result['should_decline']} queries that should decline do; a message carrying "
            "keywords from three branches is routed by which branch was written first",
        ),
        practice.Check(
            "CONTROL: score the branches instead of ordering them",
            all([result["control_accuracy"] == result["queries"],
                 result["control_accuracy"] > result["accuracy"]]),
            f"requiring a branch to bind an argument before it fires -- a city in "
            f"WEATHER_DB, an arithmetic expression, a known path, a SEARCH_DB key -- gives "
            f"{result['control_accuracy']} of {result['queries']} against "
            f"{result['accuracy']}, with the same keywords in the same order. The lesson's "
            "own path matcher needs one more fix to get there: it compares 'README' against "
            "a lowercased message, so README.md is only ever reached by the fallback",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
