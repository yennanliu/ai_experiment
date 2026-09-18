"""Exercise 5 — the cache key cannot hold exercise 1's tool, and the hit rate measures the router.

    **Implement tool call caching.** If the same tool is called with identical
    arguments within 60 seconds, return the cached result instead of
    re-executing. Use a dictionary keyed by
    `(tool_name, frozenset(args.items()))`. Measure cache hit rates across a
    conversation with 20 queries.

Reading of the exercise: the key is the one the exercise specifies, verbatim,
and the 20 queries are the ones exercise 4 labels, run through
`simulate_model_decision` so the workload is the lesson's own. The 60-second
window is exercised by injecting a clock rather than by sleeping.

**ANSWER: 5 hits in 20 calls, 25%, over 15 distinct keys.** And the hit rate is
a measurement of the decision function: 2 of the 5 hits are the Tokyo fallback
and the rest are the `README.md` fallback and the fixed `run_code` literal.
Every one is a query the router collapsed onto an argument it had already
produced, not a user asking the same thing twice.

**FINDING: the specified key raises on the tool exercise 1 asks you to build.**
`frozenset(args.items())` requires every argument value to be hashable. A
database tool takes filter conditions, which are lists, and
`frozenset({"table": "users", "filters": [["age", ">", 30]]}.items())` raises
`TypeError: unhashable type: 'list'`. The two exercises are incompatible as
written.

**FINDING: the window is never tested by its own fixture.** All 20 queries run
in well under a second, so no entry can expire and the TTL branch is never
taken. Driving the same conversation through an injected clock shows the branch
works -- at +61 seconds every key misses and the hit count falls from 5 to 0.

**FINDING: two of the five tools are unsafe to cache and the key cannot tell.**
`run_code` executes arbitrary code and `read_file` reads a mutable store; the
key is only the arguments, so a cached `read_file` result survives a change to
`FILE_SYSTEM`. Demonstrated: mutate the file, ask again, get the old content.

**CONTROL: a JSON-canonical key takes both.**
`json.dumps(args, sort_keys=True)` accepts lists and dicts, gives the same 5
hits on the same 20 queries, and additionally keys exercise 1's database tool
without raising.

Structure: `frozen_key` and `json_key` are the two keys, `Cache` is the TTL
dictionary with an injected clock, and `run` drives the 20-query conversation.
"""

from __future__ import annotations

import json

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "09-function-calling"
TTL = 60
QUERIES = ["what is the weather in Tokyo", "weather in new york", "temperature in london",
           "forecast for sydney", "what is the weather in paris", "calculate 12 * 7",
           "compute 45/9", "what is 2+2", "how much is 100-37", "math: 6*6",
           "search for python function calling", "find the MCP protocol",
           "look up weather API", "read README.md", "open data/config.json",
           "cat the config", "read the users file", "run this code",
           "execute the script", "python snippet"]
DB_ARGS = {"table": "users", "filters": [["age", ">", 30]]}


def frozen_key(name, args):
    """The key the exercise specifies, verbatim."""
    return name, frozenset(args.items())


def json_key(name, args):
    """The control: canonical JSON, which accepts lists and dicts."""
    return name, json.dumps(args, sort_keys=True)


class Cache:
    """A TTL dictionary with an injected clock, so the window can be tested."""

    def __init__(self, key, ttl=TTL):
        self.key, self.ttl, self.store, self.hits, self.calls = key, ttl, {}, 0, 0

    def get(self, ref, call, now):
        self.calls += 1
        entry = self.key(call["name"], call["arguments"])
        stored = self.store.get(entry)
        if stored and now - stored[0] < self.ttl:
            self.hits += 1
            return stored[1]
        result = ref.execute_tool_call(call)
        self.store[entry] = (now, result)
        return result


def run(ref, key, clock=lambda i: 0.0):
    cache, hit_names = Cache(key), []
    for index, query in enumerate(QUERIES):
        for call in ref.simulate_model_decision(query, [], []):
            before = cache.hits
            cache.get(ref, call, clock(index))
            if cache.hits > before:
                hit_names.append((call["name"], tuple(sorted(call["arguments"].items()))))
    return {"calls": cache.calls, "hits": cache.hits, "keys": len(cache.store),
            "hit_names": hit_names}


def stale_read(ref, key):
    """Cache a file read, change the file, ask again."""
    cache = Cache(key)
    call = {"name": "read_file", "arguments": {"path": "README.md"}}
    first = cache.get(ref, call, 0.0)["result"]["content"]
    original = ref.FILE_SYSTEM["README.md"]
    try:
        ref.FILE_SYSTEM["README.md"] = "# Changed"
        second = cache.get(ref, call, 1.0)["result"]["content"]
    finally:
        ref.FILE_SYSTEM["README.md"] = original
    return first == second, second


def solve():
    ref = parity.load_reference(PHASE, LESSON, "function_calling")
    ref.register_all_tools()
    frozen, canonical = run(ref, frozen_key), run(ref, json_key)
    expired = run(ref, frozen_key, clock=lambda i: i * (TTL + 1))
    try:
        frozen_key("query_db", DB_ARGS)
        raised = None
    except TypeError as error:
        raised = type(error).__name__
    stale, content = stale_read(ref, frozen_key)
    return {
        "queries": len(QUERIES), **frozen,
        "rate": round(frozen["hits"] / frozen["calls"], 3),
        "canonical_hits": canonical["hits"], "canonical_keys": canonical["keys"],
        "expired_hits": expired["hits"],
        "tokyo_hits": sum(1 for name, args in frozen["hit_names"]
                          if name == "get_weather" and ("city", "Tokyo") in args),
        "raised": raised, "json_ok": json_key("query_db", DB_ARGS)[1],
        "stale": stale, "stale_content": content,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 5 hits in 20 calls, and they are the router's fallbacks",
            all([result["hits"] == 5, result["calls"] == 20, result["keys"] == 15,
                 result["tokyo_hits"] == 2]),
            f"{result['hits']} hits in {result['calls']} calls, {result['rate']:.0%}, over "
            f"{result['keys']} distinct keys. {result['tokyo_hits']} of the hits are the "
            "Tokyo fallback -- queries the router collapsed onto an argument it had already "
            "produced -- so the hit rate measures the decision function, not the workload",
        ),
        practice.Check(
            "FINDING: the specified key raises on the tool exercise 1 asks you to build",
            all([result["raised"] == "TypeError", result["json_ok"]]),
            f"`frozenset(args.items())` needs every argument value hashable, and a database "
            f"tool takes filter conditions, which are lists: frozenset over {DB_ARGS} raises "
            f"{result['raised']}. The canonical-JSON key takes the same arguments and "
            f"returns {result['json_ok']!r}. The two exercises are incompatible as written",
        ),
        practice.Check(
            "FINDING: the 60-second window is never tested by its own fixture",
            all([result["expired_hits"] == 0, result["hits"] > 0]),
            f"all {result['queries']} queries run in well under a second, so no entry can "
            f"expire and the TTL branch is never taken. Driving the same conversation "
            f"through an injected clock at +{TTL + 1}s per query takes the hits "
            f"{result['hits']} -> {result['expired_hits']}: the branch works and the "
            "fixture cannot reach it",
        ),
        practice.Check(
            "FINDING: the key cannot tell which tools are safe to cache",
            all([result["stale"], result["stale_content"] != "# Changed"]),
            f"the key is only the tool name and the arguments, so a cached `read_file` "
            f"survives a change to FILE_SYSTEM: reading README.md, mutating it, and reading "
            f"again returns {result['stale_content']!r} rather than the new content. "
            "`run_code` has the same shape -- arbitrary code behind an argument key",
        ),
        practice.Check(
            "CONTROL: a JSON-canonical key takes both",
            all([result["canonical_hits"] == result["hits"],
                 result["canonical_keys"] == result["keys"]]),
            f"json.dumps(args, sort_keys=True) gives {result['canonical_hits']} hits over "
            f"{result['canonical_keys']} keys -- identical to the specified key on this "
            "workload -- and additionally keys the database tool without raising. One line, "
            "and exercise 1's tool becomes cacheable",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
