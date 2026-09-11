"""Exercise 3 — the reply is written before the tool runs.

    **Hard.** Add tool calling: implement `get_weather` (any API) and
    `set_timer`. Route the LLM through the tools and verify that when the user
    says "set a 5 minute timer" the right function fires and the spoken reply
    confirms it.

Reading of the exercise: `openai`, `anthropic`, `torch` and `requests` are all
absent, so "any API" is a local table; everything else the exercise asks for is
buildable and, more usefully, the shipped version is measurable against it. The
named test case is the one case that already passes -- `"set a 5 minute timer"`
fires `set_timer(seconds=300)` and the reply says five minutes -- so the exercise
is read as "verify it on more than the one utterance it was fitted to". Eight
probes, each with an intended tool and an intended duration:

| router | right tool | right arguments | distinct replies |
|---|---:|---:|---:|
| shipped `llm_with_tools` | **5 / 8** | **2 / 8** | **2** |
| routed through real tools | **8 / 8** | **8 / 8** | **7** |

**`get_weather` can never fire.** `llm_with_tools` is `if "timer" in transcript`
and nothing else, so both weather requests fall through to no tool call and the
text `'OK.'`. `dispatch_tool` does have a fallback for it -- `{'ok': False}`, a
single key, with no error string for a model to read and no retry, which is the
failure the doc's own Pitfalls section asks you to handle.

**The duration is a constant.** All four timer utterances return
`seconds: 300`, so `"give me a two hour timer"` sets five minutes. The substring
match also fires on `"cancel the timer"` -- a tool call for a request to make one
stop.

**And the spoken reply cannot confirm anything, because it is written first.**
`response["text"]` is one of two literals chosen inside `llm_with_tools`, and
`main()` dispatches the tool *afterwards* and never feeds the result back. The
doc's Step 4 pseudocode has `continue_streaming(result)` for exactly this. As
shipped, a dispatch that returns `{'ok': False}` still leaves the assistant
saying "Sure, setting a 5 minute timer." Deriving the reply from the result
instead makes the failure audible: an unknown location answers **"Sorry, that did
not work: no station for 'mars'."**

Structure: `matches` is a keyword test; `seconds_in` reads a duration out of an
utterance; `route` picks the tool; `dispatch` runs it; `speak` builds the reply
from the result; `score` counts one router against the probe set.
"""

from __future__ import annotations

import importlib.util
import re
import time

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "12-voice-assistant-pipeline"
WORDS = {"one": 1, "two": 2, "three": 3, "five": 5, "ten": 10, "fifteen": 15, "thirty": 30}
UNITS = {"second": 1, "seconds": 1, "minute": 60, "minutes": 60, "hour": 3600, "hours": 3600}
WEATHER = re.compile(r"weather|rain|temperature|forecast")
CANCEL = re.compile(r"cancel|stop|never mind")
STATIONS, STEPS = {"taipei": ("cloudy", 24), "your area": ("clear", 18)}, (
    ("hour", 3600), ("minute", 60), ("second", 1))
DEFAULT_PLACE, DEFAULT_SECONDS, ABSENT = "your area", 300, (
    "openai", "anthropic", "torch", "requests")
PROBES = (("set a 5 minute timer", "set_timer", 300), ("set a ten minute timer", "set_timer", 600),
          ("set a timer for thirty seconds", "set_timer", 30),
          ("give me a two hour timer", "set_timer", 7200), ("cancel the timer", None, None),
          ("what is the weather in taipei", "get_weather", None),
          ("will it rain tomorrow", "get_weather", None),
          ("how many minutes are in an hour", None, None))


def seconds_in(text):
    """The duration an utterance names, as a count-and-unit word pair."""
    words = re.findall(r"[a-z0-9]+", text)
    for count, unit in zip(words, words[1:]):
        number = int(count) if count.isdigit() else WORDS.get(count)
        if number is not None and unit in UNITS:
            return number * UNITS[unit]
    return None


def route(text):
    """(tool, arguments) for one utterance -- the branch `llm_with_tools` does not have."""
    low = text.lower()
    if CANCEL.search(low):
        return None, {}
    if WEATHER.search(low):
        found = re.search(r"\bin ([a-z ]+)$", low)
        return "get_weather", {"location": found.group(1).strip() if found else DEFAULT_PLACE}
    span = seconds_in(low)
    if span or "timer" in low:
        return "set_timer", {"seconds": span or DEFAULT_SECONDS}
    return None, {}


def dispatch(name, args):
    """`dispatch_tool` plus a weather station and an error string a model could read."""
    if name == "set_timer":
        return {"ok": True, "expires_at": time.time() + args["seconds"]}
    if name == "get_weather":
        hit = STATIONS.get(args["location"])
        return ({"ok": True, "sky": hit[0], "celsius": hit[1]} if hit
                else {"ok": False, "error": f"no station for {args['location']!r}"})
    return {"ok": False, "error": f"unknown tool {name!r}"}


def speak(name, args, result):
    """The reply, derived from the tool result rather than chosen before it."""
    if name is None:
        return "OK."
    if not result["ok"]:
        return f"Sorry, that did not work: {result['error']}."
    if name == "get_weather":
        return f"It is {result['sky']} and {result['celsius']} degrees in {args['location']}."
    unit, size = next((u, s) for u, s in STEPS if args["seconds"] >= s)
    return f"Sure, setting a {args['seconds'] // size} {unit} timer."


def score(calls):
    """(right tool, right arguments, distinct replies) for one router over the probes."""
    return (sum(n == want for (_, want, _), (n, _, _) in zip(PROBES, calls)),
            sum(a.get("seconds") == s if want == "set_timer" else n == want
                for (_, want, s), (n, a, _) in zip(PROBES, calls)),
            len({reply for _, _, reply in calls}))


def shipped(ref, text):
    answer = ref.llm_with_tools(text)
    call = answer["tool_calls"][0] if answer["tool_calls"] else {"name": None, "args": {}}
    return call["name"], call["args"], answer["text"]


def routed(text):
    name, args = route(text)
    return name, args, speak(name, args, dispatch(name, args) if name else {"ok": True})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    old, new = [shipped(ref, t) for t, _, _ in PROBES], [routed(t) for t, _, _ in PROBES]
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "shipped": score(old), "routed": score(new), "probes": len(PROBES),
        "named": PROBES[0][0], "named_call": dict(zip(("name", "args", "reply"), old[0])),
        "weather": [call for call, (_, want, _) in zip(old, PROBES) if want == "get_weather"],
        "failure_keys": sorted(ref.dispatch_tool("get_weather", {})),
        "seconds": {args.get("seconds") for _, args, _ in old if args},
        "false_fire": old[4][0], "audible": routed("what is the weather in mars")[2]}


def verify(result):
    old, new = result["shipped"], result["routed"]
    return [
        practice.Check(
            "ANSWER: the utterance the exercise names is the one the router was fitted to",
            result["named_call"]["name"] == "set_timer" and result["seconds"] == {300},
            f"{result['named']!r} fires {result['named_call']['name']}"
            f"({result['named_call']['args']}) and answers {result['named_call']['reply']!r}. "
            f"Over {result['probes']} probes it picks the right tool {old[0]}/"
            f"{result['probes']} times and the right arguments {old[1]}/{result['probes']}",
        ),
        practice.Check(
            "FINDING: no weather branch, a constant duration, and a call on a cancellation",
            all(call[0] is None for call in result["weather"])
            and result["seconds"] == {300} and result["false_fire"] == "set_timer",
            f"`llm_with_tools` is `if 'timer' in transcript` and nothing else: both weather "
            f"requests return no call and 'OK.', all four timer utterances return seconds "
            f"{result['seconds']} so 'give me a two hour timer' sets five minutes, and 'cancel "
            f"the timer' fires {result['false_fire']!r}. `dispatch_tool('get_weather')` answers "
            f"with keys {result['failure_keys']} -- no error string, no retry",
        ),
        practice.Check(
            "MECHANISM: the reply is one of two literals, chosen before the tool runs",
            old[2] == 2 and new[2] > old[2],
            f"`llm_with_tools` picks it and `main()` dispatches afterwards without feeding the "
            f"result back, so {result['probes']} probes draw {old[2]} distinct replies; deriving "
            f"it from the result gives {new[2]}, and an unknown location answers "
            f"{result['audible']!r}",
        ),
        practice.Check(
            "ANSWER: routing through real tools scores 8/8 on both",
            new[0] == new[1] == result["probes"] and len(result["absent"]) == len(ABSENT),
            f"a duration parser, a `get_weather` over a station table -- find_spec is None for "
            f"{result['absent']} -- and a reply built from the result: {new[0]}/"
            f"{result['probes']} tools and {new[1]}/{result['probes']} arguments, against "
            f"{old[0]} and {old[1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
