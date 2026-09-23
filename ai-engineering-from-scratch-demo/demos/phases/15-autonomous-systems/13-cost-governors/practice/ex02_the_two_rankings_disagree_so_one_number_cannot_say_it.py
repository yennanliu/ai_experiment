"""Exercise 2 — the two rankings disagree, so one number cannot say it.

    Design a per-tool cap set for a browser agent (Lesson 11). Which tool
    needs the tightest cap? Which tool can run unbounded without risk?

Reading of the exercise: both questions presume a single ordering of tools by
risk. Ranking the same four tools by dollars and by irreversibility produces
two orders that are near-reverses of each other, so the design is a cap set
with two denominations rather than four numbers on one scale.

**ANSWER: tightest on the consequential write; nothing runs unbounded.** The
write gets **1** call per run behind a human tap, because its risk is not
denominated in money at all. Reading gets a dollar cap -- at **10000** tokens
a page it is **$0.0300** a call, **50.0x** a click -- and clicks and typing
get count caps in the hundreds. No tool is free: the read is both the money
and, from Lesson 11, the injection vector.

**FINDING: the two rankings are near-reverses.** By dollars the order is
read, write, click, type; by irreversibility it is write, type, click, read.
The tool at the top of one list is at the bottom of the other, which is why a
"per-tool cap" expressed as one number per tool cannot state the policy --
the caps have different units and different owners.

**FINDING: the shipped governor has no per-tool anything.** `Governor`
carries **12** fields and **0** name a tool. The lesson's own stack lists
**12** control types and the simulator implements **5** of them, with the
per-tool cap -- item **4**, and the fix in the $1,200 case study -- not among
them.

**FINDING: a page read sits exactly on the request cap.** **10000** tokens
is `max_tokens_per_request` to the token, so the cap truncates the first page
the agent opens. A truncated page is a page half-read, which is a correctness
failure the cost layer records as a success -- the one place these two
concerns are not separable, and the one the simulator cannot see.

Structure: `TOOLS` is the cap set; `by_dollars()` and `by_risk()` are the two
orderings it has to satisfy.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "13-cost-governors"

# (tool, tokens per call, reversible without a third party, proposed cap)
TOOLS = (
    ("read page", 10_000, True, "dollar cap"),
    ("click", 200, True, "count cap"),
    ("type into form", 200, True, "count cap"),
    ("submit or pay", 500, False, "1 per run, human tap"),
)
IRREVERSIBILITY = {"submit or pay": 3, "type into form": 2, "click": 1, "read page": 0}


def dollars(ref, tokens):
    return round(tokens / 1000.0 * ref.DOLLARS_PER_KTOK, 4)


def by_dollars(ref):
    return [name for name, tokens, *_rest in
            sorted(TOOLS, key=lambda row: -dollars(ref, row[1]))]


def by_risk():
    return [name for name, *_rest in
            sorted(TOOLS, key=lambda row: -IRREVERSIBILITY[row[0]])]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    governor = ref.Governor()
    read, click = dollars(ref, TOOLS[0][1]), dollars(ref, TOOLS[1][1])
    money, risk = by_dollars(ref), by_risk()
    return {
        "tools": len(TOOLS),
        "tightest": [name for name, _t, reversible, _c in TOOLS if not reversible],
        "unbounded": [name for name, _t, _r, cap in TOOLS if cap == "unbounded"],
        "read_dollars": read,
        "click_dollars": click,
        "read_over_click": round(read / click, 1),
        "by_dollars": money,
        "by_risk": risk,
        "top_of_money_is_bottom_of_risk": money[0] == risk[-1],
        "governor_fields": len(ref.Governor.__dataclass_fields__),
        "tool_fields": [name for name in ref.Governor.__dataclass_fields__
                        if "tool" in name],
        "stack_items": 12,
        "implemented": len([name for name in ref.Governor.__dataclass_fields__
                            if name.startswith("enable_")]),
        "request_cap": governor.max_tokens_per_request,
        "read_tokens": TOOLS[0][1],
        "read_at_cap": TOOLS[0][1] >= governor.max_tokens_per_request,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: tightest on the consequential write, and nothing is free",
            all([result["tools"] == 4, result["tightest"] == ["submit or pay"],
                 result["unbounded"] == [], result["read_dollars"] == 0.03,
                 result["read_over_click"] == 50.0]),
            f"of {result['tools']} tools the tightest cap goes on "
            f"{result['tightest'][0]} -- risk not denominated in money -- while a page "
            f"read is ${result['read_dollars']} a call, {result['read_over_click']}x a "
            f"click; {len(result['unbounded'])} tools run unbounded without risk",
        ),
        practice.Check(
            "FINDING: the two rankings are near-reverses",
            all([result["by_dollars"][0] == "read page",
                 result["by_risk"][0] == "submit or pay",
                 result["top_of_money_is_bottom_of_risk"]]),
            f"by dollars the order is {result['by_dollars']} and by irreversibility "
            f"{result['by_risk']} -- the tool at the top of one is at the bottom of the "
            "other, so one number per tool cannot state the policy",
        ),
        practice.Check(
            "FINDING: the shipped governor has no per-tool anything",
            all([result["governor_fields"] == 12, result["tool_fields"] == [],
                 result["implemented"] == 5, result["stack_items"] == 12]),
            f"Governor carries {result['governor_fields']} fields and "
            f"{len(result['tool_fields'])} name a tool; the lesson's stack lists "
            f"{result['stack_items']} control types and the simulator implements "
            f"{result['implemented']}, without the per-tool cap that was the fix in the "
            "case study",
        ),
        practice.Check(
            "FINDING: a page read sits exactly on the request cap",
            all([result["read_at_cap"], result["read_tokens"] == result["request_cap"]]),
            f"a {result['read_tokens']}-token page is max_tokens_per_request to the "
            "token, so the cap truncates the first page the agent opens -- a "
            "correctness failure the cost layer records as a success",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
