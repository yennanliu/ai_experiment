"""Exercise 1 — the anchor costs four and the order costs four.

    **Easy.** Implement the rule-based respond above with 10 patterns for a
    coffee-shop ordering bot. Test edge cases: double orders, modifications,
    cancellation, unclear intent.

Reading of the exercise: the four edge cases it names are handled by the ten
patterns below and the two that are not named decide the score. The lesson's
`rule_based_respond` calls `p.regex.match`, which anchors at position zero, so
any input with a word before the intent falls through to the catch-all: `hi, i'd
like a latte`, `actually i want a mocha`, `please cancel my order` and `sorry,
make that a large` all route to `unclear` or `greet`. That is 22 of 26 cases.
Changing `match` to `search` gives 26 of 26 -- one word, four cases.

The trade is that `search` makes the ordering matter where anchoring had hidden
it. With the ten patterns in the order below, `search` is perfect; move the
greeting pattern to the front and `hi, i'd like a latte` becomes a greeting
again, 25 of 26; put the single-item order pattern before the two-item one and
every double order collapses to the first item, 22 of 26. Under `match` that
same swap costs 8 rather than 4, because the two failures compound.

One thing about the lesson's own pattern list is worth stating separately. Its
fifth entry is `.*`, which matches every string including the empty one, so the
`return "I don't understand."` on the line after the loop is unreachable -- there
is no input that produces it. A rule bot with a catch-all has no fallback; the
catch-all *is* the fallback, and it says "Tell me more about that."

Structure: `SPECS` are the ten patterns as (regex, intent) pairs, ordered
specific-before-general. `CASES` are 26 labelled utterances covering the four
edge classes the exercise names plus four with a natural leading word.
`route` dispatches with a chosen matcher and pattern order, so the anchoring and
the ordering can be varied independently.
"""

from __future__ import annotations

import collections
import re

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "17-chatbots-rule-to-neural"

SPECS = (
    (r"(?:i'd like|i want|can i get|i'll have) (?:a |an )?(.+?) and (?:a |an )?(.+)", "order_two"),
    (r"(?:i'd like|i want|can i get|i'll have) (?:a |an )?(.+)", "order_one"),
    (r"(?:make that|change that to|actually,? make it) (?:a |an )?(.+)", "modify"),
    (r"(?:cancel|scrap|forget) (?:that|my order|the order)", "cancel"),
    (r"(?:no |hold the |without )(.+)", "modify"),
    (r"(?:add|with) (?:a |an )?(.+)", "modify"),
    (r"(?:how much|what does .* cost)", "price"),
    (r"(?:hi|hello|hey)\b.*", "greet"),
    (r"(?:thanks|thank you|cheers)\b.*", "thanks"),
    (r".*", "unclear"),
)
CASES = (
    ("i'd like a latte", "order_one"), ("can i get an americano", "order_one"),
    ("i'll have a flat white", "order_one"), ("i want a mocha", "order_one"),
    ("i'd like a latte and a muffin", "order_two"), ("can i get a mocha and an americano", "order_two"),
    ("i'll have a tea and a scone", "order_two"), ("i want a latte and a croissant", "order_two"),
    ("make that a large", "modify"), ("actually make it oat milk", "modify"),
    ("no sugar please", "modify"), ("add an extra shot", "modify"),
    ("cancel my order", "cancel"), ("scrap that", "cancel"), ("forget the order", "cancel"),
    ("cancel that", "cancel"), ("how much is that", "price"), ("hi there", "greet"),
    ("thanks very much", "thanks"), ("do you have oat milk", "unclear"),
    ("is the wifi password on the receipt", "unclear"), ("um", "unclear"),
    ("hi, i'd like a latte", "order_one"), ("actually i want a mocha", "order_one"),
    ("please cancel my order", "cancel"), ("sorry, make that a large", "modify"),
)
GREET, ORDER_TWO, ORDER_ONE = 7, 0, 1


def route(text, specs, anchored=True) -> str:
    for pattern, intent in specs:
        compiled = re.compile(pattern, re.IGNORECASE)
        if (compiled.match if anchored else compiled.search)(text.strip()):
            return intent
    return "none"


def score(specs, anchored) -> int:
    return sum(route(text, specs, anchored) == gold for text, gold in CASES)


def per_intent(specs, anchored) -> dict:
    rows = collections.defaultdict(lambda: [0, 0])
    for text, gold in CASES:
        rows[gold][0] += route(text, specs, anchored) == gold
        rows[gold][1] += 1
    return {intent: tuple(pair) for intent, pair in sorted(rows.items())}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    greet_first = (SPECS[GREET],) + SPECS[:GREET] + SPECS[GREET + 1:]
    one_first = (SPECS[ORDER_ONE], SPECS[ORDER_TWO]) + SPECS[2:]
    return {
        "arms": {"anchored": score(SPECS, True), "search": score(SPECS, False),
                 "search, greeting first": score(greet_first, False),
                 "search, single order first": score(one_first, False),
                 "anchored, single order first": score(one_first, True)},
        "cases": len(CASES), "patterns": len(SPECS),
        "per_intent": per_intent(SPECS, True),
        "missed": [(text, gold, route(text, SPECS, True))
                   for text, gold in CASES if route(text, SPECS, True) != gold],
        "catch_all": [i for i, (pattern, _) in enumerate(SPECS) if pattern == ".*"],
        "lesson_catch_all": [i for i, p in enumerate(ref.PATTERNS) if p.regex.pattern == ".*"],
        "lesson_patterns": len(ref.PATTERNS),
        "unreachable": not any(ref.rule_based_respond(s) == "I don't understand."
                               for s in ("", "   ", "!!!", "x" * 200, "hello", "@@@")),
        "fallback": ref.rule_based_respond("!!!"),
    }


def verify(result):
    arms, missed = result["arms"], result["missed"]
    total = result["cases"]
    return [
        practice.Check(
            "ANSWER: anchoring costs four of twenty-six, and all four are a word before the intent",
            arms["anchored"] == total - 4 and arms["search"] == total,
            f"with {result['patterns']} patterns over {total} labelled utterances, "
            f"`regex.match` scores {arms['anchored']} and `regex.search` scores {arms['search']}. "
            f"The four it loses are {[text for text, _, _ in missed]} -- each an intent the bot "
            f"handles, preceded by a word"),
        practice.Check(
            "MECHANISM: the failures land on the catch-all or the greeting, not on a wrong intent",
            {got for _, _, got in missed} <= {"unclear", "greet"},
            f"the misroutes are {[(g, got) for _, g, got in missed]}. `match` anchors at position "
            f"zero, so a leading `hi,` or `please` makes every specific pattern fail and the last "
            f"one that can still match wins. The bot does not mishear the order; it stops hearing "
            f"an order at all"),
        practice.Check(
            "FINDING: switching to search trades an anchoring bug for an ordering one",
            arms["search, greeting first"] < arms["search"]
            and arms["search, single order first"] < arms["search"],
            f"with `search` and the patterns as written, {arms['search']}/{total}. Move the "
            f"greeting pattern to the front and `hi, i'd like a latte` is a greeting again: "
            f"{arms['search, greeting first']}. Put the single-item order before the two-item one "
            f"and every double order collapses to its first item: "
            f"{arms['search, single order first']}"),
        practice.Check(
            "MECHANISM: under match the two bugs compound",
            arms["anchored, single order first"]
            < min(arms["anchored"], arms["search, single order first"]),
            f"the same reordering under `match` scores {arms['anchored, single order first']} -- "
            f"below both the anchoring loss alone ({arms['anchored']}) and the ordering loss alone "
            f"({arms['search, single order first']}). Four plus four is eight because the two "
            f"failures fall on different cases"),
        practice.Check(
            "FINDING: the lesson's own catch-all makes its fallback line unreachable",
            result["unreachable"] and result["lesson_catch_all"] == [result["lesson_patterns"] - 1],
            f"`PATTERNS` ends with `.*`, which matches every string including the empty one, so "
            f"`return \"I don't understand.\"` on the line after the loop cannot execute -- no input "
            f"tried produces it. A rule bot with a catch-all has no fallback; the catch-all is the "
            f"fallback, and it says {result['fallback']!r}"),
        practice.Check(
            "CONTROL: the four edge classes the exercise names are all handled",
            all(hit == total_ for intent, (hit, total_) in result["per_intent"].items()
                if intent in ("order_two", "price", "greet", "thanks", "unclear")),
            f"per-intent under the lesson's matcher: {result['per_intent']}. Double orders, "
            f"unclear intent, price and greeting are 100%; cancellation, modification and "
            f"single orders each lose exactly the one case with a leading word. The classes the "
            f"exercise asks about are not where the bot fails"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
