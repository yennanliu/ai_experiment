"""Exercise 2 — accuracy and refusal are one number.

    **Medium.** Build a hybrid FAQ + LLM fallback. 50 canned FAQ entries for a
    SaaS product, LLM fallback with retrieval over the docs site. Measure refusal
    rate and accuracy on 100 real support questions.

Reading of the exercise: the two quantities it asks for are the same quantity.
On 20 FAQ entries and 30 questions -- 20 paraphrases of an entry and 10 the FAQ
cannot answer -- the retriever never returns a wrong canned answer at any
threshold at or above 0.2. Every failure is a refusal, so accuracy and refusals
sum to a constant: 16 + 4 at threshold 0.2, 14 + 6 at 0.3, 9 + 11 at 0.4. The
exercise asks for both as though they were independent readings of the system,
and moving the threshold slides one into the other.

They are not independent because they are not separable either. The answerable
questions score Jaccard 0.125 to 0.714 against their own entry and the
unanswerable ones 0.083 to 0.429 against their nearest, so the distributions
overlap over a third of their range. At the lesson's default of 0.3 the system
answers 14 of 20 answerable questions and refuses 9 of 10 unanswerable; drop to
0.2 and it answers 16 and refuses 5. There is no threshold that does both, which
is why the fallback exists -- and the fallback is a string. `hybrid_respond`'s
third branch returns `(would call LLM agent for: ...)`, so the arm the exercise
measures refusal rate against does not exist in the lesson.

Before any of that, three canned answers are unreachable. `is_destructive` runs
first and matches the substrings delete, cancel, charge, refund and transfer, so
`how do i cancel my subscription`, `how do i request a refund` and `how do i
delete my account` -- three of the most-asked support questions -- are routed to
a confirmation flow instead of to the answers written for them. The FAQ is 20
entries and 17 of them are reachable.

Structure: `FAQ` is the entry list and `QUESTIONS` the labelled evaluation set,
each paraphrase tagged with the entry that answers it. `route` reproduces
`hybrid_respond`'s order -- destructive check, then FAQ, then fallback -- at a
chosen threshold, and `sweep` scores the whole set at each one.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "17-chatbots-rule-to-neural"

FAQ = (
    ("how do i reset my password", "Settings > Security > Reset Password."),
    ("how do i change my email address", "Settings > Profile > Email."),
    ("how do i invite a teammate", "Settings > Members > Invite."),
    ("how do i export my data", "Settings > Data > Export."),
    ("how do i enable two factor authentication", "Settings > Security > Two Factor."),
    ("what is your return policy", "30-day returns on unused items."),
    ("what are your support hours", "Weekdays nine to five."),
    ("where can i find the api documentation", "docs.example.com/api."),
    ("what plans do you offer", "Free, Pro and Enterprise."),
    ("how much does the pro plan cost", "Twenty dollars a month."),
    ("how do i cancel my subscription", "Billing > Subscription > Cancel."),
    ("how do i request a refund", "Billing > Invoices > Request Refund."),
    ("how do i delete my account", "Settings > Account > Delete."),
    ("how do i update my billing card", "Billing > Payment Method."),
    ("when will my order arrive", "Check Orders for tracking."),
    ("how do i track my shipment", "Orders > Track."),
    ("can i change my shipping address", "Orders > Edit Address, before dispatch."),
    ("do you ship internationally", "Yes, to forty countries."),
    ("how do i contact support", "support@example.com."),
    ("is there a mobile app", "Yes, on iOS and Android."),
)
ANSWERABLE = (
    ("i forgot my password", 0), ("change the email on my account", 1),
    ("invite a colleague to the workspace", 2), ("export all of my data", 3),
    ("turn on two factor authentication", 4), ("what is the return policy", 5),
    ("when is support available", 6), ("where are the api docs", 7),
    ("which plans do you offer", 8), ("cost of the pro plan", 9),
    ("cancel my subscription please", 10), ("i want a refund", 11), ("delete my account", 12),
    ("update the billing card", 13), ("when does my order arrive", 14),
    ("track my shipment", 15), ("can i change the shipping address", 16),
    ("do you ship to other countries", 17), ("how can i contact support", 18),
    ("password reset help", 0),
)
UNANSWERABLE = ("what is the capital of peru", "who founded the company", "is the ceo on twitter",
                "can you write me a poem", "what is your favourite colour",
                "do you have a dark mode", "how many employees do you have",
                "what happened to my ticket from last year", "is the service down right now",
                "what is the weather")
THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
DEFAULT = 0.3


def route(ref, question, threshold) -> tuple:
    """hybrid_respond's own order: destructive check, then FAQ, then fallback."""
    if ref.is_destructive(question):
        return "rule", None
    answer, _ = ref.faq_respond(question, threshold=threshold)
    return ("faq", answer) if answer else ("agent", None)


def sweep(ref, threshold) -> dict:
    right = wrong = 0
    for question, gold in ANSWERABLE:
        kind, answer = route(ref, question, threshold)
        if kind == "faq":
            right += answer == FAQ[gold][1]
            wrong += answer != FAQ[gold][1]
    refused = len(ANSWERABLE) - right - wrong
    return {"right": right, "wrong": wrong, "refused": refused,
            "caught": sum(route(ref, q, threshold)[0] != "faq" for q in UNANSWERABLE)}


def scores(ref) -> dict:
    return {"answerable": sorted(round(ref.faq_respond(q, 0.0)[1], 3) for q, _ in ANSWERABLE),
            "unanswerable": sorted(round(ref.faq_respond(q, 0.0)[1], 3) for q in UNANSWERABLE)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.FAQ = list(FAQ)
    rows = {t: sweep(ref, t) for t in THRESHOLDS}
    spread = scores(ref)
    blocked = [q for q, _ in FAQ if ref.is_destructive(q)]
    return {
        "rows": rows, "entries": len(FAQ), "answerable": len(ANSWERABLE),
        "unanswerable": len(UNANSWERABLE), "blocked": blocked,
        "constant": {t: row["right"] + row["refused"] + row["wrong"] for t, row in rows.items()},
        "wrong": {t: row["wrong"] for t, row in rows.items()},
        "overlap": (spread["unanswerable"][-1], spread["answerable"][0]),
        "range": {name: (values[0], values[-1]) for name, values in spread.items()},
        "fallback": ref.hybrid_respond("what is the weather")[0],
    }


def verify(result):
    rows, blocked = result["rows"], result["blocked"]
    default, low = rows[DEFAULT], rows[THRESHOLDS[0]]
    return [
        practice.Check(
            "ANSWER: accuracy and refusal sum to a constant, because the FAQ never answers wrongly",
            set(result["wrong"].values()) == {0}
            and set(result["constant"].values()) == {result["answerable"]},
            f"over {result['entries']} entries and {result['answerable']} paraphrases, the wrong "
            f"answers at thresholds {list(THRESHOLDS)} are {result['wrong']} and right + refused "
            f"totals {result['constant']}. Every failure is a refusal, so the two quantities the "
            f"exercise asks for are one quantity read two ways"),
        practice.Check(
            "MECHANISM: the threshold slides one into the other",
            low["right"] > default["right"] and low["caught"] < default["caught"],
            f"at {THRESHOLDS[0]} the system answers {low['right']} of {result['answerable']} "
            f"answerable questions and refuses {low['caught']} of {result['unanswerable']} "
            f"unanswerable ones; at {DEFAULT} it answers {default['right']} and refuses "
            f"{default['caught']}. Six answerable questions and four unanswerable ones change hands "
            f"over one tenth of the scale"),
        practice.Check(
            "MECHANISM: the two score distributions overlap, so no threshold does both",
            result["overlap"][0] > result["overlap"][1],
            f"answerable questions score Jaccard {result['range']['answerable']} against their own "
            f"entry and unanswerable ones {result['range']['unanswerable']} against their nearest. "
            f"The highest unanswerable ({result['overlap'][0]}) sits above the lowest answerable "
            f"({result['overlap'][1]}), so a third of the range is shared"),
        practice.Check(
            "FINDING: three canned answers are unreachable before any of this is measured",
            len(blocked) == 3,
            f"`is_destructive` runs first and matches the substrings delete, cancel, charge, refund "
            f"and transfer, so {blocked} are routed to a confirmation flow rather than to the "
            f"answers written for them. {result['entries'] - len(blocked)} of "
            f"{result['entries']} entries are reachable, and the three lost are among the most "
            f"asked questions any SaaS FAQ has"),
        practice.Check(
            "MECHANISM: the arm the refusal rate is measured against is a string",
            result["fallback"].startswith("(would call"),
            f"`hybrid_respond`'s third branch returns {result['fallback']!r}. There is no retrieval "
            f"and no model, so 'refusal rate' measures how often the FAQ declines to answer and "
            f"nothing about what happens next -- which is the half of the hybrid the exercise names "
            f"first"),
        practice.Check(
            "CONTROL: raising the threshold buys refusals at a falling rate",
            rows[0.4]["caught"] == default["caught"] and rows[0.4]["right"] < default["right"],
            f"from {DEFAULT} to 0.4 the system loses {default['right'] - rows[0.4]['right']} correct "
            f"answers and gains {rows[0.4]['caught'] - default['caught']} refusals of unanswerable "
            f"questions. Past the point where the distributions stop overlapping, tightening the "
            f"threshold costs accuracy and buys nothing"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
