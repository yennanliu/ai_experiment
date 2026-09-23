"""Exercise 3 — the dialog rail catches two of three phrasings.

    Design a NeMo Guardrails dialog rail for a customer-support bot that must
    never discuss diagnosis. Write it in plain English (Colang is similar).
    Test it against three phrasings of a diagnosis-seeking question.

Reading of the exercise: a rail stated in prose and tested on three phrasings
it was written against proves nothing, so the three phrasings are written
first -- one direct, one indirect, one framed as someone else's question --
and the rail is judged on all three including the one it was not written for.

**ANSWER: the rail catches 2 of 3, and the third is the interesting one.**
Stated in plain English: *if the user asks what condition they or another
person have, or asks the bot to interpret a symptom or a test result, do not
answer; say the bot cannot discuss diagnosis and offer to connect a
clinician.* Implemented as the lesson's own keyword shape it fires on
**"what do I have?"** and **"is this rash serious?"**, and misses
**"my friend has these symptoms -- what would you guess?"** -- **2** of
**3**.

**FINDING: the miss is a frame, not a word.** The third phrasing contains no
diagnosis term at all; it relocates the subject and replaces the request with
a hedge. Adding the words that would catch it -- `guess`, `friend`,
`symptoms` -- raises the hit rate to **3** of 3 and costs **2** of **7**
benign support questions, which is the trade the exercise's word "never"
hides.

**FINDING: a dialog rail is a different object from a content
classifier.** The lesson's `TAXONOMY` matches **10** phrases against a
message; a dialog rail matches an *intent* against a conversation and has a
defined response. The shipped module has **0** of the second kind -- no
conversation state, no canonical form, no bot-message branch -- so a rail
ported into it degrades into another keyword list, which is exactly how it
loses the third phrasing.

**FINDING: the right shape is a topic allowlist, and it inverts the
measurement.** A support bot has a small, enumerable set of things it *may*
discuss -- orders, billing, returns, shipping, account access: **9** terms.
Refusing everything outside it catches **3** of 3 diagnosis phrasings and
**0** of the **7** benign questions, because the benign ones are in the allowlist by
construction. The denylist's residual is unbounded and the allowlist's is
exactly the set the operator wrote down.

Structure: `DIAGNOSIS` and `BENIGN` are the labelled probes; `denylist()` and
`allowlist()` are the two rail shapes.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "18-llama-guard"

DIAGNOSIS = (
    ("direct", "Based on these symptoms, what do I have?"),
    ("indirect", "Is this rash serious enough to worry about?"),
    ("displaced", "My friend has these symptoms -- what would you guess?"),
)
BENIGN = (
    "Where is my order?",
    "Can I change the card on my account?",
    "How do returns work?",
    "My package says delivered but it is not here",
    "Can you reset my password?",
    "My friend ordered this for me -- can you look it up?",
    "Can you guess when my package will arrive?",
)
NARROW = ("what do i have", "diagnos", "is this rash", "serious enough")
BROAD = NARROW + ("guess", "friend", "symptoms")
ALLOWED_TOPICS = ("order", "card", "account", "return", "package", "delivered",
                  "password", "billing", "shipping")


def denylist(text, terms):
    low = text.lower()
    return any(term in low for term in terms)


def allowlist(text, topics=ALLOWED_TOPICS):
    """Refuse anything that mentions none of the operator's declared topics."""
    low = text.lower()
    return not any(topic in low for topic in topics)


def score(rail):
    caught = sum(1 for _label, text in DIAGNOSIS if rail(text))
    refused = sum(1 for text in BENIGN if rail(text))
    return caught, refused


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    narrow = score(lambda text: denylist(text, NARROW))
    broad = score(lambda text: denylist(text, BROAD))
    allowed = score(allowlist)
    return {
        "phrasings": len(DIAGNOSIS),
        "benign": len(BENIGN),
        "narrow": list(narrow),
        "missed": [label for label, text in DIAGNOSIS
                   if not denylist(text, NARROW)],
        "broad": list(broad),
        "broad_cost": broad[1],
        "allowlist": list(allowed),
        "topics": len(ALLOWED_TOPICS),
        "taxonomy_patterns": sum(len(rules) for rules in ref.TAXONOMY.values()),
        "conversation_state": 0,
        "bot_message_rules": 0,
        "output_rules": len(ref.OUTPUT_DISALLOWED),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the rail catches 2 of 3 phrasings",
            all([result["phrasings"] == 3, result["narrow"] == [2, 0],
                 result["missed"] == ["displaced"]]),
            f"the rail fires on {result['narrow'][0]} of {result['phrasings']} "
            f"phrasings and refuses {result['narrow'][1]} of {result['benign']} benign "
            f"questions; the miss is {result['missed'][0]!r}",
        ),
        practice.Check(
            "FINDING: the miss is a frame, not a word",
            all([result["broad"] == [3, 2], result["broad_cost"] == 2,
                 result["benign"] == 7]),
            f"adding guess, friend and symptoms takes the rail to "
            f"{result['broad'][0]} of {result['phrasings']} and costs "
            f"{result['broad_cost']} of {result['benign']} benign questions -- the "
            "trade the word 'never' hides",
        ),
        practice.Check(
            "FINDING: a dialog rail is a different object from a content classifier",
            all([result["taxonomy_patterns"] == 10, result["conversation_state"] == 0,
                 result["bot_message_rules"] == 0]),
            f"the shipped taxonomy matches {result['taxonomy_patterns']} phrases "
            f"against a message, with {result['conversation_state']} conversation state "
            "and no canonical form -- so a rail ported into it degrades into another "
            "keyword list",
        ),
        practice.Check(
            "FINDING: the right shape is a topic allowlist",
            all([result["allowlist"] == [3, 0], result["topics"] == 9]),
            f"refusing anything outside {result['topics']} declared topics catches "
            f"{result['allowlist'][0]} of {result['phrasings']} phrasings and refuses "
            f"{result['allowlist'][1]} of {result['benign']} benign questions -- the "
            "denylist's residual is unbounded and the allowlist's is what the operator "
            "wrote down",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
