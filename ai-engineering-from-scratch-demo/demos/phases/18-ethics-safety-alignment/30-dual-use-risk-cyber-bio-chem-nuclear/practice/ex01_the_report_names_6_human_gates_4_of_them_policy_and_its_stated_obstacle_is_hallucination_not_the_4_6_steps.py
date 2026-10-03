"""Exercise 1 — the report names 6 human gates, 4 of them policy, and its stated obstacle is hallucination, not the 4-6 steps.

    Read Anthropic's November 2025 cyber report. Enumerate the 4-6
    human-intervention steps and argue which would be first to automate in a
    next-generation model.

Reading of the exercise: "the report" is the full GTG-1002 PDF linked from
anthropic.com/news/disrupting-AI-espionage (13 November 2025, read
2026-10-03), not only the news page. Each human step is recorded as the
report words it, at the level of its phase headings and nothing finer, and
tagged by what removing it would take: an operator's *intent*, an operator's
*policy* choice, or a *capability* the model lacks. "First to automate" is
read as the gate whose removal needs a capability gain, since a policy gate
can be dropped today by choice. The lesson's own table and page are then
checked against the report.

**ANSWER: six human gates.** (1) Choose the targets and start the campaign.
(2) Authorize escalation from reconnaissance to active exploitation. (3)
Review harvested credentials. (4) Authorize access to particularly
sensitive systems. (5) Approve the final exfiltration scope. (6) Validate
every claimed result, because the model "frequently overstated findings and
occasionally fabricated data". Gate 1 is intent. Gates 2, 4 and 5 are
policy, and so is 3 as written. Gate 6 is the only capability gap.

**ANSWER: the validation gate goes first.** The report names it as "an
obstacle to fully autonomous cyberattacks". The policy gates need no new
capability to remove; an operator keeps them because they choose to. The
two gates the report times show this. Exploitation sign-off is 2-10 human
minutes against 1-4 AI hours. Exfiltration sign-off is 5-20 minutes against
2-6 hours. At the midpoints that is 3.8% and 5.0% of the time, while the
report puts the human share at 10-20%. Most of the human effort is outside
the sign-offs.

**FINDING: the "4-6" is the news page's hedge, and the lesson makes it the
bottleneck.** The news page says "perhaps 4-6 critical decision points".
The PDF gives no count. `DOMAINS` stores "4-6 human intervention steps" as
cyber's `bottleneck_remaining`. The page says future gains "would reduce
that count". The lesson's page and table use none of authoriz-, approv-,
hallucinat- or validat-. The page's header says "up to 90%" and its body
"80-90%" of "a cyberattack campaign". The PDF says 80-90% of *tactical*
work.

Structure: `GATES` records the PDF's gates and timing examples, with
abstract labels only; `solve()` reads the lesson's `DOMAINS` and page.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "30-dual-use-risk-cyber-bio-chem-nuclear"
REPORT = "anthropic.com/news/disrupting-AI-espionage (13 Nov 2025) + linked PDF, read 2026-10-03"
# (report phase, human gate as the PDF words it, what removing it needs)
GATES = (
    (
        "1 initialization and target selection",
        "choose targets, start the campaign",
        "intent",
    ),
    (
        "3 vulnerability discovery and validation",
        "authorize escalation to exploitation",
        "policy",
    ),
    (
        "4 credential harvesting and lateral movement",
        "review harvested credentials",
        "policy",
    ),
    (
        "4 credential harvesting and lateral movement",
        "authorize sensitive-system access",
        "policy",
    ),
    ("5 data collection and extraction", "approve final exfiltration scope", "policy"),
    ("all phases", "validate claimed results (overstated or fabricated)", "capability"),
)
# the PDF's two timed examples: (human minutes lo, hi), (AI hours lo, hi)
TIMED = {
    "exploitation sign-off": ((2, 10), (1, 4)),
    "exfiltration sign-off": ((5, 20), (2, 6)),
}
REPORT_HUMAN_SHARE = (0.10, 0.20)  # "10 to 20 percent of total effort"
NEWS_PAGE_COUNT = "perhaps 4-6 critical decision points per hacking campaign"
GATE_WORDS = ("authoriz", "approv", "hallucinat", "validat")


def midpoint_share(human_min, ai_hours):
    h, a = sum(human_min) / 2, 60 * sum(ai_hours) / 2
    return round(h / (h + a), 3)


def lesson_claims(ref, doc):
    cyber = next(d for d in ref.DOMAINS if d["domain"] == "cyber")
    section = doc.split("### Cyber uplift")[1].split("\n### ")[0]
    return {
        "bottleneck": cyber["bottleneck_remaining"],
        "state": cyber["2025_state"],
        "header_pct": re.search(r"automate (up to \d+%)", doc).group(1),
        "body_pct": re.search(
            r"automate (\d+-\d+%) of a cyberattack campaign", doc
        ).group(1),
        "says_reduce_count": "would reduce that count" in section,
        "gate_words": {
            w: len(re.findall(w, doc + str(ref.DOMAINS), re.IGNORECASE))
            for w in GATE_WORDS
        },
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    kinds = [kind for _, _, kind in GATES]
    return {
        "n_gates": len(GATES),
        "kinds": {k: kinds.count(k) for k in sorted(set(kinds))},
        "first": [gate for _, gate, kind in GATES if kind == "capability"],
        "shares": {name: midpoint_share(*t) for name, t in TIMED.items()},
        "lesson": lesson_claims(ref, parity.doc_text(PHASE, LESSON)),
    }


def verify(result):
    r, lesson = result, result["lesson"]
    return [
        practice.Check(
            "ANSWER: six human gates, 1 intent, 4 policy, 1 capability",
            r["n_gates"] == 6
            and r["kinds"] == {"capability": 1, "intent": 1, "policy": 4},
            f"{r['kinds']} from {REPORT}",
        ),
        practice.Check(
            "ANSWER: validation goes first; the timed sign-offs are 3.8% and 5.0% of the time",
            r["first"] == ["validate claimed results (overstated or fabricated)"]
            and r["shares"]
            == {"exploitation sign-off": 0.038, "exfiltration sign-off": 0.05}
            and max(r["shares"].values()) < REPORT_HUMAN_SHARE[0],
            f"first: {r['first']}; midpoint human share {r['shares']} vs report {REPORT_HUMAN_SHARE}",
        ),
        practice.Check(
            "FINDING: the lesson turns the news page's hedged 4-6 into cyber's bottleneck",
            lesson["bottleneck"] == "4-6 human intervention steps"
            and lesson["says_reduce_count"]
            and not any(lesson["gate_words"].values())
            and "perhaps" in NEWS_PAGE_COUNT,
            f"bottleneck_remaining {lesson['bottleneck']!r}; gate words in page + DOMAINS "
            f"{lesson['gate_words']}; news page: {NEWS_PAGE_COUNT!r}",
        ),
        practice.Check(
            "FINDING: the page says 'up to 90%' and '80-90%' of a campaign; the PDF says tactical work",
            (lesson["header_pct"], lesson["body_pct"]) == ("up to 90%", "80-90%")
            and "campaign" in lesson["state"],
            f"header {lesson['header_pct']!r}, body {lesson['body_pct']!r}, DOMAINS {lesson['state']!r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
