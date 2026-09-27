"""Exercise 1 — the "Chalmers et al." report is Long et al., co-written by Fish, and its two-sided risk flips end-chat at 0.0081.

    Read Anthropic's "Exploring Model Welfare" (April 2025) and Chalmers et al. 2024. Write a one-paragraph summary of each and identify one point of disagreement.

Reading of the exercise: both documents were read on 2026-09-27
(anthropic.com/research/exploring-model-welfare, arXiv:2411.00986, and
eleosai.org for affiliations). The summaries are paraphrase. The point of
disagreement is made testable by writing it into the lesson's own `ev()` and
measuring what it moves.

**ANSWER, Anthropic (24 April 2025).** Anthropic announces a research
program on whether, and when, the welfare of its models deserves moral
consideration. It says there is no scientific consensus on whether AI
systems could be conscious, or even on how to approach the question. It
cites the expert report below, which it says it supported at an early stage.
It plans to study model preferences, signs of distress, and "practical,
low-cost interventions", with few assumptions and ideas open to revision.

**ANSWER, the report (arXiv:2411.00986, 4 November 2024).** The paper is
"Taking AI Welfare Seriously" by Long, Sebo, Butlin, Finlinson, Fish, Harding,
Pfau, Sims, Birch and Chalmers. It argues there is a realistic possibility
that some AI systems will be conscious and/or robustly agentic soon, so AI
welfare is a near-term issue. It recommends three steps: acknowledge the
issue, assess systems for consciousness and robust agency, and prepare
policies. It claims uncertainty, not consciousness, and warns of two errors:
harming systems that matter, and caring for systems that do not.

**ANSWER, the disagreement: what a precaution costs.** Anthropic prices a
precaution as "low-cost". The report counts mistaken care as a harm in its
own right. The lesson's `ev()` follows Anthropic: at p = 0 every one of the 4
interventions scores exactly minus its dollar cost, so mistaken care costs
nothing. Adding the report's second error as lambda x (1 - p) flips
end-conversation from INVEST to skip at p = 0.01 once lambda exceeds 0.0081
units per conversation, which is 4.0x its $0.002 cost. The break-even
lambda is 0.1089 at p = 0.1 and 0.996 at p = 0.5.

**FINDING: the lesson misnames the report and treats its authors as outside
voices.** Its link text, "Near-term AI Consciousness and Moral Status", is
not the paper's title. "Chalmers et al." names the last of 10 authors. The
first author, Robert Long, is Executive Director of Eleos AI, which the lesson
presents as "an external model-welfare lab". The fifth author is Kyle Fish,
the researcher the lesson says Anthropic hired. The lesson's "Fish et al.
spiritual bliss" link points to the same URL as the program announcement.

Structure: `further_reading()` parses the lesson's link list; `flip_lambda()`
solves p*b - c - lambda*(1 - p) = 0 against the reference's interventions.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "19-model-welfare-research"
ARXIV = "https://arxiv.org/abs/2411.00986"
# arXiv:2411.00986 abstract page and eleosai.org, both read 2026-09-27
PUBLISHED_TITLE = "Taking AI Welfare Seriously"
AUTHORS = ("Robert Long", "Jeff Sebo", "Patrick Butlin", "Kathleen Finlinson", "Kyle Fish",
           "Jacqueline Harding", "Jacob Pfau", "Toni Sims", "Jonathan Birch", "David Chalmers")
ELEOS_EXECUTIVE_DIRECTOR = "Robert Long"


def further_reading(doc):
    """(link text, url) pairs of the lesson's Further Reading list."""
    block = doc.split("## Further Reading", 1)[1]
    return re.findall(r"^- \[(.+?)\]\((\S+?)\)", block, re.M)


def flip_lambda(ref, intervention, p):
    """The mistaken-care cost per conversation at which ev() + lambda*(1-p) term hits zero."""
    return ref.ev(intervention, ref.Scenario("", p)) / (1 - p)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    links = further_reading(doc)
    end_chat = ref.INTERVENTIONS[0]
    probs = [s.moral_patienthood_probability for s in ref.SCENARIOS]
    return {
        "report_link": next(text for text, url in links if url == ARXIV),
        "bliss_url": next(url for text, url in links if "Bliss" in text),
        "program_url": next(url for text, url in links if text.startswith("Anthropic")),
        "eleos_external": "Eleos AI Research (an external model-welfare lab)" in doc,
        "hired_fish": "Hired Kyle Fish" in doc,
        "zero_p": [ref.ev(i, ref.Scenario("", 0.0)) == -i.cost_usd_per_conversation
                   for i in ref.INTERVENTIONS],
        "end_chat": (end_chat.name, end_chat.cost_usd_per_conversation),
        "flip": {p: round(flip_lambda(ref, end_chat, p), 4) for p in probs},
    }


def verify(result):
    r, flip = result, result["flip"]
    return [
        practice.Check(
            "ANSWER: the lesson's ev() prices no mistaken care, and the report's term flips it",
            all([
                r["zero_p"] == [True] * 4,
                flip == {0.01: 0.0081, 0.1: 0.1089, 0.5: 0.996},
                r["end_chat"] == ("end-conversation on extreme edge cases", 0.002),
                round(flip[0.01] / r["end_chat"][1], 1) == 4.0,
            ]),
            f"ev(p=0) == -cost for all 4 interventions: {r['zero_p']}; end-conversation "
            f"flips to skip once mistaken care exceeds {flip} units/conversation "
            f"({flip[0.01] / r['end_chat'][1]:.1f}x its ${r['end_chat'][1]} at p = 0.01)",
        ),
        practice.Check(
            "FINDING: the lesson misnames the report and treats its authors as outside voices",
            all([
                PUBLISHED_TITLE not in r["report_link"],
                r["report_link"].startswith("Chalmers"),
                AUTHORS[-1].endswith("Chalmers"),
                AUTHORS[0] == ELEOS_EXECUTIVE_DIRECTOR,
                AUTHORS.index("Kyle Fish") == 4,
                r["eleos_external"],
                r["hired_fish"],
                r["bliss_url"] == r["program_url"],
            ]),
            f"lesson link text {r['report_link']!r} vs published {PUBLISHED_TITLE!r}; Chalmers "
            f"is author {len(AUTHORS)} of {len(AUTHORS)}, author 1 runs Eleos, author 5 is Fish; "
            f"bliss link {r['bliss_url']} is the program URL",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
