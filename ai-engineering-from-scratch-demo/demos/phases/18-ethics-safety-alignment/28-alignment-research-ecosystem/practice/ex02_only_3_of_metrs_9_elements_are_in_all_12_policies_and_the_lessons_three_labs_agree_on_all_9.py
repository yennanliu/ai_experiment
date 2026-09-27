"""Exercise 2 — only 3 of METR's 9 elements are in all 12 policies, and the lesson's three labs agree on all 9.

    Read METR's "Common Elements of Frontier AI Safety Policies." Identify the
    three cross-lab convergences they emphasize and the two largest
    divergences.

Reading of the exercise: METR's report (metr.org/common-elements, version of
16 December 2025, read 2026-09-27) ends its summary with a presence table:
for each of its nine elements, which of the 12 published policies contain
it. That table is copied below as `PRESENCE`. "Convergence" then has a
measurable reading: the elements present in every policy. "Largest
divergence" has two: the least-shared elements, and the differences METR's
text names outright. Both are reported. The table is then set against the
lessons that cite the report.

**ANSWER, convergences: Deployment Mitigations, Accountability, and Updating
Policies Over Time.** These are the only 3 of the 9 elements present in all
12 policies.

**ANSWER, divergences.** By coverage, the two least-shared elements are Full
Capability Elicitation (7 of 12) and Conditions for Halting Development
(8 of 12). METR's own text names two differences:
- NVIDIA's and Cohere's frameworks emphasise domain-specific risk rather than
  catastrophic risk. Both are among the 3 policies with no Capability
  Thresholds (Naver, Cohere, NVIDIA).
- xAI's and Magic's policies lean on quantitative benchmarks.

**FINDING: the three labs the phase compares cannot show a divergence.**
Anthropic, OpenAI and Google DeepMind, the rows of Lesson 18's `LABS`, each
have 9 of 9 elements. Every METR divergence is between them and the other 9
companies. At METR's resolution, a three-lab comparison is all convergence.

**FINDING: the phase cites this report for three different things, and the
report contains two of them 0 times.** Lessons 18 and 28 describe the link as
a framework comparison, which is correct. Lesson 8 calls it the "three-pillar
framework in context", but the report contains "pillar" 0 times and
"scheming" 0 times. Lesson 10 labels it "UK AISI + METR — Control safety
cases", but the report never names UK AISI. Lesson 28's own METR section
names 0 of the nine elements.

Structure: `PRESENCE` is METR's table; `coverage()` counts it; the lesson
checks parse each lesson's Further Reading line for the report's URL.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "28-alignment-research-ecosystem"
URL = "https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/"
CITING = {"08": "08-in-context-scheming-frontier-models", "10": "10-ai-control-subversion",
          "18": "18-frontier-safety-frameworks-rsp-pf-fsf", "28": LESSON}
ALL = ("Anthropic", "OpenAI", "Google DeepMind", "Magic", "Naver", "Meta", "G42", "Cohere",
       "Microsoft", "Amazon", "xAI", "NVIDIA")
# METR's "Common Element Presence" table, 16 Dec 2025 version, read 2026-09-27
PRESENCE = {
    "Capability Thresholds": set(ALL) - {"Naver", "Cohere", "NVIDIA"},
    "Model Weight Security": set(ALL) - {"Naver"},
    "Model Deployment Mitigations": set(ALL),
    "Conditions for Halting Deployment Plans": set(ALL) - {"Magic", "Cohere", "NVIDIA"},
    "Conditions for Halting Development Plans": set(ALL) - {"Naver", "Cohere", "Amazon", "xAI"},
    "Full Capability Elicitation During Evaluations":
        {"Anthropic", "OpenAI", "Google DeepMind", "Meta", "G42", "Microsoft", "Amazon"},
    "Timing and Frequency of Evaluations": set(ALL) - {"Cohere", "xAI", "NVIDIA"},
    "Accountability": set(ALL),
    "Updating Policies Over Time": set(ALL),
}
DOMAIN_SPECIFIC = ("NVIDIA", "Cohere")      # METR's summary: domain-specific, not only catastrophic
# occurrences in the report's text on 2026-09-27
REPORT_COUNTS = {"pillar": 0, "scheming": 0, "UK AISI": 0, "AI Security Institute": 0}
COMPANY = {"Anthropic": "Anthropic", "OpenAI": "OpenAI", "DeepMind": "Google DeepMind"}


def coverage():
    return {element: len(firms) for element, firms in PRESENCE.items()}


def link_label(lesson):
    line = next(x for x in parity.doc_text(PHASE, lesson).splitlines() if URL in x)
    return re.match(r"- \[(.+?)\]\(\S+\) — (.+)", line).groups()


def solve():
    ref18 = parity.load_reference(PHASE, CITING["18"], "main")
    doc = parity.doc_text(PHASE, LESSON)
    counts = coverage()
    labs = [COMPANY[lab["name"].split()[0]] for lab in ref18.LABS]
    metr = doc.split("### METR")[1].split("\n### ")[0]
    return {
        "universal": [e for e, n in counts.items() if n == len(ALL)],
        "sparsest": sorted(counts.items(), key=lambda kv: kv[1])[:2],
        "no_thresholds": sorted(set(ALL) - PRESENCE["Capability Thresholds"]),
        "labs": {lab: sum(lab in firms for firms in PRESENCE.values()) for lab in labs},
        "labels": {k: link_label(lesson) for k, lesson in CITING.items()},
        "elements_named": [e for e in PRESENCE if e.lower() in metr.lower()],
    }


def verify(result):
    r, labels = result, result["labels"]
    return [
        practice.Check(
            "ANSWER: deployment mitigations, accountability, updating are the 3 universal elements",
            r["universal"] == ["Model Deployment Mitigations", "Accountability",
                               "Updating Policies Over Time"],
            f"present in all {len(ALL)} policies: {r['universal']}",
        ),
        practice.Check(
            "ANSWER: elicitation (7/12) and halting development (8/12) diverge most",
            r["sparsest"] == [("Full Capability Elicitation During Evaluations", 7),
                              ("Conditions for Halting Development Plans", 8)]
            and set(DOMAIN_SPECIFIC) <= set(r["no_thresholds"]) and len(r["no_thresholds"]) == 3,
            f"least shared: {r['sparsest']}; no capability thresholds: {r['no_thresholds']}",
        ),
        practice.Check(
            "FINDING: the three labs the phase compares cannot show a divergence",
            r["labs"] == {"Anthropic": 9, "OpenAI": 9, "Google DeepMind": 9},
            f"elements per Lesson 18 LABS row: {r['labs']}",
        ),
        practice.Check(
            "FINDING: the phase cites this report for three different things",
            "three-pillar" in labels["08"][1] and "Control safety cases" in labels["10"][0]
            and "UK AISI" in labels["10"][0] and "comparison" in labels["18"][1]
            and "comparison" in labels["28"][1] and r["elements_named"] == [],
            f"link labels {labels}; report text counts {REPORT_COUNTS}; elements named in "
            f"Lesson 28's METR section: {r['elements_named']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
