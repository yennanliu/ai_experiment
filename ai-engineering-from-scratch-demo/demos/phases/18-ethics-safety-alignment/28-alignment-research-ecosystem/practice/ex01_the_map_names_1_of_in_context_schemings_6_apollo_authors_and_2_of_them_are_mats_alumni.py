"""Exercise 1 — the map names 1 of In-Context Scheming's 6 Apollo authors, and 2 of them are MATS alumni.

    Pick one paper from Lessons 7-15 and identify the organisations involved.
    Cross-check the authors against MATS alumni and current ecosystem
    affiliations.

Reading of the exercise: the paper is In-Context Scheming (Lesson 8,
arXiv:2412.04984), because it is the one the lesson's own `ECOSYSTEM` table
cites. The organisations come from Lesson 8's page and the arXiv author list.
The MATS cross-check uses two public sources, read 2026-09-27: MATS's alumni
page (matsprogram.org/alumni) and the authors' own pages. Current affiliation
means being listed on apolloresearch.ai/team that day. The same check is then
run on what the lesson itself can supply: which of the six names it carries,
and whether its org attributions for Lessons 7-9 match those lessons' pages.

**ANSWER: one organisation, Apollo Research, and six authors.** Lesson 8's
header lists Meinke, Schoen, Scheurer, Balesni, Shah, Hobbhahn (Apollo
Research). That matches arXiv's author order exactly, and `ECOSYSTEM` cites
the same arXiv id for Apollo. Against MATS: Marius Hobbhahn is on MATS's
alumni page ("Apollo almost certainly would not have happened without
MATS"). Mikita Balesni's own site says he was a MATS scholar with Owain
Evans, and the alumni page does not list him. So at least 2 of 6 are MATS
alumni, and the official page shows only 1. Against current affiliations: 5
of 6 are on Apollo's team page. Balesni, a founding member, is not.

**FINDING: the lesson cannot run this cross-check.** It names 1 of the 6
authors (Meinke, in `ECOSYSTEM`) and no MATS alumnus at all. Its talent
pipeline is drawn as MATS -> orgs, but the paper's senior author came
through MATS and then founded the organisation.

**FINDING: the lesson's multi-org list includes a single-org paper, and
credits Lesson 7 with orgs that Lesson 7's page never names.** "The
multi-org structure is the quality control" is illustrated by 4 papers. In
one of them, In-Context Scheming, "was Apollo" alone. Its skill file would
hard-reject that paper ("Any single-org safety claim without an external
replication or check"). It calls Sleeper Agents "Anthropic + Redwood", but
Lesson 7's header names neither organisation, and its page mentions Redwood
0 times. Lessons 8 and 9 do match their attributions.

Structure: `header_orgs()` reads a lesson's blockquote header;
`attributions()` parses "X was A + B" out of "Why this layer matters".
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "28-alignment-research-ecosystem"
SOURCES = {"Sleeper Agents": "07-sleeper-agents-persistent-deception",
           "In-Context Scheming": "08-in-context-scheming-frontier-models",
           "Alignment Faking": "09-alignment-faking"}
ORGS = ("Anthropic", "Redwood", "Apollo", "OpenAI")
# arXiv:2412.04984 abstract page, read 2026-09-27
ARXIV_AUTHORS = ("Alexander Meinke", "Bronson Schoen", "Jérémy Scheurer", "Mikita Balesni",
                 "Rusheb Shah", "Marius Hobbhahn")
# matsprogram.org/alumni (featured profiles) and mikitabalesni.com, read 2026-09-27
MATS_ALUMNI_PAGE = ("Marius Hobbhahn",)
MATS_SELF_REPORTED = ("Mikita Balesni",)
# apolloresearch.ai/team, read 2026-09-27
APOLLO_TEAM_PAGE = ("Alexander Meinke", "Bronson Schoen", "Jérémy Scheurer", "Rusheb Shah",
                    "Marius Hobbhahn")
SKILL_RULE = "Any single-org safety claim without an external replication or check."


def header(lesson):
    return next(line for line in parity.doc_text(PHASE, lesson).splitlines() if line.startswith("> "))


def header_orgs(lesson):
    return [org for org in ORGS if org in header(lesson)]


def attributions(doc):
    block = doc.split("### Why this layer matters")[1].split("\n### ")[0]
    pairs = re.findall(r"([A-Z][\w-]+(?: [A-Z][\w-]+)*) (?:paper \(Lesson \d+\) )?was (\w+(?: \+ \w+)*)",
                       block)
    return {paper: orgs.split(" + ") for paper, orgs in pairs}


def named(names, text):
    return [n for n in names if n in text]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    apollo = next(o for o in ref.ECOSYSTEM if o["org"] == "Apollo")
    mats = doc.split("### MATS")[1].split("\n### ")[0] + str(ref.ECOSYSTEM[0])
    ics = header(SOURCES["In-Context Scheming"])
    surname = [a.split()[-1] for a in ARXIV_AUTHORS]
    claimed = attributions(doc)
    return {
        "lesson8_authors": re.match(r"> ([^(]+) \(", ics).group(1).split(", "),
        "lesson8_ids": (re.search(r"arXiv:([\d.]+)", ics).group(1),
                        re.search(r"arXiv:([\d.]+)", apollo["canonical_output"]).group(1)),
        "arxiv_surnames": surname,
        "named_in_lesson": named(surname, doc + str(ref.ECOSYSTEM)),
        "mats_names_in_lesson": named(surname, mats),
        "claimed": claimed,
        "sources": {p: header_orgs(lesson) for p, lesson in SOURCES.items()},
        "lesson7_redwood": parity.doc_text(PHASE, SOURCES["Sleeper Agents"]).count("Redwood"),
        "single_org": [p for p, orgs in claimed.items() if len(orgs) == 1],
        "skill_rule": SKILL_RULE in (parity.lesson_dir(PHASE, LESSON) / "outputs"
                                     / "skill-ecosystem-map.md").read_text(),
    }


def verify(result):
    r = result
    mats = named(ARXIV_AUTHORS, MATS_ALUMNI_PAGE + MATS_SELF_REPORTED)
    left = sorted(set(ARXIV_AUTHORS) - set(APOLLO_TEAM_PAGE))
    mismatched = dict(filter(lambda kv: kv[1][0] != kv[1][1],
                             ((p, (r["claimed"][p], got)) for p, got in r["sources"].items())))
    return [
        practice.Check(
            "ANSWER: Apollo only, six authors; at least 2 of 6 MATS alumni, 5 of 6 still at Apollo",
            all([r["lesson8_authors"] == r["arxiv_surnames"],
                 r["sources"]["In-Context Scheming"] == ["Apollo"],
                 r["lesson8_ids"] == ("2412.04984", "2412.04984"), len(mats) == 2,
                 named(mats, MATS_ALUMNI_PAGE) == ["Marius Hobbhahn"], left == ["Mikita Balesni"]]),
            f"Lesson 8 header {r['lesson8_authors']} == arXiv order; ids {r['lesson8_ids']}; "
            f"MATS alumni {mats} (official page shows {list(MATS_ALUMNI_PAGE)}); not on Apollo's "
            f"team page: {left}",
        ),
        practice.Check(
            "FINDING: the lesson cannot run this cross-check",
            (r["named_in_lesson"], r["mats_names_in_lesson"]) == (["Meinke"], []),
            f"authors the lesson names: {r['named_in_lesson']} of 6; in its MATS section: "
            f"{r['mats_names_in_lesson']}",
        ),
        practice.Check(
            "FINDING: a single-org paper in the multi-org list, and Lesson 7 names no org",
            all([len(r["claimed"]) == 4, r["single_org"] == ["In-Context Scheming"], r["skill_rule"],
                 mismatched == {"Sleeper Agents": (["Anthropic", "Redwood"], [])},
                 r["lesson7_redwood"] == 0]),
            f"lesson's attributions {r['claimed']}; source headers {r['sources']}; mismatches "
            f"{mismatched}; 'Redwood' on Lesson 7's page: {r['lesson7_redwood']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
