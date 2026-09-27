"""Exercise 3 — the rename keeps 4 of 16 named harms and drops all 6 societal ones, though free speech was never in the remit.

    The UK AI Security Institute's rename narrows scope. Argue for and against
    the narrower framing. Identify the policy assumption each position depends
    on.

Reading of the exercise: "narrows scope" is measured before it is argued. The
old remit is the harms named in the four evaluation areas of *Introducing the
AI Safety Institute* (gov.uk, November 2023). The new remit is the focus
sentence of the rename announcement (gov.uk, 14 February 2025). Both were read
2026-09-27. Each old harm counts as kept if its key term appears in the new
sentence. The for and against positions are then argued from what was
dropped.

**ANSWER, for:** the kept harms (chemical and biological weapons, cyber
attacks) are ones no other UK body evaluates on frontier models before
release, so a small institute should spend its model access there. *The
assumption:* the dropped harms have other owners, such as equality,
data-protection and online-safety regulators, and the outside bodies the 2023
remit already leaned on for societal work.

**ANSWER, against:** the 2023 remit said societal harms need "both pre and
post-deployment evaluations", and pre-deployment access sits with the
institute. Dropping them leaves those harms with regulators who only see
deployed systems. *The assumption:* manipulation and disinformation are
separable from security, although the 2023 remit filed disinformation and
persuasion under dual-use.

**FINDING: the narrowing is wider than the lesson says.** The new sentence
keeps 4 of the 16 named harms: 3 of 5 dual-use, 1 of 2 system-security, 0 of
6 societal and 0 of 3 loss-of-control. It adds 2 harms the old remit never
named, fraud and child sexual abuse. The lesson says the rename "drops
algorithmic bias and free-speech framings". But "freedom of speech" appears 0
times in the 2023 remit, so there was no free-speech framing to drop. The cut
is the whole societal-impacts area, plus persuasion and disinformation.

**FINDING: the reference timeline cannot place the rename.** 7 of TIMELINE's
14 dates carry day "00". Sorting them as strings moves the rename (actually
14 Feb 2025) ahead of the 2 Feb EU prohibitions; those 2 entries swap.
The Paris summit, which the rename followed "just days after", is cited on the
page and absent from TIMELINE. Two EU dates have also moved: the AI Omnibus
(in force 27 July 2026) put Annex III high-risk at 2 Dec 2027, 16 months after
TIMELINE's 2 Aug 2026, and embedded high-risk at 2 Aug 2028, 12 months after
TIMELINE's 2 Aug 2027.

Structure: `OLD` holds the 2023 harms with their key terms, `NEW` the 2025
focus sentence; `kept()` matches them; `timeline()` audits the reference
TIMELINE.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "24-regulatory-frameworks-eu-us-uk-korea"
# gov.uk "Introducing the AI Safety Institute" (Nov 2023): area -> [(named harm, key term)]
OLD = {
    "dual-use": [("cyber-criminality", "cyber"), ("biological or chemical science", "biological|chemical"),
                 ("human persuasion", "persua"), ("large-scale disinformation campaigns", "disinformation"),
                 ("weapons acquisition", "weapon")],
    "societal": [("psychological impacts", "psycholog"), ("privacy harms", "privacy"),
                 ("manipulation and persuasion", "manipulat|persua"), ("biased outputs and reasoning", "bias"),
                 ("impacts on democracy and trust in institutions", "democra"),
                 ("systemic discrimination", "discriminat")],
    "system security": [("efficacy and limitations of system safeguards", "safeguard"),
                        ("adequacy of cybersecurity measures", "cyber")],
    "loss of control": [("deceiving human operators", "deceiv"), ("autonomously replicating", "replicat"),
                        ("adapting to human attempts to intervene", "interven")],
}
OLD_TEXT_MENTIONS = {"freedom of speech": 0, "fraud": 0, "child sexual abuse": 0}   # whole 2023 page
# gov.uk rename announcement, 14 February 2025, the focus sentence verbatim
NEW = ("how the technology can be used to develop chemical and biological weapons, how it can be used "
       "to carry out cyber-attacks, and enable crimes such as fraud and child sexual abuse")
OMNIBUS = {"Annex III high-risk": ("2026-08-02", "2027-12-02"),       # AI Omnibus, in force 2026-07-27
           "embedded high-risk": ("2027-08-02", "2028-08-02")}


def kept():
    return {area: [h for h, key in harms if re.search(key, NEW)] for area, harms in OLD.items()}


def months(a, b):
    (y1, m1, _), (y2, m2, _) = (map(int, d.split("-")) for d in (a, b))
    return (y2 - y1) * 12 + m2 - m1


def timeline(ref, doc):
    dates = [d for d, _ in ref.TIMELINE]
    moved = [e for (d, e), s in zip(ref.TIMELINE, sorted(ref.TIMELINE)) if (d, e) != s]
    return {
        "n": len(dates), "day00": sum(d.endswith("-00") for d in dates),
        "moved": moved,
        "paris": ("Paris" in doc, any("Paris" in e for _, e in ref.TIMELINE)),
        "omnibus": {k: (a in dates, months(a, b)) for k, (a, b) in OMNIBUS.items()},
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    uk = doc.split("### UK AI Security Institute")[1].split("\n### ")[0]
    added = [c for c in ("fraud", "child sexual abuse") if c in NEW and OLD_TEXT_MENTIONS[c] == 0]
    k = kept()
    return {"kept": k, "counts": {a: (len(k[a]), len(OLD[a])) for a in OLD},
            "total": (sum(map(len, k.values())), sum(map(len, OLD.values()))), "added": added,
            "speech": [h for harms in OLD.values() for h, _ in harms if "speech" in h],
            "lesson_drops": re.search(r"drops ([^;]+);", uk).group(1),
            "timeline": timeline(ref, doc)}


def verify(result):
    r, t = result, result["timeline"]
    return [
        practice.Check(
            "FINDING: the narrowing is wider than the lesson says",
            (r["total"], r["counts"], r["added"], r["lesson_drops"], r["speech"])
            == ((4, 16), {"dual-use": (3, 5), "societal": (0, 6), "system security": (1, 2),
                          "loss of control": (0, 3)}, ["fraud", "child sexual abuse"],
                "algorithmic bias and free-speech framings", []),
            f"kept {r['total'][0]} of {r['total'][1]}: {r['kept']}; (kept, named) per area "
            f"{r['counts']}; added {r['added']}; lesson says it drops {r['lesson_drops']!r}; "
            f"mentions in the 2023 page {OLD_TEXT_MENTIONS}",
        ),
        practice.Check(
            "FINDING: the reference timeline cannot place the rename",
            (t["n"], t["day00"], t["paris"], [e[:6] for e in t["moved"]], t["omnibus"])
            == (14, 7, (True, False), ["EU AI ", "UK AIS"],
                {"Annex III high-risk": (True, 16), "embedded high-risk": (True, 12)}),
            f"{t['day00']} of {t['n']} dates end -00; string sort moves {t['moved']}; Paris on "
            f"page / in TIMELINE {t['paris']}; Omnibus (date in TIMELINE, months moved) {t['omnibus']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
