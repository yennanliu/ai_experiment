"""Exercise 1 — open-source GPAI skips the Transparency chapter the lesson says binds everyone, and 10 of 12 commitments are systemic-only.

    Read the EU AI Act (regulation 2024/1689) and the GPAI Code of Practice
    (10 July 2025). Identify three obligations that apply to every GPAI
    provider and three that apply only to systemic-risk GPAI.

Reading of the exercise: "every GPAI provider" is read literally, as the
obligations that survive for all four provider profiles the Act
distinguishes (closed or open-source, with or without systemic risk). Art 53(2)
exempts free and open-source models without systemic risk from Art 53(1)(a)
and (b). The obligations come from the regulation text (Arts 51, 53 and 55,
read 2026-09-27 at artificialintelligenceact.eu). The Code's chapter sizes
come from the final Code of 10 July 2025. Each chapter is mapped to the
article it implements, and the result is set against the lesson's own
description of the Code.

**ANSWER, every GPAI provider (3 of 3 survive all four profiles):** a
copyright policy that respects text-and-data-mining opt-outs (Art 53(1)(c)),
a public summary of training content on the AI Office template
(Art 53(1)(d)), and cooperation with the Commission and national authorities
(Art 53(3)). Technical documentation for the AI Office (a) and for downstream
providers (b) bind 3 of the 4 profiles. **Systemic-risk only (4 of 4 Art 55
duties):** model evaluation with adversarial testing, assessment and
mitigation of systemic risk at Union level, serious-incident reporting to the
AI Office, and cybersecurity for the model and its physical infrastructure.

**FINDING: the lesson's Transparency chapter does not bind every GPAI
provider.** The page says Transparency and Copyright bind "All GPAI
providers". Transparency implements Art 53(1)(a)-(b), so it binds 3 of the 4
profiles and skips open-source models without systemic risk. Copyright binds
4 of 4.

**FINDING: the lesson's numbers are right, and they show where the weight
sits.** Its 12 commitments (page and main.py) are 1 + 1 + 10, so 10 of 12 bind
only systemic-risk providers. Its >1e25 FLOP threshold is Art 51(2)'s 10^25.
Signing the Code is voluntary (Art 56); the Act binds, the Code only shows
how.

Structure: `ART53`/`ART55` hold the duties with the open-source exemption
flag; `obligations()` applies them to a profile; `lesson_code()` reads the
page's chapter list, commitment count and threshold.
"""

from __future__ import annotations

import itertools
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "24-regulatory-frameworks-eu-us-uk-korea"
# Regulation 2024/1689, read 2026-09-27: (article, duty, exempt for open-source non-systemic models)
ART53 = [
    ("53(1)(a)", "technical documentation (Annex XI) for the AI Office", True),
    ("53(1)(b)", "documentation for downstream providers (Annex XII)", True),
    ("53(1)(c)", "copyright policy respecting TDM opt-outs", False),
    ("53(1)(d)", "public summary of training content", False),
    ("53(3)", "cooperation with the Commission and national authorities", False),
]
ART55 = [("55(1)(a)", "evaluation incl. adversarial testing"), ("55(1)(b)", "systemic-risk mitigation"),
         ("55(1)(c)", "serious-incident reporting"), ("55(1)(d)", "cybersecurity")]
SYSTEMIC_FLOP = 1e25                                            # Art 51(2)
# GPAI Code of Practice, 10 July 2025: chapter -> (commitments, articles it implements)
CODE = {"Transparency": (1, ("53(1)(a)", "53(1)(b)")), "Copyright": (1, ("53(1)(c)",)),
        "Safety and Security": (10, tuple(a for a, _ in ART55))}
PROFILES = list(itertools.product((False, True), repeat=2))    # (open_source, systemic)


def obligations(open_source, systemic):
    kept = [a for a, _, exempt in ART53 if not (exempt and open_source and not systemic)]
    return kept + ([a for a, _ in ART55] if systemic else [])


def lesson_code(ref):
    doc = parity.doc_text(PHASE, LESSON)
    section = doc.split("### GPAI Code of Practice")[1].split("\n### ")[0]
    event = next(e for d, e in ref.TIMELINE if "Code of Practice published" in e)
    return {
        "chapters": dict(re.findall(r"^- \*\*([^.]+)\.\*\* ([^.(]+)", section, re.M)),
        "commitments": int(re.search(r"(\d+) commitments total", section).group(1)),
        "timeline_commitments": int(re.search(r"(\d+) commitments", event).group(1)),
        "threshold": float(re.search(r"\(>(\S+) FLOP", doc).group(1)),
    }


def split(sets):
    """Duties held by every profile, and duties held only by systemic-risk profiles."""
    every = [a for a, _, _ in ART53 if all(a in s for s in sets.values())]
    plain = set(sets[(False, False)]) | set(sets[(True, False)])
    return every, [a for a in sets[(False, True)] if a not in plain]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sets = {p: obligations(*p) for p in PROFILES}
    every, systemic_only = split(sets)
    binds = {ch: sum(bool(set(arts) & set(s)) for s in sets.values()) for ch, (_, arts) in CODE.items()}
    return {"sizes": {str(p): len(s) for p, s in sets.items()}, "every": every,
            "systemic_only": systemic_only, "binds": binds, "lesson": lesson_code(ref),
            "code_total": sum(n for n, _ in CODE.values()), "code_systemic": CODE["Safety and Security"][0]}


def verify(result):
    r, lesson = result, result["lesson"]
    chapters = lesson["chapters"]
    return [
        practice.Check(
            "ANSWER: three duties bind every profile, four bind only systemic-risk providers",
            (r["every"], r["systemic_only"]) == (["53(1)(c)", "53(1)(d)", "53(3)"], [a for a, _ in ART55])
            and list(chapters) == list(CODE),
            f"every profile: {r['every']}; systemic only: {r['systemic_only']}; duties per "
            f"(open_source, systemic) profile {r['sizes']}; lesson chapters {list(chapters)}",
        ),
        practice.Check(
            "FINDING: the lesson's Transparency chapter does not bind every GPAI provider",
            chapters["Transparency"].strip() == "All GPAI providers"
            and (r["binds"]["Transparency"], r["binds"]["Copyright"]) == (3, 4),
            f"lesson: {chapters}; profiles each chapter binds, of 4: {r['binds']}",
        ),
        practice.Check(
            "FINDING: the lesson's numbers are right: 12 = 1 + 1 + 10, threshold 1e25",
            (lesson["commitments"], lesson["timeline_commitments"], r["code_systemic"])
            == (r["code_total"], 12, 10) and lesson["threshold"] == SYSTEMIC_FLOP
            and chapters["Safety and Security"].startswith("Systemic-risk"),
            f"page {lesson['commitments']}, main.py {lesson['timeline_commitments']}, Code "
            f"{r['code_total']} of which {r['code_systemic']} systemic-only; page threshold "
            f"{lesson['threshold']:g} vs Art 51(2) {SYSTEMIC_FLOP:g}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
