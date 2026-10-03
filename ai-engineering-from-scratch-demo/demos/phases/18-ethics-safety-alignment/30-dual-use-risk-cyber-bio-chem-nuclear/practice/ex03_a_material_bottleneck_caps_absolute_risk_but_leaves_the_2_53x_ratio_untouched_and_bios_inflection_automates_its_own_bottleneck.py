"""Exercise 3 — a material bottleneck caps absolute risk but leaves the 2.53x ratio untouched, and bio's inflection automates its own bottleneck.

    Nuclear uplift appears bounded by material access. Argue for and against
    the position that a future AI breakthrough could shift this bottleneck.

Reading of the exercise: "bounded" is tested on the smallest model where
the claim can hold. An attempt passes two stages in series, an
informational stage with success p_info and a material-access stage with
success p_mat. AI multiplies one stage's success by u, capped at 1. The
lesson's own terms are then read off `DOMAINS`, Key Terms, `main()`, and
Lesson 17's evaluation domains. All stage values are abstract; the model
contains no domain content.

**ANSWER, against: the bound is real for absolute success.** With AI acting
only on information (p_info 0.2, u = 2.53, the lesson's bio figure), success
can never exceed p_mat. Even unlimited information help reaches exactly
p_mat, which is 0.0001 when p_mat = 1e-4.

**ANSWER, for: a breakthrough only has to reach the acquisition stage.**
If AI also multiplies p_mat by v = 2, the ceiling itself moves, and the
uplift becomes 2.53 x 2 = 5.06x. The lesson's own table names the mechanism.
Bio's 2025 inflection is "acquisition-phase automation", and Key Terms
define the acquisition phase as procurement, equipment and permit stages.
Bio's `bottleneck_remaining` names procurement and equipment, the same
stages its inflection automates. Nuclear's reads "fissile-material
acquisition dominates". So the "for" case is: whatever moved bio's
acquisition phase would be aimed at nuclear's.

**FINDING: "bounded by material access" bounds absolute risk, not
novice-relative uplift.** Across p_mat from 1e-1 to 1e-4 the relative uplift
stays 2.53x, while the absolute gain falls from 0.0306 to 0.00003. The
lesson reports bio as a 2.53x ratio and nuclear as "limited". On the
ratio, a nuclear attempt with the same information help would read 2.53x
too. The difference lies in absolute risk, the quantity it uses only for
experts.

**FINDING: the four domains are not CBRN.** `DOMAINS` holds bio, chem,
cyber and nuclear. Three are CBRN letters, radiological is missing and
cyber is not one. `main()` prints "three of four CBRN domains crossed
thresholds", but the three are bio, chem and cyber, so 2 of the 3 CBRN
domains present crossed. Lesson 17's harness, which the page calls the
measurement methodology, has 0 nuclear or radiological domains.

Structure: `success()` is the two-stage model; `lesson_terms()` reads the
reference table, Key Terms and printed takeaway.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE = "18-ethics-safety-alignment"
LESSON, WMDP = "30-dual-use-risk-cyber-bio-chem-nuclear", "17-wmdp-dual-use-evaluation"
P_INFO, U, V = 0.2, 2.53, 2.0
P_MATS = (1e-1, 1e-2, 1e-3, 1e-4)
CBRN = {"chem", "bio", "rad", "nuclear"}
ACQUISITION = {
    "bio_inflection": "acquisition-phase automation",
    "bio_bottleneck": "pathogen procurement, biosafety equipment",
    "acquisition_term": "Procurement, equipment, permit stages of a bio threat",
    "nuclear_bottleneck": "fissile-material acquisition dominates",
}
DOMAIN_SETS = {
    "domains": ["bio", "chem", "cyber", "nuclear"],
    "cbrn_present": ["bio", "chem", "nuclear"],
    "crossed": ["bio", "chem", "cyber"],
    "cbrn_crossed": ["bio", "chem"],
    "wmdp_nuclear": [],
}


def success(p_mat, u_info=1.0, v_mat=1.0):
    return min(1.0, P_INFO * u_info) * min(1.0, p_mat * v_mat)


def sweep():
    """p_mat -> (relative uplift, absolute gain) with AI on information only."""
    return {
        p: (round(success(p, U) / success(p), 2), round(success(p, U) - success(p), 5))
        for p in P_MATS
    }


def lesson_terms(ref, doc, wmdp):
    rows = {d["domain"]: d for d in ref.DOMAINS}
    acquisition = doc.split("| Acquisition phase |")[1].split("\n")[0]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.main()
    return {
        "bio_inflection": rows["bio"]["inflection"],
        "bio_bottleneck": rows["bio"]["bottleneck_remaining"],
        "nuclear_bottleneck": rows["nuclear"]["bottleneck_remaining"],
        "acquisition_term": acquisition.split("|")[1].strip(),
        "domains": sorted(rows),
        "crossed": sorted(d for d, r in rows.items() if r["2025_state"] != "limited"),
        "prints_cbrn": "three of four CBRN domains crossed thresholds"
        in out.getvalue(),
        "wmdp_nuclear": [d for d in wmdp.DOMAINS if "nucl" in d or "radio" in d],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    terms = lesson_terms(
        ref, parity.doc_text(PHASE, LESSON), parity.load_reference(PHASE, WMDP, "main")
    )
    return {
        "sweep": sweep(),
        "ceiling": success(P_MATS[-1], u_info=1e9),
        "both": round(success(1e-3, U, V) / success(1e-3), 2),
        "terms": terms,
        "cbrn_present": sorted(CBRN & set(terms["domains"])),
        "cbrn_crossed": sorted(CBRN & set(terms["crossed"])),
    }


def verify(result):
    r, t = result, result["terms"]
    rel = {p: v[0] for p, v in r["sweep"].items()}
    gain = {p: v[1] for p, v in r["sweep"].items()}
    return [
        practice.Check(
            "ANSWER, against: information help alone never lifts success past p_mat",
            r["ceiling"] == P_MATS[-1],
            f"success with unlimited information help at p_mat {P_MATS[-1]}: {r['ceiling']}",
        ),
        practice.Check(
            "ANSWER, for: acting on acquisition multiplies the uplift to 5.06x; bio's row shows how",
            r["both"] == 5.06 and {k: t[k] for k in ACQUISITION} == ACQUISITION,
            f"bio inflection {t['bio_inflection']!r}, bottleneck {t['bio_bottleneck']!r}; "
            f"Key Terms {t['acquisition_term']!r}; nuclear {t['nuclear_bottleneck']!r}",
        ),
        practice.Check(
            "FINDING: the ratio stays 2.53x at every p_mat while the absolute gain falls 1000x",
            set(rel.values()) == {U} and (gain[1e-1], gain[1e-4]) == (0.0306, 3e-05),
            f"relative uplift {rel}; absolute gain {gain}",
        ),
        practice.Check(
            "FINDING: the four domains are not CBRN; 2 of the 3 CBRN present crossed",
            {k: {**t, **r}[k] for k in DOMAIN_SETS} == DOMAIN_SETS and t["prints_cbrn"],
            f"DOMAINS {t['domains']}; crossed {t['crossed']}; CBRN crossed {r['cbrn_crossed']}; "
            f"Lesson 17 nuclear/radiological domains {t['wmdp_nuclear']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
