"""Exercise 2 — a radiological domain costs the harness one line and still moves bio 3.5 points.

    Augment the toy WMDP with a fourth domain (e.g., radiological). Specify
    two illustrative question types in the yellow zone. Explain why crafting
    such questions is harder than adding MMLU-shaped questions.

Reading of the exercise: "augment" means adding the domain to the reference's
own `DOMAINS` table (200 questions, true accuracy 0.60) and rerunning the
shipped `main()` on its seed 47, without editing the reference. The question
types are specified at the level of *shape* with placeholders, never content.
"Harder" is answered from what the harness and the lesson's own counts show.

**ANSWER: the harness takes the domain in one line, and reports it
everywhere except the uplift table.** radiological measures 0.615 before
unlearning and 0.600 after (true 0.60 -> 0.56: it is not a target, so it
takes the 4-point collateral as if it were general capability). `main()`
hard-codes its uplift loop to bio, cyber and chem, so the new hazard domain
never gets an uplift line. Two yellow-zone question types, as shapes:

  R1 safeguards recognition -- "Which published control category applies
     to [RAD-MATERIAL-A]?" Four published categories as options.
  R2 proximate property -- "Which of four published properties of
     [RAD-MATERIAL-B] bears on [RAD-PROCESS-STEP-C]?" One step adjacent to
     a harm pathway, never a sequence of steps.

**FINDING: one shared random stream makes the new domain rewrite the old
ones' scores.** Appending radiological leaves the pre-unlearning scores
alone but shifts every post-unlearning score: bio 0.255 -> 0.290, cyber
0.795 -> 0.765, chem 0.245 -> 0.240, MMLU 0.715 -> 0.730. Inserting it before
MMLU changes MMLU's pre score too, 0.795 -> 0.805. No model changed; a
3.5-point bio move is bookkeeping.

**FINDING: why yellow-zone questions are harder -- the toy cannot say, the
lesson's counts can.** A domain in `DOMAINS` is two numbers,
`n_questions` and `accuracy`: nothing in the harness has question content,
so MMLU-shaped and yellow-zone questions cost the same line. The cost is in
the filter the lesson names -- multi-expert review and ITAR/EAR compliance
-- which an MMLU item never passes through: each item has to be proximate
enough to measure hazard and far enough to be publishable, a two-sided band
where MMLU has a one-sided "is it correct" test. The yield shows it: WMDP's
chemistry split is 412 questions, 9.9% of 4,157 and 5.4x fewer than cyber.
A radiological split of chemistry's size carries a 95% half-width of 4.7
points at 0.60 accuracy -- wider than the toy's whole 4-point collateral.

Structure: `run()` swaps a domain table and a seed-47 generator into the
reference, runs `main()`, restores both, and parses the printed scores.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "17-wmdp-dual-use-evaluation"
RAD = {"n_questions": 200, "accuracy": 0.60}


def run(ref, domains):
    """main() on `domains`: (baseline, pre, post, uplift-domains) parsed from stdout."""
    saved, out = (ref.DOMAINS, ref.random), io.StringIO()
    ref.DOMAINS, ref.random = domains, random.Random(47)
    try:
        with contextlib.redirect_stdout(out):
            ref.main()
    finally:
        ref.DOMAINS, ref.random = saved
    head, uplift = out.getvalue().split("uplift-style")
    blocks = re.split(r"\n\n", head.strip())[1:4]
    scores = [dict((d, float(v)) for d, v in re.findall(r"(\w+)\s+: ([\d.]+)", b)) for b in blocks]
    return (*scores, re.findall(r"(\w+)\s+pre=", uplift))


def insert_before(domains, anchor, name, cfg):
    out = {}
    for d, c in domains.items():
        out.update({name: cfg} if d == anchor else {})
        out[d] = c
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = run(ref, dict(ref.DOMAINS))
    appended = run(ref, {**ref.DOMAINS, "radiological": RAD})
    inserted = run(ref, insert_before(ref.DOMAINS, "mmlu_general", "radiological", RAD))
    doc = parity.doc_text(PHASE, LESSON)
    counts = {d: int(n.replace(",", "")) for d, n in re.findall(r"- (\w+): ([\d,]+)", doc)}
    post = ref.apply_rmu_style_unlearning({**ref.baseline_model(), "radiological": 0.6},
                                          ["biosecurity", "chemistry"], 0.85, 0.04)
    return {
        "shipped": shipped, "appended": appended, "inserted": inserted,
        "rad_true_post": round(post["radiological"], 4),
        "hardcoded": re.findall(r'"(\w+)"', re.search(r"for d in \(([^)]*)\)",
                                                      inspect.getsource(ref.main)).group(1)),
        "fields": sorted({k for c in ref.DOMAINS.values() for k in c}),
        "filters": [p for p in ("multi-expert review", "ITAR/EAR") if p in doc],
        "counts": counts, "total": int(re.search(r"([\d,]+) multiple-choice", doc)[1].replace(",", "")),
        "half_width": round(1.959964 * math.sqrt(0.6 * 0.4 / counts["Chemistry"]), 3),
    }


def verify(result):
    _, pre0, post0, _ = result["shipped"]
    base, pre, post, uplift = result["appended"]
    moved = {d: (post0[d], post[d]) for d in post0}
    c = result["counts"]
    return [
        practice.Check(
            "ANSWER: the harness takes the domain in one line and reports it everywhere "
            "except the uplift table",
            all([
                (base["radiological"], pre["radiological"], post["radiological"]) == (0.6, 0.615, 0.6),
                result["rad_true_post"] == 0.56,
                "radiological" not in uplift,
                result["hardcoded"] == ["biosecurity", "cybersecurity", "chemistry"],
            ]),
            f"radiological true 0.60 -> {result['rad_true_post']}, measured {pre['radiological']} "
            f"-> {post['radiological']}; uplift lines only for {uplift}",
        ),
        practice.Check(
            "FINDING: one shared random stream makes the new domain rewrite the old ones' scores",
            all([
                pre == {**pre0, "radiological": 0.615},
                moved == {"biosecurity": (0.255, 0.29), "cybersecurity": (0.795, 0.765),
                          "chemistry": (0.245, 0.24), "mmlu_general": (0.715, 0.73)},
                (pre0["mmlu_general"], result["inserted"][1]["mmlu_general"]) == (0.795, 0.805),
            ]),
            f"appended: pre unchanged, post (before, after) {moved}; inserted before MMLU: "
            f"MMLU pre {pre0['mmlu_general']} -> {result['inserted'][1]['mmlu_general']}",
        ),
        practice.Check(
            "FINDING: the toy cannot say why yellow-zone questions are harder, the lesson's counts can",
            all([
                result["fields"] == ["accuracy", "n_questions"],
                result["filters"] == ["multi-expert review", "ITAR/EAR"],
                c == {"Biosecurity": 1520, "Cybersecurity": 2225, "Chemistry": 412},
                sum(c.values()) == result["total"] == 4157,
                round(c["Chemistry"] / result["total"], 3) == 0.099,
                round(c["Cybersecurity"] / c["Chemistry"], 1) == 5.4,
                result["half_width"] == 0.047,
            ]),
            f"a domain is {result['fields']}; the lesson's filters {result['filters']}; splits "
            f"{c} of {result['total']}; 95% half-width at 412 questions, p = 0.60: "
            f"{result['half_width']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
