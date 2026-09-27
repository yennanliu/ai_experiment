"""Exercise 5 — a refusal gate outscores real unlearning on the toy's WMDP board, so ASL-3 needs two more studies.

    Articulate what a safety case for ASL-3 requires beyond passing WMDP
    unlearning. Name at least two complementary elicitation studies.

Reading of the exercise: "beyond passing WMDP unlearning" is made concrete
on the lesson's harness. The shipped unlearned model (bio + chem, strength
0.85, collateral 0.04) is set beside a model that was never unlearned but
whose policy gate answers bio and chem questions at chance -- the case the
lesson's skill file calls a refusal-path problem. Both are scored with the
reference `evaluate`, in the position `main()` scores its post model, on
seed 47 and over 1000 seeds. What the case needs is
then read off what the board cannot tell apart, and off the skill file's
own hard rejects.

**ANSWER: a WMDP pass cannot distinguish removed capability from gated
capability, so the case needs a refusal-path audit plus two elicitation
studies.** On seed 47 the gated model measures bio 0.255, chem 0.245 and
MMLU 0.785; the unlearned model measures bio 0.255, chem 0.245 and MMLU
0.715 -- identical on the hazard domains (same question draws), and the gate
looks 7 points *better* on general capability because it paid no
collateral. Over 1000 pairs of independent runs the gated model's measured
MMLU beats the unlearned one's on 79.3%.
Bypass the gate and bio is back at its true 0.72. The two studies:

  1. A novice-in-the-loop acquisition trial (the Anthropic-style study the
     skill file names): novice-relative uplift on tasks, not MCQ accuracy,
     run on the deployed stack.
  2. An expert maximum-elicitation study on the raw model -- gate bypassed,
     jailbreaks and fine-tuning allowed -- for the expert-absolute ceiling
     the lesson says a safety case must also bound.

**FINDING: 200 questions cannot certify "at chance".** The unlearned bio
score of 0.255 has a one-sided 95% upper bound of 0.306, 1.22x chance. At
WMDP's full 1,520 bio questions the same score bounds at 0.273, 1.09x. A
case has to state the bound, not the point score.

**FINDING: the toy's own claim fails one of the skill file's three hard
rejects and says nothing about cyber.** It reports per-domain WMDP-style
scores and an MMLU delta, which clears the first two; it has no
novice-in-the-loop study, which the third requires before any "no
meaningful uplift" claim. And cyber, never targeted, stays at 3.18x chance
after unlearning: a bio/chem pass is not a dual-use pass.

Structure: `board()` scores a model dict with a seeded generator swapped into
the reference; `upper()` is the normal one-sided bound on a proportion.
"""

from __future__ import annotations

import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "17-wmdp-dual-use-evaluation"
TARGETS, SEEDS, Z95 = ["biosecurity", "chemistry"], range(1000), 1.644854


def board(ref, model, seed, first=False):
    """Scores as main() measures them: its post model is the second evaluation on the stream."""
    saved, ref.random = ref.random, random.Random(seed)
    try:
        pre = ref.evaluate(ref.baseline_model())
        return pre if first else ref.evaluate(model)
    finally:
        ref.random = saved


def upper(p, n):
    return round(p + Z95 * math.sqrt(p * (1 - p) / n), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base = ref.baseline_model()
    unlearned = ref.apply_rmu_style_unlearning(base, TARGETS, strength=0.85, collateral=0.04)
    gated = {**base, **{t: 0.25 for t in TARGETS}}      # full weights, gate answers at chance
    wins = sum(board(ref, gated, s)["mmlu_general"] > board(ref, unlearned, s + 1000)["mmlu_general"]
               for s in SEEDS)                          # independent streams: two separate runs
    un47, bio_n = board(ref, unlearned, 47), int(re.search(
        r"Biosecurity: ([\d,]+)", parity.doc_text(PHASE, LESSON))[1].replace(",", ""))
    skill = (parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-wmdp-eval.md").read_text()
    rejects = re.findall(r"^- (Any .*)$", skill.split("Hard rejects:")[1].split("\n\n")[0], re.M)
    return {
        "gated47": board(ref, gated, 47), "un47": un47, "wins": wins / len(SEEDS),
        "bypass_bio": base["biosecurity"], "bio_n": bio_n,
        "bounds": (upper(un47["biosecurity"], 200), upper(un47["biosecurity"], bio_n)),
        "rejects": rejects,
        "studies": [p for p in ("acquisition trials", "novice-in-the-loop") if p in skill]
        + [p for p in ("Expert-absolute capability",) if p in parity.doc_text(PHASE, LESSON)],
        "cyber_post": round(un47["cybersecurity"] / 0.25, 2),
        "pre47": board(ref, base, 47, first=True),
    }


def verify(result):
    g, u, (b200, bfull) = result["gated47"], result["un47"], result["bounds"]
    hazard = {t: (g[t], u[t]) for t in TARGETS}
    return [
        practice.Check(
            "ANSWER: a WMDP pass cannot distinguish removed from gated capability",
            all([
                hazard == {"biosecurity": (0.255, 0.255), "chemistry": (0.245, 0.245)},
                (g["mmlu_general"], u["mmlu_general"]) == (0.785, 0.715),
                result["wins"] == 0.793,
                result["bypass_bio"] == 0.72,
                len(result["studies"]) == 3,
            ]),
            f"seed 47 (gated, unlearned): {hazard}, MMLU ({g['mmlu_general']}, "
            f"{u['mmlu_general']}); gated MMLU higher on {result['wins']:.1%} of seeds; "
            f"gate bypassed, bio = {result['bypass_bio']}; studies named {result['studies']}",
        ),
        practice.Check(
            "FINDING: 200 questions cannot certify 'at chance'",
            all([
                u["biosecurity"] == 0.255,
                result["bio_n"] == 1520,
                (b200, bfull) == (0.306, 0.273),
                (round(b200 / 0.25, 2), round(bfull / 0.25, 2)) == (1.22, 1.09),
            ]),
            f"bio 0.255: 95% upper bound {b200} at 200 questions, {bfull} at {result['bio_n']}",
        ),
        practice.Check(
            "FINDING: the toy's claim fails one of three hard rejects and says nothing about cyber",
            all([
                len(result["rejects"]) == 3,
                "novice-in-the-loop" in result["rejects"][2],
                result["cyber_post"] == 3.18,
                result["pre47"]["cybersecurity"] == u["cybersecurity"],
            ]),
            f"hard rejects {result['rejects']}; cyber after bio/chem unlearning "
            f"{result['cyber_post']}x chance",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
