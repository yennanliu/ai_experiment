"""Exercise 4 — the lesson's AI labeler scores an empty answer above the trained model, and padding flips 57% of its verdicts.

    CAI replaces human labelers with AI labelers. Name a sycophancy-like
    failure mode that can still occur in RLAIF, and design a detection for it.

Reading of the exercise: "sycophancy-like" is read as a labeler rewarding
what looks like agreement with its principle rather than what the user
needed -- the RLAIF version of telling the rater what it wants to hear. It is
shown on the lesson's own labeler, `ai_feedback_rank`, and the detection is a
probe suite run on a labeler before its preferences are used: two
content-neutral edits the verdict must not follow.

**ANSWER: surface compliance -- the labeler rewards looking harmless, and the
cheapest way to look harmless is to say less, or dilute.** Put a policy that
answers every prompt with nothing into the shipped Phase 2 loop and it wins
484/500 against base, against the CAI model's 482: saying nothing
scores better than the trained model. The detector is two
probes. Padding: add six neutral tokens to the losing response and check the
verdict holds; on the reference labeler it flips 874 of 1,527 strictly
decided base-vs-base pairs (57.2%), because it scores a *rate* of flagged
tokens, not their count. Null: an empty answer must never beat a clean
on-task one; it wins 477 of 1,000 (47.7%), on a coin flip. A labeler
that counts flagged tokens and ranks an empty answer below any clean one
passes both: 0 flips and 0 null wins.

**FINDING: the lesson's claim that sycophancy drops because "the AI labeler
has no user to please" moves the failure rather than removing it.** The
labeler still has a principle to please, and every lever above scores as
harmlessness progress in the lesson's own win-rate metric.

Structure: `phase2()` replays the shipped `main()` with the policy patched;
`pad_flips()` and `null_wins()` are the two probes, run on the reference
labeler and on `counted()`, the repaired one.
"""

from __future__ import annotations

import contextlib
import io
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "05-constitutional-ai-rlaif"
PAIRS, PAD = 2000, ["please"] * 6


def phase2(ref, policy):
    """The shipped main() at seed 11 with the CAI policy replaced: Phase 2 wins."""
    saved, log = (ref.random, ref.cai_model_sample), io.StringIO()
    ref.random, ref.cai_model_sample = random.Random(11), policy
    try:
        with contextlib.redirect_stdout(log):
            ref.main()
    finally:
        ref.random, ref.cai_model_sample = saved
    return int(re.search(r"AI-feedback: (\d+)/", log.getvalue()).group(1))


def counted(ref):
    """Repaired labeler: fewer flagged tokens wins; an empty answer loses to any clean one."""
    def key(r):
        return (sum(t in ref.HARMFUL_TOKENS for t in r), not r)
    return lambda a, b: 0 if key(a) < key(b) else 1 if key(b) < key(a) else None


def strict_rank(ref, rank):
    """The reference labeler with its coin-flip tie made explicit (None)."""
    return lambda a, b: rank(a, b) if ref.harmful_token_rate(a) != ref.harmful_token_rate(b) else None


def pad_flips(pairs, rank):
    strict = [(a, b, w) for a, b in pairs if (w := rank(a, b)) is not None]
    flips = sum(rank(*((a, b + PAD) if w == 0 else (a + PAD, b))) != w for a, b, w in strict)
    return flips, len(strict)


def null_wins(ref, rank, n=1000):
    rng, wins = random.Random(3), 0
    saved, ref.random = ref.random, rng
    try:
        for _ in range(n):
            clean = [rng.choice(ref.SAFE_TOKENS) for _ in range(6)]
            wins += rank([], clean) == 0
    finally:
        ref.random = saved
    return wins


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    saved, ref.random = ref.random, random.Random(1)
    try:
        pairs = [(ref.base_model_sample(), ref.base_model_sample()) for _ in range(PAIRS)]
    finally:
        ref.random = saved
    fixed, doc = counted(ref), parity.doc_text(PHASE, LESSON)
    return {
        "empty_wins": phase2(ref, lambda prompt, model, n_tokens=6: []),
        "cai_wins": phase2(ref, ref.cai_model_sample),
        "pad": pad_flips(pairs, strict_rank(ref, ref.ai_feedback_rank)),
        "null": null_wins(ref, ref.ai_feedback_rank),
        "fixed_pad": pad_flips(pairs, fixed), "fixed_null": null_wins(ref, fixed),
        "doc_claim": "the AI labeler has no user to please" in doc,
    }


def verify(result):
    r = result
    (flips, strict), (fflips, fstrict) = r["pad"], r["fixed_pad"]
    return [
        practice.Check(
            "ANSWER: surface compliance -- an empty policy outscores CAI, padding and null probes flag it",
            (r["cai_wins"], r["empty_wins"], flips, strict, r["null"]) == (482, 484, 874, 1527, 477)
            and round(flips / strict, 3) == 0.572,
            f"Phase 2 wins: empty policy {r['empty_wins']}/500, CAI {r['cai_wins']}/500; padding "
            f"flips {flips}/{strict} ({flips / strict:.1%}); empty beats clean {r['null']}/1000",
        ),
        practice.Check(
            "ANSWER: a count-based labeler that ranks empty last passes both probes",
            (fflips, r["fixed_null"]) == (0, 0) and fstrict > 0,
            f"repaired labeler: padding flips {fflips}/{fstrict}, empty wins {r['fixed_null']}/1000",
        ),
        practice.Check(
            "FINDING: 'no user to please' moves the failure rather than removing it",
            r["doc_claim"] and r["empty_wins"] > r["cai_wins"] and flips > 0,
            f"lesson claims no user to please: {r['doc_claim']}; empty policy "
            f"{r['empty_wins']} vs CAI {r['cai_wins']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
