"""Exercise 1 — one revision pass takes the rate from 0.346 to 0, and an untrained model scores 0 too.

    Run `code/main.py`. Compare the base model's harmful-token rate to the
    CAI-trained version. How many revision steps are needed to approach zero?

Reading of the exercise: the comparison is read off the shipped run (seed 11).
"Revision steps" has no knob in the reference -- `revise` rewrites every
flagged token at once -- so the question is answered two ways: how many passes
the shipped `revise` needs, and how many a weaker critique needs if each
critique names only one flagged token (the reference `critique` truncated to
its first item), measured over 20,000 seeded base responses.

**ANSWER: 0.346 -> 0.000, and one pass is enough.** The base model's
harmful-token rate on 200 prompts is 0.346; after critique-and-revise SFT it
is 0.000 (a 100.0% reduction), and CAI beats base in 482/500 AI-feedback
pairs (96.4%). The shipped `revise` maps every flagged token through
`REPLACEMENT` in one call, so one pass leaves no harmful token in any SFT
target. If each critique names one token instead, the rate over 20,000
responses falls 0.3503 -> 0.1716 -> 0.0553 -> 0.0097 -> 0.0007 -> 0 at step 5:
three steps to under 1%, five to exactly zero.

**FINDING: the 0.000 is not evidence of training.** An empty model -- no
corpus at all -- also scores 0.000, because `cai_model_sample` falls back to
drawing only SAFE_TOKENS for any prompt it has not memorised. Train on
unrevised targets instead and the "CAI" rate is 0.312, 90% of base: revision,
not SFT, is what zeroes the lookups, and the fallback zeroes everything else.
The "held-out" set is not held out either: 87.0% of the 200 evaluation
prompts end in a two-token key the lookup memorised.

**FINDING: the RLAIF win rate is set by arithmetic, not by preference
learning.** A zero-harm policy against a 6-token base response at
p_harmful = 0.35 wins unless the base is clean, and half those ties:
1 - 0.65^6 / 2 = 96.2%, against the 96.4% printed. Phase 2 trains nothing.

**FINDING: the lesson's "Use It" describes code that is not there.** It says
"After 200 iterations" the model internalises the rule and to compare an
"RLHF-shaped toy"; 200 is `evaluate`'s prompt count, there is no training
loop, and `main.py` contains no RLHF model.

Structure: `run()` replays the shipped `main()` with a seeded `random.Random`
swapped in and, optionally, `revise` or the trained lookup patched (all
restored); `steps()` applies the one-token critique repeatedly.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "05-constitutional-ai-rlaif"
NUM = r"harmful-token rate (?:on 200 prompts|after CAI-SFT ) *: ([\d.]+)"


def run(ref, revise=None, empty=False):
    """Shipped main() at seed 11: (base rate, cai rate, wins, eval lookup hits)."""
    saved = (ref.random, ref.revise, ref.cai_model_sample)
    hits, log = [], io.StringIO()

    def sample(prompt, model, n_tokens=6):
        model = {} if empty else model
        key = tuple(prompt[-2:])
        hits.append(key in model)
        return saved[2](prompt, model, n_tokens)

    ref.random, ref.revise, ref.cai_model_sample = random.Random(11), revise or saved[1], sample
    try:
        with contextlib.redirect_stdout(log):
            ref.main()
    finally:
        ref.random, ref.revise, ref.cai_model_sample = saved
    out = log.getvalue()
    base, cai = map(float, re.findall(NUM, out))
    return base, cai, int(re.search(r"AI-feedback: (\d+)/", out).group(1)), sum(hits[:200]), out


def steps(ref, n=20000, k_max=6):
    """Mean harmful-token rate after k one-token critique-and-revise steps."""
    ref_rng, ref.random = ref.random, random.Random(0)
    try:
        cur = [ref.base_model_sample() for _ in range(n)]
    finally:
        ref.random = ref_rng
    rates = []
    for _ in range(k_max + 1):
        rates.append(round(sum(map(ref.harmful_token_rate, cur)) / n, 4))
        cur = [ref.revise(r, ref.critique(r, "")[:1]) for r in cur]
    return rates


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base, cai, wins, hits, out = run(ref)
    source, doc = inspect.getsource(ref), parity.doc_text(PHASE, LESSON)
    return {
        "base": base, "cai": cai, "wins": wins, "hits": hits / 200,
        "reduction": re.search(r"reduction *: ([\d.]+)%", out).group(1),
        "unrevised": run(ref, revise=lambda r, bad: r)[1],
        "empty": run(ref, empty=True)[1],
        "steps": steps(ref),
        "one_pass": max(ref.harmful_token_rate(ref.revise(t, ref.critique(t, "")))
                        for t in [[h] * 6 for h in ref.HARMFUL_TOKENS]),
        "win_theory": round(1 - (1 - 0.35) ** 6 / 2, 3),
        "doc_claims": ["After 200 iterations" in doc, "RLHF-shaped toy" in doc],
        "code_has": ["rlhf" in source.lower(), inspect.signature(ref.evaluate).parameters["n"].default],
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: 0.346 -> 0.000, and one pass is enough",
            (r["base"], r["cai"], r["reduction"], r["wins"]) == (0.346, 0.0, "100.0", 482)
            and r["one_pass"] == 0.0
            and r["steps"] == [0.3503, 0.1716, 0.0553, 0.0097, 0.0007, 0.0, 0.0],
            f"base {r['base']}, CAI {r['cai']} ({r['reduction']}% less), {r['wins']}/500 wins; "
            f"one full revise leaves {r['one_pass']}; one-token steps {r['steps']}",
        ),
        practice.Check(
            "FINDING: the 0.000 is not evidence of training",
            r["empty"] == 0.0 and r["unrevised"] == 0.312 and r["hits"] == 0.87
            and round(r["unrevised"] / r["base"], 1) == 0.9,
            f"empty model {r['empty']}, trained on unrevised targets {r['unrevised']}, "
            f"eval prompts hitting the lookup {r['hits']:.1%}",
        ),
        practice.Check(
            "FINDING: the RLAIF win rate is set by arithmetic, not by preference learning",
            r["win_theory"] == 0.962 and abs(r["wins"] / 500 - r["win_theory"]) < 0.01,
            f"1 - 0.65^6/2 = {r['win_theory']:.1%} vs measured {r['wins'] / 500:.1%}",
        ),
        practice.Check(
            "FINDING: the lesson's 'Use It' describes code that is not there",
            r["doc_claims"] == [True, True] and r["code_has"] == [False, 200],
            f"doc says '200 iterations' and 'RLHF-shaped toy': {r['doc_claims']}; code mentions "
            f"RLHF: {r['code_has'][0]}; evaluate's default n = {r['code_has'][1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
