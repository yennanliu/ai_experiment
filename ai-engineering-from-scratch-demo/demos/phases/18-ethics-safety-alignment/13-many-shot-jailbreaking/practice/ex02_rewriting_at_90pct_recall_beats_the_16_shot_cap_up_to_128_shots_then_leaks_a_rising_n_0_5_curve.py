"""Exercise 2 — rewriting at 90% recall beats the 16-shot cap up to 128 shots, then leaks a rising n^0.5 curve.

    Implement a simple MSJ defense: run a classifier over the full context;
    if N pattern-match examples of harmful-compliance pairs are detected,
    truncate or rewrite. Measure the new shot-vs-ASR curve.

Reading of the exercise: contexts are lists of faux user/assistant pairs with
abstract placeholders ("<harmful request #k>" / "<compliant answer #k>", and
"<benign task #k>" for benign turns). A pair classifier flags each pair with a
stated recall on harmful pairs and a 1% false-positive rate on benign ones.
Once N = 4 pairs are flagged, the defense either *rewrites* every flagged
answer to a refusal or *truncates* every turn before the target query. The
target is the lesson's own `target_asr`, applied to the number of harmful
pairs whose answer still complies. Expected ASR is averaged over 400 seeded
contexts per shot count, and it is compared with the reference's
`defense_adjusted`, which caps effective shots at 16.

**ANSWER: at 90% recall the rewrite curve is 0.050 / 0.062 / 0.027 / 0.032
/ 0.050 / 0.070 / 0.094 / 0.128 / 0.171 / 0.234 at 1-512 shots.**
Undefended is 0.699 at 512. The reference's cap is flat at 0.140 from 16
shots on, so the rewrite beats it through 128 shots and loses from 256. The
10% of pairs it misses still follow the lesson's power law: fitted on 32-512
shots with the 0.02 offset removed, the leak climbs as n^0.52. At perfect
recall the rewrite is 0.000 from 4 shots on. Below the trigger (1 and 2
shots) the context passes untouched.

**FINDING: truncation stops the attack by destroying benign ICL.** At 90%
recall, truncation is 0.000 from 8 shots on. But on a pure-benign 512-pair
context at a 1% false-positive rate it keeps only 23.5% of benign pairs,
because four false flags wipe the whole context. Rewriting keeps 99.1%.

**FINDING: the lesson's "61% -> 2%" is out of the toy's reach unless no
compliant shot survives.** One leaked compliant pair already gives
target_asr(1) = 0.050, and the reference's 16-shot cap gives 0.140. Only a
policy that leaves zero compliant pairs gets under 2%, and target_asr(0) is
exactly 0.

Structure: `context()` builds a placeholder context; `classify()` is the
pair classifier; `defend()` applies rewrite or truncate once N pairs are
flagged; `curve()` averages `target_asr` over seeded contexts.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "13-many-shot-jailbreaking"
SHOTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
N_TRIGGER, CONTEXTS, BENIGN_SHOTS, FPR = 4, 400, 512, 0.01
REFUSAL = "<refusal>"


def context(n_harm, n_benign, rng):
    """Faux turns with abstract placeholders; the target query follows the last pair."""
    pairs = [{"user": f"<harmful request #{k}>", "assistant": f"<compliant answer #{k}>", "harmful": True}
             for k in range(n_harm)]
    pairs += [{"user": f"<benign task #{k}>", "assistant": f"<benign answer #{k}>", "harmful": False}
              for k in range(n_benign)]
    rng.shuffle(pairs)
    return pairs


def classify(pairs, rng, recall, fpr=FPR):
    """Indices the pair classifier flags as harmful-compliance, with the given error rates."""
    return [i for i, p in enumerate(pairs) if rng.random() < (recall if p["harmful"] else fpr)]


def defend(pairs, flagged, policy):
    """Apply the policy once N pairs are flagged: rewrite flagged answers, or truncate all turns."""
    if len(flagged) < N_TRIGGER:
        return pairs
    if policy == "truncate":
        return []
    hit = set(flagged)
    return [dict(p, assistant=REFUSAL) if i in hit else p for i, p in enumerate(pairs)]


def compliant_shots(pairs):
    """What the target's power law counts: harmful pairs whose answer still complies."""
    return sum(p["harmful"] and p["assistant"] != REFUSAL for p in pairs)


def curve(ref, policy, recall, seed=0):
    rng, out = random.Random(seed), {}
    for n in SHOTS:
        asrs = []
        for _ in range(CONTEXTS):
            pairs = context(n, 0, rng)
            asrs.append(ref.target_asr(compliant_shots(defend(pairs, classify(pairs, rng, recall), policy))))
        out[n] = round(sum(asrs) / CONTEXTS, 3)
    return out


def benign_kept(policy, seed=0):
    """Share of benign pairs in a pure benign context that survive the defense unchanged."""
    rng, kept = random.Random(seed), 0
    for _ in range(CONTEXTS):
        pairs = context(0, BENIGN_SHOTS, rng)
        after = defend(pairs, classify(pairs, rng, 1.0), policy)
        kept += sum(p["assistant"] != REFUSAL for p in after)
    return round(kept / (CONTEXTS * BENIGN_SHOTS), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    r = {
        "none": {n: round(ref.target_asr(n), 3) for n in SHOTS},
        "cap16": {n: round(ref.defense_adjusted(n), 3) for n in SHOTS},
        "rewrite_1.0": curve(ref, "rewrite", 1.0),
        "rewrite_0.9": curve(ref, "rewrite", 0.9),
        "truncate_0.9": curve(ref, "truncate", 0.9),
        "kept": {p: benign_kept(p) for p in ("rewrite", "truncate")},
    }
    lk = r["rewrite_0.9"]
    r["floor"] = (ref.target_asr(0), round(ref.target_asr(1), 3))
    r["claim"] = "from 61% to 2%" in parity.doc_text(PHASE, LESSON)
    r["leak_fit"] = ref.fit_power_law(SHOTS[5:], [lk[n] - 0.02 for n in SHOTS[5:]])
    return r


def verify(result):
    rw, cap, none, kept = (result[k] for k in ("rewrite_0.9", "cap16", "none", "kept"))
    beats = [n for n in SHOTS[4:] if rw[n] < cap[n]]
    return [
        practice.Check(
            "ANSWER: rewrite at 90% recall beats the 16-shot cap up to 128 shots, then leaks n^0.52",
            (list(rw.values()), beats, round(result["leak_fit"][0], 2), none[512],
             set(list(cap.values())[4:]), list(result["rewrite_1.0"].values())[2:])
            == ([0.05, 0.062, 0.027, 0.032, 0.05, 0.07, 0.094, 0.128, 0.171, 0.234],
                [16, 32, 64, 128], 0.52, 0.699, {0.14}, [0.0] * 8),
            f"rewrite@0.9 {rw}; cap-16 {cap}; leak exponent {result['leak_fit'][0]:.3f}; "
            f"rewrite@1.0 {result['rewrite_1.0']}",
        ),
        practice.Check(
            "FINDING: truncation stops the attack by destroying benign ICL",
            (list(result["truncate_0.9"].values())[3:], kept)
            == ([0.0] * 7, {"rewrite": 0.991, "truncate": 0.235}),
            f"truncate@0.9 {result['truncate_0.9']}; benign pairs kept at 512 shots, 1% FPR: {kept}",
        ),
        practice.Check(
            "FINDING: '61% -> 2%' is out of reach unless no compliant shot survives",
            (result["claim"], result["floor"], cap[512]) == (True, (0.0, 0.05), 0.14),
            f"target_asr(0), target_asr(1) = {result['floor']}; cap-16 at 512 shots {cap[512]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
