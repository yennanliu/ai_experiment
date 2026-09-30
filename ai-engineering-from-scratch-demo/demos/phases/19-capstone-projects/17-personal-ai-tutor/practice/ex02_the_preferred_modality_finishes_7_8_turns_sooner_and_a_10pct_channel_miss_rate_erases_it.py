"""Exercise 2 — the preferred modality finishes 7.8 turns sooner, and a 10% channel miss rate erases it.

    Add a multimodal probe: the same concept question delivered as text, voice, and photo. Measure whether learners converge faster with the modality they prefer.

Reading of the exercise: the lesson ships no voice or photo path and no
learners, so the probe runs inside its simulator. Each of 10 learners (the
study's size; ability from `random.Random(seed)` as in `main()`) takes the
same curriculum three times with the reference `run_adaptive`, once per
modality, all three on the same seeded stream. "Converge" is the reference's
own stopping rule: every concept at BKT mastery 0.85, counted in turns.
The modality enters in two ways that are assumptions, not measurements. A
preferred modality adds 0.3 logits to the learner's knowledge, about +7
points of P(correct) at the simulator's starting 54%. A channel can lose a
right answer -- an ASR or OCR misread -- with a per-answer miss rate; the
main run uses 10% for voice and 5% for photo, and text is clean.
`simulate_answer` is wrapped so the reference loop itself sees both.

**ANSWER: yes on clean channels, and the size is decided by the channel,
not the preference.** With no misreads, the preferred modality converges
7.8 turns sooner (of about 48; sd 2.4 across 100 cohorts of 10, faster in
all 100). With the assumed misreads, text-preferrers gain 15.7 turns,
photo-preferrers 9.2 and voice-preferrers only 2.0. Over 300 voice-preferring
learners, the 0.3-logit preference is worth -7.9 turns on a clean channel and
is gone at a 10% miss rate: that is where (1 - miss) x P(correct | preferred)
falls back to the unpreferred P(correct).

**FINDING: without any preference, channel error alone reads as one.** With
the preference effect set to 0 and the same misread rates, voice-preferring
learners take 7.1 more turns in "their" modality and text-preferrers 7.2
fewer -- a probe that does not separate the channel from the learner would
report a preference effect in both directions that does not exist.

**FINDING: the learner model cannot see the modality.** `bkt_update(mastery,
correct, p)` takes a bare boolean, so a misread is charged to the learner as
a slip: one right answer lost to the channel at mastery 0.648 leaves 0.277
instead of 0.927. The 10-learner study also splits 4/3/3 across preferences,
so each per-modality estimate rests on 3-4 learners.

Structure: `turns()` is one reference run under one modality;
`preference_gap()` is preferred minus the mean of the other two;
`break_even()` sweeps the voice miss rate.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "17-personal-ai-tutor"
MODALITIES = ("text", "voice", "photo")
BONUS = 0.3  # assumed: logits a learner gains in the modality they prefer
MISREAD = {"text": 0.0, "voice": 0.10, "photo": 0.05}  # assumed ASR / OCR miss rate per right answer
CLEAN = dict.fromkeys(MODALITIES, 0.0)


def turns(ref, ability, modality, preferred, bonus, miss, seed):
    """Turns until the reference tutor stops (all concepts at 0.85), in one modality."""
    real = ref.simulate_answer

    def through_channel(ek, d, rng):
        right = real(ek + (bonus if modality == preferred else 0.0), d, rng)
        return (rng.random() >= miss[modality]) and right  # the draw is always taken: same stream everywhere

    ref.simulate_answer = through_channel
    try:
        return len(ref.run_adaptive("probe", ability, ref.curriculum_map(ref.ALGEBRA), 400, random.Random(seed)).history)
    finally:
        ref.simulate_answer = real


def cohort(ref, seed, bonus, miss):
    rng, rows = random.Random(seed), []
    for i in range(10):
        ability, pref = rng.gauss(0.3, 0.4), MODALITIES[i % 3]
        t = {m: turns(ref, ability, m, pref, bonus, miss, seed * 1000 + i) for m in MODALITIES}
        rows.append((pref, t[pref] - statistics.mean(t[m] for m in MODALITIES if m != pref)))
    return rows


def preference_gap(ref, bonus, miss, n=100):
    cohorts = [cohort(ref, s, bonus, miss) for s in range(n)]
    means = [statistics.mean(g for _, g in c) for c in cohorts]
    by_pref = {m: round(statistics.mean(g for c in cohorts for p, g in c if p == m), 1) for m in MODALITIES}
    return {"mean": round(statistics.mean(means), 1), "sd": round(statistics.stdev(means), 1),
            "faster": sum(x < 0 for x in means), "by_pref": by_pref}


def break_even(ref, n=300):
    """Voice-preferring learners: voice minus text turns, as the voice miss rate rises."""
    rng = random.Random(5)
    abilities = [rng.gauss(0.3, 0.4) for _ in range(n)]
    gap = lambda e: statistics.mean(
        turns(ref, a, "voice", "voice", BONUS, {**CLEAN, "voice": e}, 7000 + i)
        - turns(ref, a, "text", "voice", BONUS, {**CLEAN, "voice": e}, 7000 + i) for i, a in enumerate(abilities))
    sweep = {round(e / 100, 2): round(gap(e / 100), 1) for e in range(0, 21, 2)}
    return sweep, next(e for e, g in sweep.items() if g >= 0)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    p = ref.BKTParams()
    sweep, even = break_even(ref)
    m1 = ref.bkt_update(p.p_init, True, p)
    return {
        "clean": preference_gap(ref, BONUS, CLEAN), "misread": preference_gap(ref, BONUS, MISREAD),
        "no_preference": preference_gap(ref, 0.0, MISREAD), "sweep": sweep, "break_even": even,
        "lost_answer": [round(ref.bkt_update(m1, True, p), 3), round(ref.bkt_update(m1, False, p), 3)],
        "split": [sum(MODALITIES[i % 3] == m for i in range(10)) for m in MODALITIES],
    }


def verify(result):
    r = result
    c, m, z = r["clean"], r["misread"], r["no_preference"]
    return [
        practice.Check(
            "ANSWER: preferred converges 7.8 turns sooner on clean channels; a 10% misread rate erases it",
            ((c["mean"], c["sd"], c["faster"]), m["by_pref"], r["sweep"][0.0], r["break_even"])
            == ((-7.8, 2.4, 100), {"text": -15.7, "voice": -2.0, "photo": -9.2}, -7.9, 0.1),
            f"clean channels: gap {c['mean']} turns (sd {c['sd']}, faster in {c['faster']}/100 cohorts); "
            f"with misreads by preference {m['by_pref']}; voice sweep {r['sweep']}; break-even miss rate {r['break_even']}",
        ),
        practice.Check(
            "FINDING: with no preference at all, channel error alone reads as a preference effect",
            (z["mean"], z["by_pref"]) == (-1.0, {"text": -7.2, "voice": 7.1, "photo": -0.9}),
            f"preference effect 0, misreads on: gap by preference {z['by_pref']}, overall {z['mean']}",
        ),
        practice.Check(
            "FINDING: bkt_update cannot see the modality, so a misread is charged to the learner",
            (r["lost_answer"], r["split"]) == ([0.927, 0.277], [4, 3, 3]),
            f"at 0.648 a right answer gives {r['lost_answer'][0]}, the same answer lost to the channel "
            f"{r['lost_answer'][1]}; 10 learners split {r['split']} across text/voice/photo",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
