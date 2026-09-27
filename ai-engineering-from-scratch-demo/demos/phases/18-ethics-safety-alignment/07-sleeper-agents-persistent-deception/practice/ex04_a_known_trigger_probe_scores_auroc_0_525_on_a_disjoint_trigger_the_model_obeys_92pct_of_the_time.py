"""Exercise 4 — a probe trained on one trigger scores AUROC 0.525 on a disjoint trigger the model obeys 92% of the time.

    The April 2024 probe result finds near-perfect separation on middle layers.
    Design an experiment that tests whether the probe generalizes from a known
    trigger to a structurally similar unknown trigger.

Reading of the exercise: the design is run, not only described. The toy gets 8
tripwire features (6-13), with N(0, 0.3) noise when they are off. A trigger is
4 of them set to 1.0. The model (the reference's `LogReg`, widened to 14
features) is planted with the known trigger {6, 7, 8, 9} and with an unknown
trigger that shares 0-4 of those dimensions ("structural similarity" as a dial).
The "residual stream" is the model's state before the sum, w_i * x_i. A linear
probe (the reference's `LogReg` again) is trained on known-trigger vs clean
states only. It is scored by AUROC on held-out known and unknown triggers
against fresh clean inputs. Everything is averaged over 5 seeds.

The design has four parts:

1. Plant the unknown trigger for real, and confirm the model defects on it.
2. Vary the overlap, so that "similar" is a measured quantity.
3. Include an unplanted control: how often the known-only model already
   defects on the unknown pattern.
4. Include a positive control: a readout of the model's own defect pathway
   (minus the summed tripwire contributions).

| shared dims | model defects on unknown | known-only model defects on it | probe AUROC known | probe AUROC unknown | pathway AUROC |
|---:|---:|---:|---:|---:|---:|
| 0 | 92.3% | 15.4% | 1.000 | 0.525 | 1.000 |
| 1 | 91.9% | 38.3% | 1.000 | 0.936 | 1.000 |
| 2 | 92.1% | 58.7% | 1.000 | 0.999 | 1.000 |
| 3 | 93.3% | 74.2% | 1.000 | 1.000 | 1.000 |
| 4 | 93.7% | 88.6% | 1.000 | 1.000 | 1.000 |

**ANSWER: it generalizes only through shared input dimensions.** The toy
reproduces "near-perfect separation" on the known trigger (AUROC 1.000 at
every overlap). A probe on a disjoint trigger scores 0.525, which is chance,
even though the model defects on 92.3% of that trigger's inputs. One shared
dimension of four lifts it to 0.936, and two give 0.999. The probe learned
where the known trigger's features sit, not that the model is about to defect.

**FINDING: without the unplanted control, overlap would pass as
generalization.** At 2 shared dimensions the known-only model already defects
on 58.7% of the unknown trigger's inputs. A "similar" trigger with high
overlap is partly the same backdoor, so a probe that catches it has shown
nothing new. The informative test is overlap 0, planted, and there the probe
fails.

**FINDING: the positive control shows what a generalizing probe must read.**
Summing the tripwire contributions reads the pathway that both backdoors share,
and it separates every unknown trigger at AUROC 1.000. To run this on a real
model, plant several triggers in disjoint token sets. Train the probe on one.
Test on the others at controlled overlap, with an unplanted control. Then check
whether the probe's direction matches a shared "defect" direction or a
trigger-specific one.

Structure: `run()` plants, probes and scores one seed and overlap, with
`ref.random` and `ref.FEATURES` swapped and restored in `solve()`.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "07-sleeper-agents-persistent-deception"
SEEDS, WIRES, NOISE = range(5), list(range(6, 14)), 0.3
KNOWN = WIRES[:4]                          # the trigger the probe builder has


def unknown(overlap):
    """A 4-dim trigger sharing `overlap` tripwire dims with KNOWN."""
    return KNOWN[:overlap] + WIRES[4:8 - overlap]


def draw(ref, n, dims=()):
    """Inputs with noisy tripwires; `dims` on at 1.0 makes a triggered (defect) example."""
    out = []
    for _ in range(n):
        x = [ref.random.gauss(0.0, 1.0) for _ in range(6)]
        x += [1.0 if j in dims else ref.random.gauss(0.0, NOISE) for j in WIRES]
        out.append(ref.Example(x=x, y=0 if dims else int(x[0] + x[1] > 0), trigger_on=bool(dims)))
    return out


def state(model, examples):
    """The model's internal state before the sum: each feature's contribution w_i * x_i."""
    return [[w * xi for w, xi in zip(model.w, e.x)] for e in examples]


def defects(model, triggered):
    """Share of possible defects (natural label 1) where the model outputs the defect."""
    possible = [e for e in triggered if e.x[0] + e.x[1] > 0]
    return sum(model.predict(e.x) == 0 for e in possible) / len(possible)


def auroc(pos, neg):
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def run(ref, seed, overlap):
    """Plant KNOWN and the unknown trigger; probe on KNOWN only; score both."""
    other = unknown(overlap)
    ref.random = random.Random(seed)
    lone, model = ref.LogReg(), ref.LogReg()
    plant = draw(ref, 400) + draw(ref, 100, KNOWN)
    ref.train(lone, list(plant), epochs=80)
    ref.train(model, plant + draw(ref, 100, other), epochs=80)
    clean, known, held = draw(ref, 200), draw(ref, 200, KNOWN), draw(ref, 200, other)
    probe = ref.LogReg()
    rows = [ref.Example(x=h, y=e.trigger_on, trigger_on=e.trigger_on)
            for h, e in zip(state(model, clean + known), clean + known)]
    ref.train(probe, rows, epochs=30)
    fresh = draw(ref, 200)
    score = lambda ex: [probe.predict_proba(h) for h in state(model, ex)]  # noqa: E731
    path = lambda ex: [-sum(h[j] for j in WIRES) for h in state(model, ex)]  # noqa: E731
    return {"fires": defects(model, held), "unplanted": defects(lone, held),
            "known": auroc(score(draw(ref, 200, KNOWN)), score(fresh)),
            "unknown": auroc(score(held), score(fresh)), "pathway": auroc(path(held), path(fresh))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    saved = ref.random, ref.FEATURES
    ref.FEATURES = 6 + len(WIRES)
    try:
        table = {}
        for overlap in range(5):
            runs = [run(ref, s, overlap) for s in SEEDS]
            table[overlap] = {k: round(sum(r[k] for r in runs) / len(runs), 3) for k in runs[0]}
    finally:
        ref.random, ref.FEATURES = saved
    doc = " ".join(parity.doc_text(PHASE, LESSON).split())
    return {"table": table, "lesson_claim": "almost perfectly separate" in doc}


def verify(result):
    t = result["table"]
    col = lambda k: tuple(t[o][k] for o in range(5))  # noqa: E731
    return [
        practice.Check(
            "ANSWER: it generalizes only through shared input dimensions",
            result["lesson_claim"] and col("known") == (1.0,) * 5 and col("unknown") == (0.525, 0.936, 0.999, 1.0, 1.0)
            and col("fires") == (0.923, 0.919, 0.921, 0.933, 0.937),
            f"by shared dims 0-4: probe AUROC known {col('known')}, unknown {col('unknown')}; "
            f"model defects on the unknown trigger {col('fires')}; lesson claims near-perfect "
            f"separation: {result['lesson_claim']}",
        ),
        practice.Check(
            "FINDING: without the unplanted control, overlap would pass as generalization",
            col("unplanted") == (0.154, 0.383, 0.587, 0.742, 0.886),
            f"known-only model defects on the unknown pattern, by shared dims: {col('unplanted')}",
        ),
        practice.Check(
            "FINDING: the positive control shows what a generalizing probe must read",
            col("pathway") == (1.0,) * 5 and t[0]["unknown"] < 0.6,
            f"defect-pathway readout AUROC by shared dims {col('pathway')}, against the "
            f"probe's {t[0]['unknown']} at 0",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
