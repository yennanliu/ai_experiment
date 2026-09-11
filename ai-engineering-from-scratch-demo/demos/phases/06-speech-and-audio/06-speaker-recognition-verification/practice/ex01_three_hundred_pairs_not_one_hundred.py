"""Exercise 1 — three hundred pairs, not one hundred.

    **Easy.** Run `code/main.py`. Builds synthetic "speakers" (different tone
    profiles), enrolls, computes EER on a 100-pair trial list.

Reading of the exercise: the sentence describes the code, so the check is whether
it describes it correctly. It does not. Five speakers with five utterances each
give `5 * C(5,2) = 50` same-speaker pairs and `C(5,2) * 25 = 250` different-speaker
pairs: **300 trials, not 100**. The EER is **0.00% at threshold 0.9888**.

At 50 and 250 the metric's grid is coarse. The false-reject rate moves in steps of
`1/50` = 2 pp and the false-accept rate in steps of `1/250` = 0.4 pp, so the
only reachable EERs are multiples of `1/500` = **0.20 pp**. Not one of the five
rows in the lesson's own leaderboard -- 0.39, 0.42, 0.65, 0.87, 3.10 -- lands on
that grid.

**The printed explanation is not the reason.** `main()` says "synthetic speakers
are near-orthogonal, so this toy hits 0% EER" four lines after printing a
different-speaker mean cosine of **0.558**, which is 56 degrees from orthogonal.
What produces 0% is the margin: the *lowest* same-speaker score is **0.9888** and
the *highest* different-speaker score is **0.7732**, a gap of 0.216. The five
utterances of a speaker are the same deterministic waveform plus independent
Gaussian noise -- same frequencies, same phases, same amplitude -- so there is no
session or channel variation, which is the entire difficulty of real verification.

**The means the demo prints do not predict the EER.** Raise the noise to 0.2 and
the two means converge to 0.999 and 0.989, a gap of 0.010 against 0.437 at the
shipped level -- and the EER is still **0.00%**. Separation lives in the tails.
At 0.5 the embeddings collapse together and EER jumps to **11.60%**.

Structure: `enroll` rebuilds `main()`'s corpus at a chosen noise level; `trials`
produces the two score lists; `drop` re-normalises an embedding with some
coefficients removed, for the c0 ablation.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "06-speaker-recognition-verification"
SR, SECONDS, PER_SPEAKER = 8000, 0.4, 5
SPEAKERS = {"alice": [200, 400, 600], "bob": [220, 330, 880], "carol": [300, 600, 1200],
            "dave": [180, 540, 1080], "eve": [260, 520, 780]}
NOISES = (0.04, 0.2, 0.5)
ENERGY = {0, 13}          # MFCC c0 in the mean half and in the std half
CLAIMED_PAIRS = 100
LEADERBOARD = (0.39, 0.42, 0.65, 0.87, 3.10)


def enroll(ref, noise, seed=123):
    random.seed(seed)
    return {name: [ref.embed_mfcc_stats(ref.tone_mix(freqs, SR, SECONDS, noise=noise), SR)
                   for _ in range(PER_SPEAKER)] for name, freqs in SPEAKERS.items()}


def trials(ref, enrolled):
    """(same-speaker scores, different-speaker scores) -- `main()`'s own trial list."""
    names, same, different = list(enrolled), [], []
    for name in names:
        vectors = enrolled[name]
        same += [ref.cosine(vectors[i], vectors[j])
                 for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            different += [ref.cosine(a, b) for a in enrolled[first] for b in enrolled[second]]
    return same, different


def drop(ref, vector, indices):
    """The embedding without some coefficients, re-normalised as `embed_mfcc_stats` does."""
    return ref.l2_normalize([x for i, x in enumerate(vector) if i not in indices])


def ablate(ref, enrolled, indices):
    return {name: [drop(ref, v, indices) for v in vectors] for name, vectors in enrolled.items()}


def summarise(ref, same, different):
    rate, threshold = ref.eer(same, different)
    return {"eer": rate, "threshold": threshold, "same": len(same), "diff": len(different),
            "same_mean": sum(same) / len(same), "diff_mean": sum(different) / len(different),
            "margin": min(same) - max(different), "floor": min(same), "ceiling": max(different)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = enroll(ref, NOISES[0])
    runs = {noise: summarise(ref, *trials(ref, enroll(ref, noise))) for noise in NOISES}
    base = runs[NOISES[0]]
    others = set(range(len(next(iter(shipped.values()))[0]))) - ENERGY
    return {
        "runs": runs, "base": base,
        "without_energy": summarise(ref, *trials(ref, ablate(ref, shipped, ENERGY))),
        "energy_only": summarise(ref, *trials(ref, ablate(ref, shipped, others))),
        "step": 1 / base["diff"] / 2,
        "off_grid": [e for e in LEADERBOARD if abs(e / (1 / base["diff"] / 2 * 100)
                                                   - round(e / (1 / base["diff"] / 2 * 100))) > 1e-9],
    }


def verify(result):
    base, runs = result["base"], result["runs"]
    quiet, loud = runs[NOISES[1]], runs[NOISES[2]]
    return [
        practice.Check(
            "ANSWER: EER 0.00%, on a 300-pair trial list the exercise calls 100",
            base["eer"] == 0.0 and base["same"] + base["diff"] == 300,
            f"5 speakers x {PER_SPEAKER} utterances give {base['same']} same-speaker and "
            f"{base['diff']} different-speaker pairs, {base['same'] + base['diff']} in all "
            f"against the {CLAIMED_PAIRS} the exercise names; EER is {base['eer'] * 100:.2f}% at "
            f"threshold {base['threshold']:.4f}",
        ),
        practice.Check(
            "FINDING: this trial list cannot express one row of the lesson's own leaderboard",
            len(result["off_grid"]) == len(LEADERBOARD),
            f"false rejects move in steps of 1/{base['same']} = 2.00 pp and false accepts in "
            f"1/{base['diff']} = 0.40 pp, so the reachable EERs are the multiples of "
            f"{result['step'] * 100:.2f} pp. All {len(result['off_grid'])} leaderboard rows "
            f"{list(LEADERBOARD)} fall between them",
        ),
        practice.Check(
            "FINDING: 'near-orthogonal' is not what the same file prints",
            base["diff_mean"] > 0.5 and base["margin"] > 0.2,
            f"different-speaker cosine averages {base['diff_mean']:.3f} -- 56 degrees from "
            f"orthogonal. What gives 0% is the margin: lowest same-speaker {base['floor']:.4f} "
            f"against highest different-speaker {base['ceiling']:.4f}, a gap of "
            f"{base['margin']:.4f}, because a speaker's five utterances are one deterministic "
            "waveform plus independent noise -- no session or channel variation at all",
        ),
        practice.Check(
            "MECHANISM: the printed means do not predict the EER",
            quiet["eer"] == 0.0 and quiet["diff_mean"] - base["diff_mean"] > 0.4,
            f"at noise {NOISES[1]} the two means converge to {quiet['same_mean']:.3f} and "
            f"{quiet['diff_mean']:.3f} -- a gap of "
            f"{quiet['same_mean'] - quiet['diff_mean']:.3f} against "
            f"{base['same_mean'] - base['diff_mean']:.3f} as shipped -- and the EER is still "
            f"{quiet['eer'] * 100:.2f}%. At {NOISES[2]} it is {loud['eer'] * 100:.2f}%",
        ),
        practice.Check(
            "CONTROL: the coefficient carrying most of the cosine carries none of the decision",
            result["without_energy"]["eer"] == 0.0 and result["energy_only"]["eer"] > 0.4,
            f"dropping MFCC c0 from both halves leaves EER at "
            f"{result['without_energy']['eer'] * 100:.2f}% and *lowers* the different-speaker "
            f"mean to {result['without_energy']['diff_mean']:.3f}; keeping only c0 gives "
            f"{result['energy_only']['eer'] * 100:.2f}%, which is chance",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
