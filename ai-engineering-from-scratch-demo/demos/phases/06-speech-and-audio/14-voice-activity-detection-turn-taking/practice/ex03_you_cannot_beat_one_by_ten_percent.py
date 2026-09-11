"""Exercise 3 — you cannot beat 1.000 by ten percent.

    **Hard.** Build a mini turn-detector: Silero VAD + a 3-layer MLP on the last
    10 words' embeddings (use sentence-transformers). Train on a hand-labeled
    turn-end dataset. Beat Silero-only by 10% F1.

Reading of the exercise: the target is stated relative to a baseline, so the
baseline has to be measured before anything is built. On the stream `code/main.py`
ships, Silero-only scores **F1 1.000** -- two gold turn ends, two predicted, no
false positives -- so **"beat it by 10%" has no solution**. The exercise's own
fixture forecloses its own target.

A fixture where the baseline is beatable has to contain the failures the doc
names, so this one does: a clatter of **15** isolated 20 ms transients between
turns ("coughs or chair noise"), and a turn split by a **640 ms** pause ("Hmm,
let me think..."). Over 10880 ms with three hand-labelled turn ends:

| arm | F1 | true pos | false pos |
|---|---:|---:|---:|
| Silero-only, as shipped | **0.750** | 3 | 2 |
| + count *consecutive* speech | **0.857** | 3 | 1 |

That is **+10.7 points**, the gain the exercise asks for, and it comes from one
line rather than from a language model.

**The mechanism is that `min_speech_ms` is cumulative, not consecutive.** In the
idle state `update` never clears `speech_ms`, so 20 ms transients accumulate
across arbitrary silence: the **13th** one fires a START whatever the spacing --
at 3600 ms with 280 ms gaps, at 9840 ms with 800 ms gaps. The guard the doc
describes as rejecting "speech shorter than 250 ms" rejects the first twelve
coughs and accepts the thirteenth. Clearing `speech_ms` on a silent chunk while
idle removes that failure entirely: 40 isolated transients then fire nothing.

**The remaining error is the one an MLP is actually for.** The 640 ms mid-turn
pause exceeds the 500 ms hangover, so one turn is reported as two, and no counter
fixes it. Raising the hangover to 700 ms reaches F1 **1.000** here, but the
window is narrow -- at 900 ms every predicted end falls outside the +/-700 ms
scoring collar and F1 goes to **0.000**. Choosing by silence alone is the problem
semantic endpointing exists to solve.

And it cannot be solved here: `sentence_transformers`, `torch` and `silero_vad`
are absent, and there are no words to embed -- `synth_chunk` returns Gaussian
noise and the module holds no text at all, so "the last 10 words' embeddings" has
no input.

Structure: `fixture` builds the labelled stream; `run` drives one detector;
`consecutive` is the one-line fix, applied as a wrapper; `score` is F1 against
the authored turn ends; `clatter` counts transients to the first spurious START.
"""

from __future__ import annotations

import importlib.util
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "14-voice-activity-detection-turn-taking"
CHUNK_MS, COLLAR, GATE, SEED = 20, 700, 0.5, 11
SHIPPED = (("silence", 10, 0), ("speech", 40, 1), ("silence", 30, 0), ("cough", 1, 0),
           ("silence", 10, 0), ("speech", 25, 2), ("silence", 35, 0))
HARD = (("silence", 15, 0), ("speech", 30, 1), ("silence", 45, 0))
HARD += (("cough", 1, 0), ("silence", 14, 0)) * 15
HARD += (("silence", 20, 0), ("speech", 25, 2), ("silence", 32, 2), ("speech", 22, 2),
         ("silence", 50, 0), ("speech", 35, 3), ("silence", 45, 0))
ABSENT = ("sentence_transformers", "torch", "silero_vad", "transformers")


def fixture(ref, plan, seed=SEED):
    """(chunks, gold turn-end times) -- the turn labels are authored, not inferred."""
    rng = random.Random(seed)
    random.seed(seed)
    chunks = [(kind, turn, ref.synth_chunk(kind, rng))
              for kind, count, turn in plan for _ in range(count)]
    turns = sorted({turn for _, turn, _ in chunks} - {0})
    ends = [(max(i for i, (k, t, _) in enumerate(chunks) if t == turn and k == "speech") + 1)
            * CHUNK_MS for turn in turns]
    return chunks, ends


def consecutive(detector, is_speech):
    """The one-line fix: a silent chunk while idle clears the speech counter."""
    if not is_speech and detector.state == "idle":
        detector.speech_ms = 0
    return detector.update(is_speech)


def run(ref, probs, fixed=False, **kwargs):
    """Turn-end times from one arm of the comparison."""
    detector = ref.TurnDetector(**kwargs)
    step = consecutive if fixed else (lambda d, s: d.update(s))
    fired = [(i * CHUNK_MS, step(detector, p >= GATE)) for i, p in enumerate(probs)]
    return [when for when, event in fired if event == "END"]


def score(ends, truth, collar=COLLAR):
    """F1 of predicted turn ends against gold, matched within a collar."""
    used, hits = set(), 0
    for when in ends:
        near = [g for g in truth if abs(when - g) <= collar and g not in used]
        if near:
            used.add(near[0])
            hits += 1
    precision, recall = hits / max(1, len(ends)), hits / max(1, len(truth))
    return {"f1": 2 * precision * recall / max(1e-9, precision + recall),
            "tp": hits, "fp": len(ends) - hits, "fn": len(truth) - hits}


def clatter(ref, gap, fixed=False, limit=40):
    """Isolated 20 ms transients until a spurious START, or None within `limit`."""
    detector = ref.TurnDetector()
    step = consecutive if fixed else (lambda d, s: d.update(s))
    for count in range(1, limit + 1):
        if step(detector, True) == "START":
            return count
        for _ in range(gap):
            step(detector, False)
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped, shipped_gold = fixture(ref, SHIPPED, seed=42)
    hard, hard_gold = fixture(ref, HARD)
    probs = [ref.fake_silero_vad(chunk, None) for _, _, chunk in hard]
    ship_probs = [ref.fake_silero_vad(chunk, None) for _, _, chunk in shipped]
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "shipped": score(run(ref, ship_probs), shipped_gold),
        "shipped_gold": len(shipped_gold), "hard_ms": len(hard) * CHUNK_MS,
        "gold": hard_gold,
        "base": score(run(ref, probs), hard_gold),
        "fixed": score(run(ref, probs, fixed=True), hard_gold),
        "hangover": {h: score(run(ref, probs, fixed=True, silence_hangover_ms=h), hard_gold)
                     for h in (700, 900)},
        "clatter": {gap: clatter(ref, gap) for gap in (14, 40)},
        "clatter_fixed": clatter(ref, 14, fixed=True),
    }


def verify(result):
    base, fixed, shipped = result["base"], result["fixed"], result["shipped"]
    gain = fixed["f1"] - base["f1"]
    return [
        practice.Check(
            "ANSWER: on the shipped stream the baseline is already 1.000",
            shipped["f1"] == 1.0,
            f"Silero-only finds {shipped['tp']} of {result['shipped_gold']} gold turn ends with "
            f"{shipped['fp']} false positives -- F1 {shipped['f1']:.3f}. 'Beat Silero-only by "
            "10% F1' has no solution on the fixture the lesson ships with the exercise",
        ),
        practice.Check(
            "FINDING: on a fixture carrying the doc's own failure modes it scores 0.750",
            base["f1"] < 0.8 and base["fp"] == 2,
            f"{result['hard_ms']} ms with three hand-labelled turn ends at {result['gold']} ms, "
            f"a clatter of 15 isolated 20 ms transients, and a turn split by a 640 ms pause: "
            f"F1 {base['f1']:.3f}, {base['tp']} true and {base['fp']} false positives. Both "
            "failures are ones the doc names in its own Pitfalls section",
        ),
        practice.Check(
            "ANSWER: one line buys the 10 points, and it is not a language model",
            gain > 0.10,
            f"clearing `speech_ms` on a silent chunk while idle takes F1 from "
            f"{base['f1']:.3f} to {fixed['f1']:.3f} -- {gain * 100:+.1f} points -- by removing "
            f"one false positive ({base['fp']} to {fixed['fp']}). The target the exercise sets "
            "for a 3-layer MLP is met by a state-machine fix",
        ),
        practice.Check(
            "MECHANISM: `min_speech_ms` counts cumulative speech, not consecutive",
            set(result["clatter"].values()) == {13} and result["clatter_fixed"] is None,
            f"`update` never clears `speech_ms` in the idle state, so 20 ms transients "
            f"accumulate across any amount of silence: the 13th fires a START at both spacings "
            f"tried ({result['clatter']}). The doc calls this guard a rejector of 'speech "
            f"shorter than 250 ms'; it rejects the first twelve coughs. With the fix, 40 "
            "transients fire nothing",
        ),
        practice.Check(
            "CONTROL: the remaining error is the one an MLP is for, and there are no words",
            fixed["fp"] == 1 and result["hangover"][700]["f1"] == 1.0
            and result["hangover"][900]["f1"] == 0.0 and len(result["absent"]) == len(ABSENT),
            f"the residual false positive is the 640 ms mid-turn pause, which no counter fixes; "
            f"a 700 ms hangover reaches {result['hangover'][700]['f1']:.3f} but at 900 ms every "
            f"predicted end falls outside the +/-{COLLAR} ms collar and F1 is "
            f"{result['hangover'][900]['f1']:.3f}. find_spec is None for {result['absent']}, "
            "and `synth_chunk` returns Gaussian noise -- there is no text to embed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
