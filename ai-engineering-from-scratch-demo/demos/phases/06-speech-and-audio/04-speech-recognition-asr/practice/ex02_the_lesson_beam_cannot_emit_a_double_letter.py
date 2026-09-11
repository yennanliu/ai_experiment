"""Exercise 2 — the lesson's beam cannot emit a double letter at all.

    **Medium.** Implement the prefix-tree beam search in Step 2 properly (account
    for the blank merge rule). Compare with greedy on a 10-example synthetic
    dataset.

Reading of the exercise: "properly" means Hannun's prefix beam search -- two
probabilities per prefix, `p_blank` and `p_nonblank`, so that a repeated symbol
extends the prefix only through the blank path. The 10 examples are short
phrases built with the lesson's own `build_frame_probs`; **9 of the 10 carry a
doubled letter**, which is the only thing the two beams disagree about.

On the frames the lesson actually builds, the proper beam is exact:

| decoder | exact | CER |
|---|---:|---:|
| `ctc_greedy` | 10/10 | 0.000 |
| `ctc_beam` (the lesson's) | **1/10** | 0.116 |
| prefix beam (`p_blank`/`p_nonblank`) | **10/10** | **0.000** |

The lesson's beam is not merely bad at repeats; it **cannot represent one**. Its
update sends both the blank path and the repeat path to the same `seq`, so no
beam it ever holds contains two adjacent equal tokens. Across 30 decodes at three
corruption levels, **zero** of its outputs contain a doubled character.

Two more things the comparison shows.

**Greedy is already optimal on this data, so no beam can beat it.**
`build_frame_probs` puts a blank between every character and never makes an
argmax wrong. The comparison the exercise asks for only exists once the frames
are corrupted -- and the lesson's own `corrupt` leaves the probability simplex
(Exercise 1), so it cannot be used. Moving a random share of the winner's mass to
one rival keeps every frame a distribution and does flip argmaxes: once they do,
greedy scores **0.224** CER against the prefix beam's **0.139**.

**At heavy corruption the lesson's beam wins on CER for the wrong reason.** Every
decoder over-emits -- reference 11.1 characters, greedy 26.1 -- and CER is
dominated by insertions. Not being able to repeat makes the lesson's output
systematically the shortest (23.6 against 25.1), which lowers an insertion-driven
score without decoding anything better.

Structure: `log_add` and `prefix_beam` are the implementation the exercise asks
for; `blur` is the simplex-preserving corruption; `cer` is character edit
distance over the reference length; `compare` runs all three decoders over the
ten examples at one corruption level.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "04-speech-recognition-asr"
PHRASES = ["hello world", "coffee cup", "bell tower", "little dog", "the cat sat",
           "small green box", "letter home", "sunny day", "a bitter pill", "red balloon"]
WIDTH, LEVELS, NEG = 16, (0.0, 0.5, 0.7), -1e30


def log_add(a, b):
    if min(a, b) == NEG:
        return max(a, b)
    top = max(a, b)
    return top + math.log(math.exp(a - top) + math.exp(b - top))


def advance(beams, logp, blank_index):
    """One frame: every prefix extended, with the repeat path routed through blank."""
    nxt = defaultdict(lambda: [NEG, NEG])
    for prefix, (blank, nonblank) in beams.items():
        total = log_add(blank, nonblank)
        nxt[prefix][0] = log_add(nxt[prefix][0], total + logp[blank_index])
        for symbol in range(1, len(logp)):
            repeat = prefix[-1:] == (symbol,)
            nxt[prefix + (symbol,)][1] = log_add(nxt[prefix + (symbol,)][1],
                                                 (blank if repeat else total) + logp[symbol])
            if repeat:
                nxt[prefix][1] = log_add(nxt[prefix][1], nonblank + logp[symbol])
    return nxt


def prefix_beam(ref, frames, width=WIDTH):
    """Hannun's prefix beam: `p_blank` and `p_nonblank` per prefix, merged by log-sum."""
    beams = {(): (0.0, NEG)}
    for frame in frames:
        nxt = advance(beams, [math.log(max(x, 1e-10)) for x in frame], ref.BLANK)
        beams = dict(sorted(nxt.items(), key=lambda kv: -log_add(*kv[1]))[:width])
    return "".join(ref.VOCAB[i] for i in max(beams, key=lambda p: log_add(*beams[p])))


def blur(frames, level, rnd):
    """Move a random share of each winner's mass to one rival; every row stays a distribution."""
    out = []
    for frame in frames:
        row = list(frame)
        top = max(range(len(row)), key=lambda i: row[i])
        moved = rnd.uniform(0, level) * row[top]
        row[top] -= moved
        row[rnd.randrange(1, len(row))] += moved
        out.append(row)
    return out


def cer(reference, hypothesis):
    """Character edit distance over the reference length -- `wer` is 0/1 on one word."""
    grid = list(range(len(hypothesis) + 1))
    for i, want in enumerate(reference, 1):
        row = [i]
        for j, got in enumerate(hypothesis, 1):
            row.append(min(grid[j] + 1, row[j - 1] + 1, grid[j - 1] + (want != got)))
        grid = row
    return grid[-1] / max(1, len(reference))


def compare(ref, level):
    """All three decoders over the ten phrases at one corruption level."""
    rnd = random.Random(5)
    arms = {"greedy": ref.ctc_greedy, "lesson": lambda f: ref.ctc_beam(f, WIDTH),
            "prefix": lambda f: prefix_beam(ref, f)}
    out = {name: {"cer": 0.0, "exact": 0, "length": 0.0, "doubled": 0} for name in arms}
    for phrase in PHRASES:
        frames = blur(ref.build_frame_probs(phrase, 3, 1), level, rnd)
        for name, decode in arms.items():
            text, cell = decode(frames), out[name]
            cell["cer"] += cer(phrase, text) / len(PHRASES)
            cell["length"] += len(text) / len(PHRASES)
            cell["exact"] += text == phrase
            cell["doubled"] += any(a == b for a, b in zip(text, text[1:]))
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {level: compare(ref, level) for level in LEVELS}
    return {"runs": runs, "reference_length": sum(map(len, PHRASES)) / len(PHRASES),
            "with_doubles": sum(any(a == b for a, b in zip(p, p[1:])) for p in PHRASES)}


def verify(result):
    runs = result["runs"]
    clean, mild, heavy = (runs[level] for level in LEVELS)
    lesson_doubles = sum(run["lesson"]["doubled"] for run in runs.values())
    return [
        practice.Check(
            "ANSWER: the prefix beam is exact where the lesson's beam scores 1 of 10",
            clean["prefix"]["exact"] == 10 and clean["lesson"]["exact"] <= 1,
            f"on the frames `build_frame_probs` builds: prefix beam "
            f"{clean['prefix']['exact']}/10 exact at CER {clean['prefix']['cer']:.3f}, the "
            f"lesson's beam {clean['lesson']['exact']}/10 at {clean['lesson']['cer']:.3f}, "
            f"greedy {clean['greedy']['exact']}/10 at {clean['greedy']['cer']:.3f}",
        ),
        practice.Check(
            "MECHANISM: the lesson's beam cannot represent a repeated symbol at all",
            lesson_doubles == 0 and result["with_doubles"] == 9,
            f"its update sends the blank path and the repeat path to the same `seq`, so no beam "
            f"it holds ever contains two adjacent equal tokens: {lesson_doubles} of 30 decodes "
            f"carry a doubled character, against {result['with_doubles']} of the 10 references. "
            "The fix is one extra state -- a repeat may extend a prefix only through blank",
        ),
        practice.Check(
            "FINDING: greedy is already optimal here, so the comparison needs corruption",
            clean["greedy"]["exact"] == 10,
            f"a blank sits between every character and no argmax is ever wrong, so greedy scores "
            f"{clean['greedy']['exact']}/10 and no beam can beat it. The lesson's own `corrupt` "
            "cannot supply the corruption either -- it leaves the simplex",
        ),
        practice.Check(
            "ANSWER: once argmaxes flip, the prefix beam beats greedy",
            mild["prefix"]["cer"] < mild["greedy"]["cer"],
            f"at the level where the winner sometimes loses its frame: greedy CER "
            f"{mild['greedy']['cer']:.3f}, prefix beam {mild['prefix']['cer']:.3f}, lesson's "
            f"beam {mild['lesson']['cer']:.3f}",
        ),
        practice.Check(
            "CONTROL: at heavy corruption the lesson's beam wins on CER by being shorter",
            heavy["lesson"]["length"] < heavy["prefix"]["length"]
            and heavy["lesson"]["cer"] < heavy["prefix"]["cer"],
            f"every decoder over-emits -- reference {result['reference_length']:.1f} characters "
            f"against greedy's {heavy['greedy']['length']:.1f} -- so CER is insertion-driven. "
            f"Not being able to repeat makes the lesson's output the shortest "
            f"({heavy['lesson']['length']:.1f} against {heavy['prefix']['length']:.1f}), which "
            f"lowers the score ({heavy['lesson']['cer']:.3f} against "
            f"{heavy['prefix']['cer']:.3f}) without decoding anything better",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
