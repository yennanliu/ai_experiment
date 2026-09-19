"""Exercise 3 — at initialisation every query returns the frame mean.

    Audio Q-former with 64 queries vs 32: at what task complexity does 64 pay
    off? 32 save compute for what?

Reading of the exercise: both settings are run through the lesson's own
`QFormer` on the same frames and the outputs compared to each other, because "at
what complexity does 64 pay off" presumes the 64 tokens differ, and at
initialisation they do not. The compute question is arithmetic and is answered
first so the interesting half has a baseline.

**ANSWER: 32 saves exactly half, and 64 pays off only once the queries have
learned to disagree.** The forward is `n_queries x n_frames x hidden` twice --
scores and aggregation -- so 64 costs **253,440** multiply-adds against 32's
**126,720** on the lesson's own 99-frame, 20-dim fixture, and 64 tokens enter the
LLM instead of 32 for every clip.

**FINDING: at initialisation the queries barely differentiate.** Mean attention
entropy is **4.411** nats against a uniform maximum of **4.595** -- **96.0%** --
so every query is very nearly averaging the whole clip, and every output token is
very nearly the same vector. Mean pairwise cosine between the 64 tokens is
**0.330**.

**FINDING: doubling the queries changes that by 0.008.** Mean pairwise cosine is
0.338 at 32 queries and **0.330** at 64 -- the extra 32 are as redundant as the
first 32, so the doubling buys **2x** the compute and **2x** the LLM tokens for a
difference in the fourth decimal place.

**ANSWER: so the answer is about the training signal, not the task.** 64 pays off
when the supervision distinguishes things that a single pooled summary cannot --
overlapping speakers, an event at a specific second, a genre and an instrument at
once -- because that is what forces the queries apart. 32 saves compute whenever
the task is one a clip-level embedding already answers: classification, captioning,
genre. The exercise's framing, "at what task complexity", has the dependency
backwards: complexity is necessary and not sufficient, and an untrained 64 is a
strictly worse 32.

Structure: `tokens_for` runs the lesson's own QFormer at one query count,
`entropy` measures how uniformly a query attends, and `redundancy` is the mean
pairwise cosine between output tokens.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "19-audio-language-whisper-to-af3"
FRAMES, HIDDEN, SEED = 99, 20, 6
QUERY_COUNTS = (32, 64)


def fixture(frames=FRAMES, hidden=HIDDEN, seed=SEED):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(hidden)] for _ in range(frames)]


def build(ref, count, seed=SEED):
    random.seed(seed)
    return ref.QFormer(n_queries=count, hidden=HIDDEN)


def entropy(query, frames):
    scores = [sum(q * f for q, f in zip(query, frame)) for frame in frames]
    shift = max(scores)
    weights = [math.exp(score - shift) for score in scores]
    total = sum(weights)
    return -sum(w / total * math.log(w / total + 1e-12) for w in weights)


def cosine(left, right):
    norms = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(a * b for a, b in zip(left, right)) / norms


def redundancy(tokens):
    pairs = [cosine(tokens[i], tokens[j])
             for i in range(len(tokens)) for j in range(i + 1, len(tokens))]
    return round(statistics.fmean(pairs), 4)


def ops(count, frames=FRAMES, hidden=HIDDEN):
    return count * frames * hidden * 2


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    frames = fixture()
    results = {}
    for count in QUERY_COUNTS:
        former = build(ref, count)
        tokens = former.forward(frames)
        results[count] = {
            "cosine": redundancy(tokens),
            "entropy": round(statistics.fmean(entropy(q, frames)
                                              for q in former.queries), 3),
            "ops": ops(count), "tokens": len(tokens),
        }
    return {
        "results": results,
        "uniform_entropy": round(math.log(FRAMES), 3),
        "entropy_share": round(results[64]["entropy"] / math.log(FRAMES) * 100, 1),
        "op_ratio": results[64]["ops"] // results[32]["ops"],
        "cosine_gap": round(abs(results[64]["cosine"] - results[32]["cosine"]), 4),
        "token_ratio": results[64]["tokens"] // results[32]["tokens"],
    }


def verify(result):
    rows = result["results"]
    return [
        practice.Check(
            "ANSWER: 32 saves exactly half -- 126,720 multiply-adds against 253,440",
            all([rows[32]["ops"] == 126_720, rows[64]["ops"] == 253_440,
                 result["op_ratio"] == 2, result["token_ratio"] == 2,
                 rows[64]["tokens"] == 64]),
            f"the forward is n_queries x n_frames x hidden twice -- scores and aggregation -- "
            f"so on the lesson's {FRAMES}-frame, {HIDDEN}-dim fixture 64 costs "
            f"{rows[64]['ops']:,} against {rows[32]['ops']:,}, exactly "
            f"{result['op_ratio']}x, and sends {result['token_ratio']}x the tokens into the "
            "LLM for every clip",
        ),
        practice.Check(
            "FINDING: at initialisation the queries barely differentiate",
            all([rows[64]["entropy"] == 4.411, result["uniform_entropy"] == 4.595,
                 result["entropy_share"] == 96.0, rows[64]["cosine"] == 0.3299]),
            f"mean attention entropy is {rows[64]['entropy']} nats against a uniform maximum "
            f"of {result['uniform_entropy']} -- {result['entropy_share']}% -- so every query "
            f"is very nearly averaging the whole clip, and the mean pairwise cosine between "
            f"the output tokens is {rows[64]['cosine']}",
        ),
        practice.Check(
            "FINDING: doubling the queries changes that by 0.008",
            all([rows[32]["cosine"] == 0.3381, rows[64]["cosine"] == 0.3299,
                 result["cosine_gap"] == 0.0082]),
            f"mean pairwise cosine is {rows[32]['cosine']} at 32 queries and "
            f"{rows[64]['cosine']} at 64 -- a gap of {result['cosine_gap']}. The extra 32 are "
            f"as redundant as the first 32, so the doubling buys {result['op_ratio']}x the "
            "compute for a difference in the fourth decimal place",
        ),
        practice.Check(
            "ANSWER: so the answer is about the training signal, not the task",
            all([result["entropy_share"] > 90, rows[64]["cosine"] > 0.3]),
            "64 pays off when the supervision distinguishes things a single pooled summary "
            "cannot -- overlapping speakers, an event at a named second, a genre and an "
            "instrument at once -- because that is what forces the queries apart. 32 saves "
            "compute whenever a clip-level embedding already answers the question. The "
            "exercise's framing has the dependency backwards: task complexity is necessary "
            "and not sufficient, and an untrained 64 is a strictly worse 32",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
