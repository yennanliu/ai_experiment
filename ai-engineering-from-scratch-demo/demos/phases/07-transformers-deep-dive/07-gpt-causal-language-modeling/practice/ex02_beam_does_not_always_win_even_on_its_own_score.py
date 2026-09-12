"""Exercise 2 — beam does not always win, not even on the score it maximises.

    **Medium.** Implement beam search for width 4. Compare perplexity of beam-4
    vs greedy on 10 short prompts. Does beam always win? (Hint: usually for
    translation, not for open-ended chat.)

Reading of the exercise: the lesson has no model -- `demo_ce_loss` invents logits
with `rng.gauss` -- so a scorer is built here: a bigram table over the lesson's
own 20-token vocabulary, one row of log-probabilities per previous token, through
the lesson's own `softmax`. Greedy is beam width 1 in the same function, so the
two arms cannot differ by implementation.

**ANSWER: no -- and the counterexample is on beam's own objective.** Over 300
bigram models, beam-4's best hypothesis has a *lower* total log-probability than
greedy's **2 times**. Beam search is not guaranteed to dominate greedy, because
greedy's prefix can be pushed out of the beam by four better-scoring prefixes
whose own continuations are worse. Width is not monotone either: on one model,
width 2 scores **-18.674** where width 4 scores **-18.769**.

**FINDING: on 10 prompts beam wins 8, ties 2 and loses 0, which proves nothing.**
The mean gain is +1.66 nats of total log-probability and the outputs coincide
26% of the time across 300 models. A 10-prompt comparison at a 0.7% failure rate
is expected to see zero failures; the exercise's sample size cannot find the
answer to the exercise's question.

**FINDING: the perplexity the exercise names does not depend on the decoder.**
`cross_entropy_shifted(logits_per_pos, target_ids)` takes a model's logits and a
reference sequence. No decoding strategy appears in it, and none can: perplexity
is a property of a model on text it did not choose. What can be compared is the
perplexity of each decoder's **own output** -- greedy 4.885 against beam 4.255 --
and beam maximises exactly that quantity by construction, so it is a definition
rather than a result.

**CONTROL: width 1 is greedy, bit for bit.** Both arms run the same function, so
the 2 losses are a property of the search and not of two implementations.

Structure: `bigram` builds the scorer; `search` is beam search at any width, with
width 1 being greedy; `sweep` counts how often beam loses across many models;
`tally` summarises the ten prompts.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "07-gpt-causal-language-modeling"
VOCAB, STEPS, WIDTH, PROMPTS, MODELS = 20, 12, 4, 10, 300


def bigram(ref, seed):
    """One row of log-probabilities per previous token, through the lesson's own softmax."""
    rng = random.Random(seed)
    return [[math.log(p) for p in ref.softmax([rng.gauss(0, 1) for _ in range(VOCAB)])]
            for _ in range(VOCAB)]


def search(table, start, width, steps=STEPS):
    """Beam search at any width. Width 1 is greedy, in the same code path."""
    beams = [([start], 0.0)]
    for _ in range(steps):
        pool = [(seq + [v], score + table[seq[-1]][v])
                for seq, score in beams for v in range(VOCAB)]
        beams = sorted(pool, key=lambda pair: -pair[1])[:width]
    return beams[0]


def sweep(ref, models=MODELS):
    """(times beam-4 scores worse than greedy, times the outputs coincide)."""
    worse = same = 0
    for seed in range(models):
        table = bigram(ref, seed)
        greedy, beam = search(table, seed % VOCAB, 1), search(table, seed % VOCAB, WIDTH)
        worse += beam[1] < greedy[1] - 1e-12
        same += beam[0] == greedy[0]
    return worse, same


def tally(pairs):
    """(beam wins, identical outputs, beam losses, mean log-prob gain) over the prompts."""
    return {
        "wins": sum(1 for g, b in pairs if b[1] > g[1]),
        "ties": sum(1 for g, b in pairs if b[0] == g[0]),
        "losses": sum(1 for g, b in pairs if b[1] < g[1] - 1e-12),
        "gain": sum(b[1] - g[1] for g, b in pairs) / len(pairs),
    }


def reference_perplexity(ref, table, length=30, seed=5):
    """The lesson's own cross_entropy_shifted on a sequence no decoder produced."""
    target = [random.Random(seed).randrange(VOCAB) for _ in range(length)]
    logits = [[table[token][v] for v in range(VOCAB)] for token in target]
    return math.exp(ref.cross_entropy_shifted(logits, target))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    prompts = [(bigram(ref, 100 + i), i % VOCAB) for i in range(PROMPTS)]
    pairs = [(search(t, s, 1), search(t, s, WIDTH)) for t, s in prompts]
    worse, same = sweep(ref)
    return {
        "worse": worse, "same": same, "models": MODELS, **tally(pairs),
        "widths": {w: search(bigram(ref, 1), 0, w)[1] for w in (1, 2, 4, 8, 16)},
        "ppl": tuple(math.exp(-sum(p[i][1] for p in pairs) / PROMPTS / STEPS) for i in (0, 1)),
        "reference_ppl": reference_perplexity(ref, bigram(ref, 100)),
    }


def verify(result):
    widths, greedy_ppl, beam_ppl = result["widths"], *result["ppl"]
    return [
        practice.Check(
            "ANSWER: no -- beam-4 scores worse than greedy on 2 of 300 models",
            result["worse"] > 0,
            f"over {result['models']} bigram models, beam-{WIDTH}'s best hypothesis has a lower "
            f"total log-probability than greedy's {result['worse']} times "
            f"({result['worse'] / result['models']:.2%}). Greedy's prefix can be pushed out of "
            f"the beam by {WIDTH} better-scoring prefixes whose own continuations are worse, so "
            "beam search does not dominate greedy even on the objective it maximises",
        ),
        practice.Check(
            "ANSWER: and width is not monotone -- width 2 beats width 4 on one model",
            widths[2] > widths[4],
            f"total log-probability by width: "
            f"{ {w: round(v, 3) for w, v in widths.items()} }. Width 2 scores {widths[2]:.3f} "
            f"where width {WIDTH} scores {widths[4]:.3f}. Widening the beam changes which "
            "prefixes survive, and a surviving prefix is not the same thing as a better one",
        ),
        practice.Check(
            "FINDING: 10 prompts cannot find that answer",
            result["losses"] == 0 and result["wins"] >= 7,
            f"on the exercise's {PROMPTS} prompts beam wins {result['wins']}, ties "
            f"{result['ties']} and loses {result['losses']}, with a mean gain of "
            f"{result['gain']:+.2f} nats. At a {result['worse'] / result['models']:.2%} failure "
            f"rate, {PROMPTS} draws are expected to see none -- the sample size cannot answer the "
            "question the exercise asks with it",
        ),
        practice.Check(
            "FINDING: the perplexity the exercise names does not depend on the decoder",
            result["reference_ppl"] > 1,
            f"cross_entropy_shifted(logits_per_pos, target_ids) takes a model's logits and a "
            f"reference sequence and gives {result['reference_ppl']:.2f} here. No decoding "
            "strategy appears in the formula and none can: perplexity is a property of a model "
            "on text it did not choose. Swapping greedy for beam cannot move it by a bit",
        ),
        practice.Check(
            "CONTROL: what can be compared is each decoder's own output, and that is circular",
            beam_ppl < greedy_ppl,
            f"perplexity of the generated sequence under the model that generated it: greedy "
            f"{greedy_ppl:.3f} against beam {beam_ppl:.3f}. Beam maximises total log-probability, "
            "which is exactly the reciprocal of this, so the comparison is a definition rather "
            f"than a result -- and it still loses {result['worse']} times in {result['models']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
