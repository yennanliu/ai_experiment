"""Exercise 2 — the gate fires after the budget is spent, and 13-gram and exact match disagree by 2.5 points.

    Add a "contamination check" gate. Given the eval dataset hash and the
    training dataset shards, compute the overlap (exact string match or 13-gram
    match). The gate fails if overlap exceeds 0.1%. Feed it a contaminated
    training set and confirm the gate holds the run.

Reading of the exercise: both overlap measures the exercise offers are built,
because "exact string match or 13-gram match" reads as interchangeable and the
case contamination checks exist for is the one where they are not. The gate
itself is a `DEFAULT_GATES` entry and is applied through the lesson's own
`gate()`, which is where the finding is: `gate` runs on `manifest.eval_metrics`,
and `run` fills those in on its last line.

**ANSWER: the gate holds the ship, not the run.** `run` executes all twelve
stages, accumulates `total_cost_usd` to **$1,941** and only then assigns
`eval_metrics`; `gate` is a separate call that reads them. A contaminated
training set is detected after the pipeline has paid for pre-training, SFT, two
policies and CAI -- **100%** of the budget it was going to spend. Nothing in
`run` consults a gate except the budget check.

**FINDING: the two measures the exercise offers disagree exactly where it
matters.** On 200 eval items against 2,000 training shards:

    clean                13-gram 0.000%   exact 0.000%   pass
    1 verbatim leak      13-gram 0.500%   exact 0.500%   HOLD
    5 verbatim leaks     13-gram 2.500%   exact 2.500%   HOLD
    5 embedded leaks     13-gram 2.500%   exact 0.000%   HOLD / pass

An eval item pasted into the middle of a longer training line is invisible to
exact match and fully visible to 13-gram. "Exact string match **or** 13-gram
match" is the one choice in the exercise that changes the answer.

**FINDING: 0.1% is below the resolution of the eval set.** One leaked item of
200 is **0.500%** overlap -- five times the threshold -- and the finest non-zero
reading the metric has is `1/200`. To distinguish 0.1% from 0% the eval set
would need at least 1,000 items, so on anything smaller the gate is binary:
either nothing leaked, or the threshold is already exceeded several times over.

**MECHANISM: the eval dataset hash cannot be used for this.** The exercise says
"given the eval dataset hash", and a SHA-256 supports one operation -- equality.
Overlap needs the shards' contents, which the pipeline never stores: every
artifact in the `ArtifactStore` is a `simulate_stage` metadata blob.

Structure: `grams` and `exact_overlap` are the two measures; `corpus` builds the
clean and contaminated shard sets; `check` runs the gate through the lesson's
own `gate()` with the metric injected.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "13-building-complete-llm-pipeline"
SEED, EVALS, SHARDS, WORDS, NGRAM = 7, 200, 2000, 20, 13
THRESHOLD = 0.001
CASES = (("clean", 0, 0), ("1 verbatim leak", 1, 0),
         ("5 verbatim leaks", 5, 0), ("5 embedded leaks", 0, 5))


def grams(text, n=NGRAM):
    words = text.split()
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def corpus(leak, embed):
    """`EVALS` eval items and `SHARDS` training lines, with `leak` copied and `embed` buried."""
    rng = random.Random(SEED)
    vocab = [f"w{i}" for i in range(400)]
    line = lambda length: " ".join(rng.choice(vocab) for _ in range(length))  # noqa: E731
    evals = [line(WORDS) for _ in range(EVALS)]
    train = [line(WORDS) for _ in range(SHARDS)]
    for i in range(leak):
        train[i] = evals[i]
    for i in range(embed):
        train[leak + i] = f"{line(6)} {evals[leak + i]} {line(6)}"
    return evals, train


def overlaps(evals, train):
    """Both measures the exercise offers, on the same two sets."""
    eval_grams = set().union(*[grams(t) for t in evals])
    train_grams, lines = set().union(*[grams(t) for t in train]), set(train)
    return {"ngram": len(eval_grams & train_grams) / max(len(eval_grams), 1),
            "exact": sum(1 for t in evals if t in lines) / len(evals)}


def check(ref, overlap):
    """The contamination gate as a DEFAULT_GATES entry, applied by the lesson's own gate()."""
    manifest = ref.Manifest()
    manifest.gates = dict(ref.DEFAULT_GATES, contamination={"op": "<=", "value": THRESHOLD})
    manifest.eval_metrics = dict(mmlu=68.4, humaneval=42.1, truthfulqa=53.7,
                                 safety_refusal_rate=0.03, kl_from_reference=18.5,
                                 cost_total_usd=0.0, contamination=overlap)
    return ref.gate(manifest)[0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {}
    for label, leak, embed in CASES:
        measured = overlaps(*corpus(leak, embed))
        rows[label] = dict(measured, ngram_ships=check(ref, measured["ngram"]),
                           exact_ships=check(ref, measured["exact"]))
    spent = ref.run(ref.Manifest(), ref.ArtifactStore())
    return {"rows": rows, "spent": spent.total_cost_usd, "stages_run": len(spent.stages),
            "metrics_set": bool(spent.eval_metrics), "resolution": 1 / EVALS,
            "shippable_before_gate": spent.shippable}


def column(rows, field, fmt):
    return ", ".join(f"{name} {format(row[field], fmt)}" for name, row in rows.items())


def verify(result):
    rows = result["rows"]
    embedded, single = rows["5 embedded leaks"], rows["1 verbatim leak"]
    return [
        practice.Check(
            "ANSWER: the gate holds the ship, not the run -- 100% of the budget is already spent",
            result["metrics_set"] and not result["shippable_before_gate"],
            f"run executes all {result['stages_run']} stages, accumulates total_cost_usd to "
            f"${result['spent']:,.0f} and only then assigns eval_metrics; gate is a separate call "
            f"that reads them. A contaminated training set is caught after pre-training, SFT, two "
            f"policies and CAI have been paid for, and shippable is still "
            f"{result['shippable_before_gate']} when run returns because nothing in run consults "
            "a gate except the budget check",
        ),
        practice.Check(
            "FINDING: the two measures disagree exactly where contamination checks are for",
            embedded["ngram"] > THRESHOLD and embedded["exact"] == 0.0,
            "on 200 eval items against 2,000 shards the 13-gram overlap is "
            + column(rows, "ngram", ".3%") + " and the exact-match overlap "
            + column(rows, "exact", ".3%")
            + f". An eval item pasted into the middle of a longer training line reads "
            f"{embedded['ngram']:.3%} to one measure and {embedded['exact']:.3%} to the other, so "
            "'exact string match or 13-gram match' is the one choice in the exercise that changes "
            "the answer",
        ),
        practice.Check(
            "FINDING: 0.1% is below the resolution of the eval set",
            single["ngram"] == result["resolution"] > THRESHOLD,
            f"one leaked item of {EVALS} is {single['ngram']:.3%} overlap, "
            f"{single['ngram'] / THRESHOLD:.0f}x the threshold, and the finest non-zero reading "
            f"the metric has is 1/{EVALS}. To distinguish {THRESHOLD:.1%} from zero the eval set "
            f"would need at least {int(1 / THRESHOLD):,} items, so on anything smaller the gate "
            "is binary -- either nothing leaked or the threshold is already exceeded several "
            "times over",
        ),
        practice.Check(
            "MECHANISM: the eval dataset hash cannot be used for this",
            rows["clean"]["ngram_ships"] and not single["ngram_ships"],
            "the exercise says 'given the eval dataset hash', and a SHA-256 supports one "
            "operation: equality. Overlap needs the shards' contents, which this pipeline never "
            "stores -- every artifact in the ArtifactStore is a simulate_stage metadata blob of "
            "stage name, type, input hashes and seed. The gate has to be fed a number computed "
            "outside the pipeline, which is what the ships/holds column above is: "
            + ", ".join(f"{name} {'ship' if row['ngram_ships'] else 'HOLD'}"
                        for name, row in rows.items()),
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
