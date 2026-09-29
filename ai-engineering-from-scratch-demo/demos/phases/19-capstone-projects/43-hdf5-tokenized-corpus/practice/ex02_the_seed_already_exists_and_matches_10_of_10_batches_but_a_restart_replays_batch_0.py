"""Exercise 2 — the seed already exists and two seed-7 runs match on 10 of 10 batches, but a restart replays batch 0.

    Add a deterministic seed to the sliding-window dataloader and verify two runs with the same seed produce identical batches.

Reading of the exercise: the reference `SlidingWindowDataloader` already takes
`seed: int = 0` and draws starts from a private `random.Random(seed)`. So
nothing needs adding, and the exercise reduces to the verification. "Two runs"
are two loaders built from scratch over the shipped demo corpus (window 64,
batch 4, seed 7, 10 batches), compared on inputs and targets. They are also
compared with the checksums that `run_demo()` prints, which come from a
separate write of the corpus. A seed only earns its name if it survives what
a training run does to it, so two more cases are measured: a restart from a
checkpoint, and a second worker.

**ANSWER: identical, on 10 of 10 batches (inputs and targets).** The two
seed-7 loaders agree exactly. Their blake2b checksums (e2747ebe, 14bc5794,
fe65c4c3, ...) equal all 10 that `run_demo()` prints from its own corpus
write. The batches are also identical when the shards are written with chunk
64 instead of 512, since the chunking does not change the data. Seed 8
matches 0 of 10 batches.

**FINDING: the seed makes a run reproducible, not resumable.** The loader
exposes no RNG state. A restart after batch 5 that rebuilds it with the same
seed replays from batch 0: it matches 0 of the 5 batches the interrupted run
would have produced next. Resuming takes either replaying the 20 start draws
through `_random`, which needs no token reads, or copying `_random.getstate()`
into the checkpoint. Both give 5 of 5. The demo corpus itself says
checkpoints record "the random seed so that a restart resumes exactly where it
stopped". That is not enough here.

**FINDING: one seed across workers duplicates every sample.** The lesson plans
for 16 dataloader workers. Two workers built with seed 7 produce the same 40
windows, so 100% of the second worker's samples repeat the first's. Seeds
must be offset per worker (seed + worker_id), or the effective batch shrinks
16x.

Structure: `batches()` takes n batches from a fresh loader; `resume()` compares
three restarts against the uninterrupted run.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import pathlib
import re
import tempfile

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "43-hdf5-tokenized-corpus"


def checksum(inputs):
    return f"{int(hashlib.blake2b(inputs.tobytes(), digest_size=4).hexdigest(), 16):08x}"


def same(a, b):
    return sum(bool((x[0] == y[0]).all() and (x[1] == y[1]).all()) for x, y in zip(a, b))


def build(ref, tmp, chunk):
    pipe = ref.ShardedTokenizationPipeline(ref.Tokenizer(), pathlib.Path(tmp) / str(chunk), chunk)
    return ref.MmapTokenStore(pipe.write_corpus(ref.build_demo_corpus()))


def batches(ref, store, seed, n=10):
    return ref.SlidingWindowDataloader(store, window_size=64, batch_size=4, seed=seed).take(n)


def resume(ref, store):
    """Batches 5..9 of a seed-7 run, versus three ways to restart after batch 5."""
    full = batches(ref, store, 7)[5:]
    naive = batches(ref, store, 7, 5)
    replay = ref.SlidingWindowDataloader(store, 64, 4, seed=7)
    for _ in range(5 * 4):
        replay._random.randint(0, replay._max_start)
    saved = ref.SlidingWindowDataloader(store, 64, 4, seed=7)
    saved.take(5)
    restored = ref.SlidingWindowDataloader(store, 64, 4, seed=7)
    restored._random.setstate(saved._random.getstate())
    return [same(full, naive), same(full, replay.take(5)), same(full, restored.take(5))]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        ref.run_demo()
    with tempfile.TemporaryDirectory() as tmp, build(ref, tmp, 512) as store, \
            build(ref, tmp, 64) as rechunked:
        a, b, c = batches(ref, store, 7), batches(ref, store, 7), batches(ref, store, 8)
        worker = batches(ref, store, 7)
        return {
            "same_seed": same(a, b), "other_seed": same(a, c),
            "rechunked": same(a, batches(ref, rechunked, 7)),
            "ours": [checksum(x) for x, _ in a],
            "demo": re.findall(r"checksum=([0-9a-f]{8})", log.getvalue()),
            "resume": resume(ref, store),
            "dup_workers": sum(int((x == y).all(axis=1).sum()) for (x, _), (y, _) in zip(worker, a)),
        }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: two seed-7 runs produce identical batches",
            r["same_seed"] == 10 and r["ours"] == r["demo"] and len(r["demo"]) == 10
            and r["ours"][:3] == ["e2747ebe", "14bc5794", "fe65c4c3"]
            and r["rechunked"] == 10 and r["other_seed"] == 0,
            f"seed 7 vs seed 7: {r['same_seed']}/10 identical; checksums {r['ours'][:3]}... equal "
            f"run_demo's 10: {r['ours'] == r['demo']}; chunk 64 vs 512: {r['rechunked']}/10; "
            f"seed 8: {r['other_seed']}/10",
        ),
        practice.Check(
            "FINDING: the seed makes a run reproducible, not resumable",
            r["resume"] == [0, 5, 5],
            f"restart after batch 5 matches the next 5 batches: rebuild {r['resume'][0]}/5, "
            f"replay 20 draws {r['resume'][1]}/5, restore RNG state {r['resume'][2]}/5",
        ),
        practice.Check(
            "FINDING: one seed across workers duplicates every sample",
            r["dup_workers"] == 40,
            f"a second seed-7 worker repeats {r['dup_workers']}/40 of the first worker's windows",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
