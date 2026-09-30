"""Exercise 4 -- 1% of the training data leaks 20% of the benchmark, and the lesson's check still says clean.

    Inject 1% of contamination (leak MMLU-Pro answers into training data) and rerun eval. Watch MMLU-Pro accuracy jump unrealistically. Build a contamination-check CI gate that catches this.

Reading of the exercise: MMLU-Pro cannot be downloaded at run time, so it is
stood in for by 1,000 synthetic 10-option items: 12-20 words from a
2,000-word vocabulary, then 10 option words. The training set is 20,000
random documents. Contamination replaces 1% of them (200 documents) with
benchmark items plus "answer: X". Half are verbatim. The other half are
edited: 2 words dropped and the text upper-cased. The model under eval has a
fixed genuine skill: it answers an item right when the item's crc32 mod 100
is below 45. It also memorises training text: if a training document holds
at least 30% of an item's word 3-grams, the model gives that document's
leaked answer. The gate is MinHash LSH (64 hashes, 32 bands of 2) over
lower-cased word 3-grams. Each candidate pair is confirmed at Jaccard >= 0.5,
and the gate fails CI on any hit.

**ANSWER: benchmark accuracy jumps from 0.443 to 0.571, and the gate catches
all 200 leaked documents.** One percent of the training data is 20% of the
benchmark, and 128 of those items were ones the model had got wrong. On the
clean corpus the gate passes with 0 flags. On the contaminated corpus it
fails with 200 flags: 100/100 verbatim and 100/100 edited, with 0 false
positives. Dropping the flagged documents brings accuracy back to 0.443.

**FINDING: banding tuned for near-verbatim copies lets edited leaks
through.** With 16 bands of 4, the same 64 hashes catch 195 of the 200
leaked documents but only 95 of the 100 edited ones. The 5 it misses still
score, and accuracy after that gate is 0.445.

**FINDING: the lesson's contamination check cannot fail.** `stage_contamination`
hard-codes `overlap_examples: 0` for all three benchmarks and never reads
training text. The `dataset` artifact holds only counts, so there is no
text to read. Given a manifest that declares the 200 leaked documents, it
still returns "clean".

**FINDING: at the lesson's scale, 1% is a fifth of MMLU-Pro.** The pipeline
keeps 255,336 examples, and 1% of them is 2,553 documents. That is 21.2% of
MMLU-Pro's 12,032 questions (arXiv:2406.01574, read 2026-09-29).

Structure: `make_fixture()` builds the benchmark and corpora; `minhash()` and
`gate()` are the CI check; `accuracy()` is the memorising model's eval.
"""

from __future__ import annotations

import zlib

import numpy as np

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "07-end-to-end-fine-tuning-pipeline"
N_BENCH, N_TRAIN, LEAK, SKILL, PERMS, BANDS, MMLU_PRO = 1000, 20_000, 0.01, 45, 64, 32, 12_032
P = np.uint64((1 << 31) - 1)  # a*x < 2^62 stays exact in uint64, and the mod really wraps


def make_fixture(seed=7):
    r = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(2000)]
    text = lambda n: " ".join(r.choice(words, n))  # noqa: E731
    bench = [(text(r.integers(12, 21)) + " options " + text(10), "ABCDEFGHIJ"[r.integers(10)]) for _ in range(N_BENCH)]
    clean = [text(r.integers(30, 61)) for _ in range(N_TRAIN)]
    dirty, leaked = list(clean), r.choice(N_BENCH, int(N_TRAIN * LEAK), replace=False)
    for k, i in enumerate(leaked):
        q = " ".join(w for j, w in enumerate(bench[i][0].split()) if j not in (1, 7)).upper() if k % 2 else bench[i][0]
        dirty[k * 97] = q + f" answer: {bench[i][1]}"
    return bench, clean, dirty, {k * 97 for k in range(len(leaked))}


def shingles(text):
    t = text.lower().split()
    return {zlib.crc32(" ".join(t[i : i + 3]).encode()) for i in range(len(t) - 2)}


def minhash(sh, coef):
    x = np.fromiter(sh, dtype=np.uint64)[None, :] % P
    return ((coef[:, :1] * x + coef[:, 1:]) % P).min(1)


def index_of(keys_per_item):
    index = {}
    for i, keys in enumerate(keys_per_item):
        for k in keys:
            index.setdefault(k, set()).add(i)
    return index


def gate(train, bench, bands=BANDS):
    coef, rows = np.random.default_rng(0).integers(1, int(P), (PERMS, 2), dtype=np.uint64), PERMS // bands

    def band_keys(sh):
        sig = minhash(sh, coef)
        return [(b, sig[b * rows : (b + 1) * rows].tobytes()) for b in range(bands)]

    bench_sh = [shingles(q) for q, _ in bench]
    buckets = index_of(band_keys(sh) for sh in bench_sh)
    flagged = {d for d, sh in enumerate(map(shingles, train)) if any(
        len(sh & bench_sh[i]) / len(sh | bench_sh[i]) >= 0.5 for k in band_keys(sh) for i in buckets.get(k, ()))}
    return ("fail" if flagged else "pass"), flagged


def accuracy(train, bench):
    index, right = index_of(shingles(d.rsplit(" answer:", 1)[0]) if "answer:" in d else () for d in train), 0
    for q, gold in bench:
        sh = shingles(q)
        hits = sorted(d for d in set().union(*(index.get(s, ()) for s in sh))
                      if len(sh & shingles(train[d])) / len(sh) >= 0.3)
        right += train[hits[0]][-1] == gold if hits else zlib.crc32(q.encode()) % 100 < SKILL
    return round(right / len(bench), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    bench, clean, dirty, leaked = make_fixture()
    edited = {k * 97 for k in range(1, len(leaked), 2)}
    (s0, f0), (s1, f1), (_, f16) = gate(clean, bench), gate(dirty, bench), gate(dirty, bench, 16)
    m = ref.Manifest()
    with parity.quiet():
        m.add(ref.stage_data(m, {"raw_examples": 300_000, "seed": 7}))
        m.get("dataset").payload["leaked_docs"] = len(leaked)
        report = ref.stage_contamination(m, {}).payload
    kept = m.get("dataset").payload["after_pii_scrub"]
    return {
        "acc": [accuracy(t, bench) for t in (clean, dirty, [d for i, d in enumerate(dirty) if i not in f1])],
        "gate": [s0, len(f0), s1, len(f1)], "caught": [len(f1 & leaked), len(f1 & edited), len(f1 - leaked)],
        "b16": [len(f16 & leaked), len(f16 & edited), len(f16 - leaked)],
        "acc_b16": accuracy([d for i, d in enumerate(dirty) if i not in f16], bench),
        "ref": (report["status"], [o["overlap_examples"] for o in report["overlaps"]]),
        "fields": sorted(m.get("dataset").payload), "kept": kept, "share": round(kept // 100 / MMLU_PRO, 3)}


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: accuracy 0.443 -> 0.571 with 1% leaked; the MinHash gate catches 200/200",
            (r["acc"], r["gate"], r["caught"]) == ([0.443, 0.571, 0.443], ["pass", 0, "fail", 200], [200, 100, 0]),
            f"accuracy clean/leaked/after-gate {r['acc']}; gate clean {r['gate'][:2]}, leaked {r['gate'][2:]}; "
            f"caught/edited caught/false positives {r['caught']}",
        ),
        practice.Check(
            "FINDING: banding tuned for near-verbatim copies lets edited leaks through",
            (r["b16"], r["acc_b16"]) == ([195, 95, 0], 0.445),
            f"16 bands x 4: caught/edited/false positives {r['b16']}; accuracy after that gate {r['acc_b16']}",
        ),
        practice.Check(
            "FINDING: the lesson's contamination check cannot fail",
            r["ref"] == ("clean", [0, 0, 0]) and "text" not in r["fields"],
            f"with 200 leaked docs declared: {r['ref']}; dataset fields {r['fields']}",
        ),
        practice.Check(
            "FINDING: at the lesson's scale, 1% is a fifth of MMLU-Pro",
            (r["kept"], r["kept"] // 100, r["share"]) == (255336, 2553, 0.212),
            f"1% of {r['kept']:,} kept examples = {r['kept'] // 100:,} docs = {r['share']:.1%} of {MMLU_PRO:,}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
