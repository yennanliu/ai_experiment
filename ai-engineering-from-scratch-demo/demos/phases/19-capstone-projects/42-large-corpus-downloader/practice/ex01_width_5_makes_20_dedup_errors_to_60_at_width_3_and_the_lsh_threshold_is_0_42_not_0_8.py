"""Exercise 1 — width 5 makes 20 dedup errors to 60 at width 3 and 44 at width 9, and the LSH threshold is 0.42, not 0.8.

    Add a `--shingle-width` flag and measure how the dedup verdict changes at widths 3, 5, 9. Defend the chosen default.

Reading of the exercise: `--shingle-width` is an argparse flag whose value
builds the reference `MinHasher`; everything else is the reference pipeline
unchanged (zst shards on disk, `file://` download, `process_shard`, the
default k = 128, b = 32 LSH). "The dedup verdict" is scored on a labelled
480-document fixture: 300 distinct Zipf-word originals, 120 near-duplicates
of the four kinds the lesson names (new footer, licence header, 3 edited
words, a tracking parameter on every tenth token), and 60 distinct
form-like records that share a template and differ in every seventh token.
An error is a missed near-duplicate or a flagged non-duplicate. The same
three widths are also run over the lesson's own demo corpus.

**ANSWER: default to width 5 (the module's `DEFAULT_SHINGLE_WIDTH`).**

| width | footer | header | edit | tracking | templated flagged | errors |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 30/30 | 30/30 | 30/30 | 27/30 | 57/60 | 60 |
| 5 | 30/30 | 30/30 | 28/30 | 13/30 | 1/60 | 20 |
| 9 | 30/30 | 30/30 | 16/30 | 0/30 | 0/60 | 44 |

No distinct original is flagged at any width. Width 3 catches nearly every
near-duplicate but calls 57 of 60 distinct templated records duplicates,
because short shingles fit between the slots. Width 9 never flags a template
but each edited token now destroys 9 shingles, so it misses 14 edits and all
30 tracking variants. Width 5 has the fewest errors; its misses are the
tracking variants, which need URL normalisation, not a different width.

**FINDING: the lesson's own demo runs width 3, not the default 5, and that
changes its verdicts.** On the demo corpus width 3 flags 7 of 15 documents;
widths 5 and 9 flag 5, the exact repeats. The 2 extra are the two paraphrase
pairs, and three flags at width 3 sit on pairs with exact Jaccard 0.364, 0.5
and 0.5.

**FINDING: k = 128, b = 32, r = 4 is a 0.42 threshold, not the 0.8 the doc
says.** The S-curve midpoint is (1/32)^(1/4) = 0.420 (the `LSHIndex`
docstring says so itself); a pair at s = 0.5 collides with probability 0.873.
`Dedup` drops on the first band hit and never calls `jaccard_estimate`. At
width 5, 71 of the 102 flagged documents have exact Jaccard below 0.8 with
their keeper, the lowest 0.182.

Structure: `make_fixture()` builds the labelled corpus; `parser()` is the
flag; `run()` is the pipeline behind it; `measure()` scores one width.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import tempfile

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "42-large-corpus-downloader"
WIDTHS = (3, 5, 9)
KINDS = ("footer", "header", "edit", "tracking")
FOOTER = "share this article subscribe to our newsletter for weekly updates all rights reserved".split()
HEADER = "copyright 2024 licensed under the apache license version 2.0 see the license file".split()
TEMPLATE = [f"form{j}" for j in range(48)]


def edit(d, rng, vocab):
    for _ in range(3):
        d[rng.randrange(len(d))] = rng.choice(vocab)
    return d


VARIANTS = {
    "footer": lambda d, rng, vocab: d + FOOTER,
    "header": lambda d, rng, vocab: HEADER + d,
    "edit": edit,
    "tracking": lambda d, rng, vocab: [t + "?utm=feed" if j % 10 == 0 else t for j, t in enumerate(d)],
}


def form(i):
    """A form-like record: 6 fixed template words, then one per-record value, eight times."""
    return " ".join(w for j in range(8) for w in (*TEMPLATE[6 * j : 6 * j + 6], f"item{i}x{j}"))


def make_fixture(seed=42):
    """(text, label) rows: 300 originals, 120 labelled near-duplicates, 60 templated non-duplicates."""
    rng, vocab = random.Random(seed), [f"w{i}" for i in range(3000)]
    origs = [rng.choices(vocab, [1 / (i + 1) for i in range(3000)], k=rng.randint(30, 80)) for _ in range(300)]
    rows = [(" ".join(d), "original") for d in origs]
    rows += [(" ".join(VARIANTS[KINDS[i % 4]](list(origs[i]), rng, vocab)), KINDS[i % 4]) for i in range(120)]
    return rows + [(form(i), "templated") for i in range(60)]


def parser(ref):
    p = argparse.ArgumentParser(prog="corpus-downloader")
    p.add_argument("--shingle-width", type=int, default=ref.DEFAULT_SHINGLE_WIDTH)
    return p


def run(ref, argv, groups):
    """The flag drives the real pipeline: zst shards -> file:// download -> zstd -> MinHash -> LSH -> manifest."""
    args = parser(ref).parse_args(argv)
    dedup = ref.Dedup(ref.MinHasher(ref.DEFAULT_NUM_HASHES, args.shingle_width),
                      ref.LSHIndex(ref.DEFAULT_NUM_HASHES, ref.DEFAULT_BANDS))
    manifest = ref.ManifestWriter()
    with tempfile.TemporaryDirectory() as d:
        urls = []
        for i, lines in enumerate(groups):
            path = pathlib.Path(d) / f"s{i}.zst"
            path.write_bytes(ref.zstd.ZstdCompressor(level=3).compress(("\n".join(lines) + "\n").encode()))
            urls.append(path.as_uri())
        for plan in ref.ShardPlanner.from_urls(urls):
            ref.process_shard(plan, ref.StreamingDownloader(pathlib.Path(d) / "cache"), dedup, manifest)
    return manifest.verdicts


def measure(ref, groups, w):
    """Verdicts at width w, plus the exact shingle Jaccard of every flagged doc to its keeper."""
    verdicts = run(ref, ["--shingle-width", str(w)], groups)
    grams = ref.MinHasher(1, w).shingles
    sh = {f"shard-{s:04d}:{i}": set(grams(t)) for s, g in enumerate(groups) for i, t in enumerate(g)}
    pairs = [(sh[f"{v['shard_id']}:{v['doc_index']}"], sh[v["collided_with"]]) for v in verdicts if v["collided_with"]]
    return [v["verdict"] == "near_duplicate" for v in verdicts], [len(a & b) / len(a | b) for a, b in pairs]


def demo_groups(ref):
    with tempfile.TemporaryDirectory() as d:
        paths = [pathlib.Path(u[7:]) for u in ref.build_demo_corpus(pathlib.Path(d))]
        return [ref.zstd.ZstdDecompressor().decompress(f.read_bytes()).decode().splitlines() for f in paths]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = make_fixture()
    texts, table, demo = [t for t, _ in rows], {}, {}
    for w in WIDTHS:
        flags, sims = measure(ref, [texts[:300], texts[300:420], texts[420:]], w)
        row = {k: sum(f for f, (_, lab) in zip(flags, rows) if lab == k) for k in (*KINDS, "original", "templated")}
        row["errors"] = 120 - sum(row[k] for k in KINDS) + row["original"] + row["templated"]
        row.update(flagged=len(sims), below=sum(s < 0.8 for s in sims), min_j=round(min(sims), 3))
        table[w] = row
        dflags, dsims = measure(ref, demo_groups(ref), w)
        demo[w] = (sum(dflags), len(dflags), sorted(round(s, 3) for s in dsims if s < 1))
    return {"table": table, "default": parser(ref).parse_args([]).shingle_width, "demo": demo}


def verify(result):
    t, demo, doc = result["table"], result["demo"], parity.doc_text(PHASE, LESSON)
    got = {w: ([t[w][k] for k in KINDS], t[w]["templated"], t[w]["original"], t[w]["errors"]) for w in WIDTHS}
    s_curve = round((1 / 32) ** (1 / 4), 3), round(1 - (1 - 0.5**4) ** 32, 3)
    claim = ("`s = 0.8`" in doc, "`k = 128`, `b = 32`, `r = 4`" in doc)
    return [
        practice.Check(
            "ANSWER: width 5 has the fewest dedup errors of 3, 5, 9, and the flag defaults to it",
            (got, result["default"]) == ({3: ([30, 30, 30, 27], 57, 0, 60), 5: ([30, 30, 28, 13], 1, 0, 20),
                                          9: ([30, 30, 16, 0], 0, 0, 44)}, 5),
            f"width -> (caught footer/header/edit/tracking, templated flagged, originals flagged, errors): {got}; "
            f"default {result['default']}",
        ),
        practice.Check(
            "FINDING: the lesson's demo runs width 3, not the default 5, and that changes its verdicts",
            (demo[3], demo[5], demo[9]) == ((7, 15, [0.364, 0.5, 0.5]), (5, 15, []), (5, 15, [])),
            f"demo flags {demo[3][0]}/15 at width 3 (non-exact pairs at Jaccard {demo[3][2]}), "
            f"{demo[5][0]}/15 at 5, {demo[9][0]}/15 at 9",
        ),
        practice.Check(
            "FINDING: k = 128, b = 32, r = 4 is a 0.42 threshold, not the 0.8 the doc says",
            (claim, s_curve, t[5]["flagged"], t[5]["below"], t[5]["min_j"])
            == ((True, True), (0.42, 0.873), 102, 71, 0.182),
            f"midpoint {s_curve[0]}, P(collide | s=0.5) {s_curve[1]}; at width 5 {t[5]['below']} "
            f"of {t[5]['flagged']} flags sit below Jaccard 0.8, lowest {t[5]['min_j']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
