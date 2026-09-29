"""Exercise 1 — two stand-in models disagree less than one model does with itself across processes.

    Measure ColQwen2.5-v0.2 vs ColQwen3-omni on the same corpus. Which pages does one get right and the other miss? Add a "content class" tag to the index to route by type.

Reading of the exercise: no ColQwen checkpoint is in the lesson, so the two
"models" are the lesson's own embedder at two sizes: `EMB_DIM = 16` as
shipped (stand-in A) and `EMB_DIM = 128`, the patch dimension the doc gives
for ColQwen (stand-in B). Both run through `build_index()` as shipped, with
its 50% DocPruner default, over the lesson's 10-page `CORPUS`, on 14
labelled queries (the 4 in `main()` plus 10 more, one gold page each). The
lesson's `hash_embed` seeds from the builtin `hash()`, which Python salts per
process, so a "model" is really a distribution over salts; each is measured
over 50 salts, with `hash` pinned to a salted CRC32 so the run repeats. The
content-class tag is set on every `Page`; the router reads the tag of model
A's top hit and answers with whichever model won that class on salts 0-24,
and it is scored on salts 25-49.

**ANSWER: B gets more right on 12 of 14 queries; A wins 2 (the lesson's
EMEA demo query, 22 vs 15 of 50, and "late interaction retrieval vs OCR",
33 vs 29).** The largest B-over-A gap is the handwritten "pH readings in the
lab notes" (24 vs 44 of 50). Hit@1 over salts 25-49: A 0.740, B 0.811, and
the class router 0.783. It learns "table -> A" from the EMEA query, but the
tag it reads is A's top hit: 27 of the 79 held-out queries it routes as
"table" are chart, handwriting or text queries, and they go to the weaker
model.

**FINDING: which pages one model misses is mostly process noise.** Per run
A and B differ on whether 3.80 of 14 queries are right; A differs from
itself under a different salt on 4.36, B on 2.96. Running the lesson's
`main.py` under PYTHONHASHSEED 0-9 gives its first demo query ("what was the
2024 operating margin change for EMEA") the 10-K p.88 top hit in 5 of 10
processes (the exact count depends on CPython's string hash, so the check
asserts only that it is neither 0 nor 10).

**FINDING: the demo's headline page does not contain its query's year.**
p.88 shares 3 query tokens (operating, margin, emea) and chart p.12 shares 3
too (operating, margin, 2024), so which one ranks first is decided by which
patches the 50% prune happens to drop.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zlib

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "04-multimodal-document-qa"
SALTS, FIT = 50, 25
QUERIES = [  # (query, gold doc, gold page); the first four are main()'s own
    ("what was the 2024 operating margin change for EMEA", "10k-2024", 88),
    ("late interaction retrieval vs OCR", "paper-vidore-v3", 3),
    ("handwritten experimental figures with error bars", "handwritten-lab", 6),
    ("bar chart comparing segment margins", "chart-report", 12),
    ("segment operating margin table", "10k-2024", 88), ("FX impact and macro headwinds in EMEA", "10k-2024", 92),
    ("consolidated revenue growth executive summary", "10k-2024", 14),
    ("nDCG results vision first vs OCR", "paper-vidore-v3", 7),
    ("M3DocVQA multi page evaluation protocol", "paper-m3docrag", 2),
    ("pH readings in the lab notes", "handwritten-lab", 5), ("circuit board experiment notes", "handwritten-lab", 5),
    ("revenue by segment line chart", "chart-report", 11), ("quarterly revenue Q1 to Q4 APAC", "chart-report", 11),
    ("bar chart of operating margin 2023 vs 2024", "chart-report", 12),
]
CLASS = {("10k-2024", 88): "table", ("paper-vidore-v3", 7): "table", ("chart-report", 11): "chart",
         ("chart-report", 12): "chart", ("handwritten-lab", 5): "handwriting", ("handwritten-lab", 6): "handwriting"}
GOLD_CLASS = [CLASS.get((d, p), "text") for _, d, p in QUERIES]


def run_model(ref, dim, salt):
    """Per query: (top-1 correct, content-class tag of the top-1 page), for one embedder size and salt."""
    ref.EMB_DIM, ref.hash = dim, lambda tok: zlib.crc32(f"{salt}|{tok}".encode())
    idx = ref.build_index(prune=True)
    for page in idx.pages:  # the content-class tag the exercise asks for
        page.content_class = CLASS.get((page.doc_id, page.page_num), "text")
    tops = [idx.retrieve(q, 1)[0][0] for q, _, _ in QUERIES]
    return [((p.doc_id, p.page_num) == (d, g), p.content_class) for p, (_, d, g) in zip(tops, QUERIES)]


def fit_router(runs_a, runs_b):
    """Per gold class, the model with more top-1 hits on the fitting salts."""
    wins = {c: [0, 0] for c in GOLD_CLASS}
    for s in range(FIT):
        for c, (a, _), (b, _) in zip(GOLD_CLASS, runs_a[s], runs_b[s]):
            wins[c][0], wins[c][1] = wins[c][0] + a, wins[c][1] + b
    return {c: "A" if a > b else "B" for c, (a, b) in wins.items()}


def held_out(router, runs_a, runs_b):
    rows = [(a, b, tag, c) for s in range(FIT, SALTS) for (a, tag), (b, _), c in zip(runs_a[s], runs_b[s], GOLD_CLASS)]
    ok_a, ok_b, tags, classes = zip(*rows)
    routed = sum((a, b)[router[tag] == "B"] for a, b, tag, _ in rows)
    misrouted = sum(t == "table" != c for t, c in zip(tags, classes))
    n = len(rows)
    return {"held_a": round(sum(ok_a) / n, 3), "held_b": round(sum(ok_b) / n, 3), "held_routed": round(routed / n, 3),
            "table_routed": tags.count("table"), "misrouted": misrouted}


def disagreement(x, y, shift=0):
    """Mean number of queries two run sets judge differently (right vs wrong), salt s against s + shift."""
    return sum(u[0] != v[0] for s in range(SALTS) for u, v in zip(x[s], y[(s + shift) % SALTS])) / SALTS


def per_query(runs_a, runs_b):
    counts = [(sum(r[i][0] for r in runs_a), sum(r[i][0] for r in runs_b)) for i in range(len(QUERIES))]
    return {"per_query": counts, "b_wins": sum(b > a for a, b in counts),
            "a_wins": [i for i, (a, b) in enumerate(counts) if a > b]}


def shipped_top_hits(seeds=range(10)):
    """The lesson's own main.py under real process salts: how often its first demo query tops p.88."""
    script = str(parity.lesson_dir(PHASE, LESSON) / "code" / "main.py")
    runs = [subprocess.run([sys.executable, script], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": str(seed)}).stdout for seed in seeds]
    return [out.split("Q: ")[1].splitlines()[1].split()[-2:] for out in runs].count(["10k-2024", "p.88"])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs_a, runs_b = ([run_model(ref, dim, s) for s in range(SALTS)] for dim in (16, 128))
    router, query = fit_router(runs_a, runs_b), set(ref.tokenize(QUERIES[0][0]))
    return {
        **per_query(runs_a, runs_b), "router": router, **held_out(router, runs_a, runs_b),
        "d_ab": disagreement(runs_a, runs_b), "d_aa": disagreement(runs_a, runs_a, 1),
        "d_bb": disagreement(runs_b, runs_b, 1), "shipped_p88": shipped_top_hits(),
        "shared": [sorted(query & set(ref.tokenize(ref.CORPUS[i][2]))) for i in (0, 9)],
    }


def verify(r):
    return [
        practice.Check(
            "ANSWER: B wins 12 of 14 queries, A wins the EMEA demo query and one other; routing by class ~ B",
            (r["b_wins"], r["a_wins"], [r["per_query"][i] for i in (0, 1, 9)])
            == (12, [0, 1], [(22, 15), (33, 29), (24, 44)])
            and (r["held_a"], r["held_b"], r["held_routed"], r["misrouted"], r["table_routed"])
            == (0.74, 0.811, 0.783, 27, 79)
            and r["router"] == {"table": "A", "text": "B", "chart": "B", "handwriting": "B"},
            f"per-query hits of 50 (A, B): {r['per_query']}; router {r['router']}; held-out hit@1 A {r['held_a']}, "
            f"B {r['held_b']}, routed {r['held_routed']}; {r['misrouted']}/{r['table_routed']} table-routed misfiled",
        ),
        practice.Check(
            "FINDING: model-vs-model disagreement is smaller than one model against itself across salts",
            (r["d_ab"], r["d_aa"], r["d_bb"]) == (3.8, 4.36, 2.96) and 0 < r["shipped_p88"] < 10,
            f"queries judged differently per run: A-B {r['d_ab']}, A-A {r['d_aa']}, B-B {r['d_bb']}; "
            f"main.py under PYTHONHASHSEED 0-9 puts p.88 first for its EMEA query in {r['shipped_p88']}/10",
        ),
        practice.Check(
            "FINDING: the demo's headline page does not contain its query's year and ties chart p.12",
            r["shared"] == [["emea", "margin", "operating"], ["2024", "margin", "operating"]],
            f"query tokens shared with p.88 and p.12: {r['shared']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
