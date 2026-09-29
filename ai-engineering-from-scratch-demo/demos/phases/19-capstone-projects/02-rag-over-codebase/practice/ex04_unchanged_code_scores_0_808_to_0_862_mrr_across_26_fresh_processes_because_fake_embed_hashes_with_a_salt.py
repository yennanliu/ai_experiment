"""Exercise 4 -- unchanged code scores 0.808 to 0.862 MRR@10 across 26 fresh processes, because `fake_embed` hashes with a per-process salt.

    Add a sampling-based drift check: weekly, rerun the 100-question eval. Alert on MRR@10 drop > 5%.

Reading of the exercise: the 100-question eval is sampled from the lesson's
6-chunk `SAMPLE_CORPUS` with a fixed seed. Each question takes two words from
its gold chunk's summary and body plus one word from a different chunk, so it
is not a copy of the summary. `drift_alert` compares one week's MRR@10 with
the week before and alerts on a relative drop above 5%. The absolute reading
(0.05 points) is measured beside it. A "week" is what a scheduler actually
does: a fresh Python process that imports the lesson, builds the index, and
runs `answer` on all 100 questions. Here that is 26 subprocesses with
PYTHONHASHSEED = 0..25 and no code change between them. Two controls use a
fixed keyed hash: a real regression (the summarizer goes down, so every
chunk's summary is empty) and 1,000 bootstrap resamples of one week's answers.

**ANSWER: the check catches the regression and stays quiet on 26 unchanged
weeks.** Losing the summaries drops MRR@10 from 0.838 to 0.707 (-15.6%), and
the alert fires. Across the 26 unchanged weeks no week-over-week drop exceeds
5%.

**FINDING: the "unchanged" system is a different model every week.**
`fake_embed` uses Python's builtin `hash`, which is salted per process. So the
same code and corpus score from 0.808 to 0.862 across the 26 weeks, a spread
of 6.2% of the best week. That is larger than the alert threshold. Week over
week it stays under 5%, but a baseline frozen on a lucky week alerts on some
later week with no change at all: 4 of the 325 (baseline week,
later week) pairs alert. The
TypeScript port uses FNV-1a and does not have this problem.

**FINDING: with 100 questions, sampling noise alone crosses the line.**
Resampling one week's 100 answers 1,000 times alerts on 38 resamples under the
relative reading and on 18 under the absolute one.

Structure: `questions` samples the eval; `week_mrr` is one weekly run (it also
runs as `--week` in a subprocess); `drift_alert` is the check.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "02-rag-over-codebase"
WEEKS, THRESHOLD = 26, 0.05


def questions(ref):
    rng, corpus, out = random.Random(4), ref.SAMPLE_CORPUS, []
    for i in range(100):
        gold = i % len(corpus)
        other = rng.choice([j for j in range(len(corpus)) if j != gold])
        words = rng.sample(ref.tokenize(f"{corpus[gold].summary} {corpus[gold].body}"), 2)
        words += rng.sample(ref.tokenize(f"{corpus[other].summary} {corpus[other].body}"), 1)
        out.append((" ".join(words), gold))
    return out


def reciprocal_ranks(ref, chunks):
    dense, bm25 = ref.DenseIndex(), ref.BM25Index()
    for c in chunks:
        dense.add(c)
        bm25.add(c)
    ranks = []
    for q, gold in questions(ref):
        top, want = ref.answer(q, dense, bm25)["rerank_top"], ref.SAMPLE_CORPUS[gold].anchor()
        ranks.append(1 / (top.index(want) + 1) if want in top else 0.0)
    return ranks


def week_mrr():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return sum(reciprocal_ranks(ref, ref.SAMPLE_CORPUS)) / 100


def drift_alert(last, now, relative=True):
    drop = last - now
    return (drop / last if relative else drop) > THRESHOLD


def weekly_runs():
    runs = []
    for week in range(WEEKS):
        env = {**os.environ, "PYTHONHASHSEED": str(week)}
        done = subprocess.run([sys.executable, __file__, "--week"], env=env, capture_output=True, text=True, check=True)
        runs.append(round(json.loads(done.stdout), 4))
    return runs


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.hash = lambda s: int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
    base = reciprocal_ranks(ref, ref.SAMPLE_CORPUS)
    lost = [ref.Chunk(c.repo, c.path, c.start_line, c.end_line, c.symbol, c.body, "") for c in ref.SAMPLE_CORPUS]
    mrr, broken = sum(base) / 100, sum(reciprocal_ranks(ref, lost)) / 100
    rng, noise = random.Random(0), {"relative": 0, "absolute": 0}
    for _ in range(1000):
        resample = sum(rng.choice(base) for _ in range(100)) / 100
        noise["relative"] += drift_alert(mrr, resample)
        noise["absolute"] += drift_alert(mrr, resample, relative=False)
    runs = weekly_runs()
    return {
        "mrr": round(mrr, 3), "broken": round(broken, 3), "regression_alert": drift_alert(mrr, broken),
        "weeks": runs, "weekly_alerts": sum(drift_alert(a, b) for a, b in zip(runs, runs[1:])),
        "frozen_baseline_alerts": sum(drift_alert(a, b) for i, a in enumerate(runs) for b in runs[i + 1:]),
        "noise": noise,
    }


def verify(result):
    r, w = result, result["weeks"]
    spread = (max(w) - min(w)) / max(w)
    return [
        practice.Check(
            "ANSWER: the check fires on a -15.6% regression and stays quiet week over week on unchanged code",
            (r["mrr"], r["broken"], r["regression_alert"], r["weekly_alerts"]) == (0.838, 0.707, True, 0),
            f"MRR@10 {r['mrr']} -> {r['broken']} without summaries ({(r['broken'] - r['mrr']) / r['mrr']:+.1%}), "
            f"alert {r['regression_alert']}; week-over-week alerts on unchanged code: {r['weekly_alerts']}/{WEEKS - 1}",
        ),
        practice.Check(
            "FINDING: the 'unchanged' system is a different model every week (per-process hash salt)",
            len(set(w)) > 10 and spread > THRESHOLD and r["frozen_baseline_alerts"] > 0,
            f"{len(set(w))} distinct MRR@10 values over {WEEKS} weeks, {min(w)}-{max(w)} ({spread:.1%} of the best); "
            f"{r['frozen_baseline_alerts']} (baseline week, later week) pairs alert",
        ),
        practice.Check(
            "FINDING: with 100 questions, sampling noise alone crosses the line",
            r["noise"] == {"relative": 38, "absolute": 18},
            f"alerts on 1,000 resamples of one unchanged week: {r['noise']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if "--week" in sys.argv:
        print(json.dumps(week_mrr()))
        raise SystemExit(0)
    raise SystemExit(practice.selfcheck(globals()))
