"""Exercise 4 -- training lifts image-to-text MRR from 0.083 to 0.104 while the lesson's R@1 falls from 0.02 to 0.

    Compute mean reciprocal rank (MRR) alongside R@K. MRR is sensitive to where the correct item lands beyond the top K; R@K is sensitive to whether it lands in the top K.

Reading of the exercise: MRR is the mean of 1 / rank of the matching item,
computed in both directions on the similarity matrices the lesson's own
`main()` builds (its `recall_at_k` is wrapped and restored to record them),
before and after training. The rank of a query is the smallest k whose
`sim.topk(k)` holds the match, the same test `recall_at_k` uses, so the R@K
recomputed from these ranks equals the lesson's exactly. Because the matrix
has exact ties, MRR is also given with ties broken in the match's favour and
against it.

**ANSWER: image-to-text MRR goes 0.083 -> 0.1036, text-to-image 0.1007 ->
0.1528.** The median rank improves from 26 to 18 (i2t) and 26 to 16 (t2i) of
50. Most queries still land outside the top 10: 42 -> 37 (i2t) and 40 -> 31
(t2i).

**FINDING: R@1 and MRR disagree on the i2t direction, and the one R@1 hit is a
tie.** Training moves R@1_i2t from 0.02 to 0.0 while MRR rises by a quarter. The
0.02 is image 1, whose caption appears twice in the suite. Its top two scores
are both 0.149, and `topk(1)` happened to return the diagonal. With the tie
broken against it, R@1_i2t is 0.0 before training too.

**FINDING: the retrieval suite has 11 duplicated captions, so a perfect model
scores R@1 = 0.78.** Only 39 of the 50 captions are distinct: captions are a
function of the sample index alone. An oracle similarity (1 where two
captions are equal) scores R@1 0.78 in both directions under the lesson's
`recall_at_k`, and MRR 0.89 (1.0 at best, 0.78 at worst). All 50 eval
captions also appear verbatim in the 200-pair training corpus, although the
lesson calls the suite "held out from the training corpus".

Structure: `run_lesson()` records the matrices; `ranks()` gives per-query
ranks; `surface()` reports MRR, R@1/5/10, ties and a parity check.
"""


from __future__ import annotations

import contextlib
import io

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "63-multimodal-eval"


def run_lesson(ref):
    """The lesson's main() as shipped, recording each similarity matrix and the eval suite."""
    seen, saved_rk, saved_ev = {"sims": [], "suite": None}, ref.recall_at_k, ref.evaluate

    def recall(sim, k):
        if k == 1:
            seen["sims"].append(sim.clone())
        return saved_rk(sim, k)

    def evaluate(model, suite):
        seen["suite"] = suite
        return saved_ev(model, suite)

    ref.recall_at_k, ref.evaluate = recall, evaluate
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ref.main()
    finally:
        ref.recall_at_k, ref.evaluate = saved_rk, saved_ev
    return seen["sims"], seen["suite"]


def ranks(sim):
    """1-based rank of the diagonal per row: the smallest k whose sim.topk(k) holds it, exactly
    as recall_at_k decides a hit; then the best and the worst case over tied scores."""
    n, target = sim.shape[0], torch.arange(sim.shape[0]).unsqueeze(1)
    hits = torch.stack([(sim.topk(k, dim=1).indices == target).any(1) for k in range(1, n + 1)], 1)
    topk = hits.float().argmax(1) + 1
    diag = sim.diag().unsqueeze(1)
    return topk, 1 + (sim > diag).sum(1), (sim >= diag).sum(1)


def mrr(rank):
    return round((1.0 / rank.double()).mean().item(), 4)


def surface(ref, sim):
    """MRR and R@1/5/10 for both directions, plus a parity check against recall_at_k."""
    out = {}
    for name, m in (("i2t", sim), ("t2i", sim.T)):
        topk, best, worst = ranks(m)
        out[name] = {"mrr": mrr(topk), "mrr_best": mrr(best), "mrr_worst": mrr(worst),
                     "R": [round((topk <= k).double().mean().item(), 2) for k in (1, 5, 10)],
                     "ties": int((best != worst).sum()), "beyond10": int((topk > 10).sum()),
                     "median": int(topk.median()), "R1_worst": round((worst <= 1).double().mean().item(), 2)}
    lesson = [ref.recall_at_k(sim, k) for k in (1, 5, 10)]
    out["parity"] = all(abs(a - out["i2t"]["R"][i]) < 1e-6 and abs(b - out["t2i"]["R"][i]) < 1e-6
                        for i, (a, b) in enumerate(lesson))
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    (before, after), suite = run_lesson(ref)
    caps = [tuple(p.caption_ids[0].tolist()) for p in suite.retrieval]
    oracle = torch.tensor([[float(a == b) for b in caps] for a in caps])
    train = {tuple(c[0].tolist()) for _, c in ref.make_mock_corpus(1, 200, 128, 10)}
    return {"before": surface(ref, before), "after": surface(ref, after),
            "oracle": surface(ref, oracle), "distinct": len(set(caps)),
            "in_training": sum(c in train for c in caps),
            "tie_hit": [round(v, 4) for v in before[1].sort(descending=True).values[:2].tolist()]
            + [round(before[1, 1].item(), 4)]}


def headline(s):
    """(MRR, median rank, queries beyond rank 10), each as (i2t, t2i)."""
    return tuple(s[d][k] for k in ("mrr", "median", "beyond10") for d in ("i2t", "t2i"))


def verify(result):
    r, b, a, o = result, result["before"], result["after"], result["oracle"]
    return [
        practice.Check(
            "ANSWER: MRR i2t 0.083 -> 0.1036, t2i 0.1007 -> 0.1528",
            (b["parity"], a["parity"], o["parity"], headline(b), headline(a))
            == (True, True, True, (0.083, 0.1007, 26, 26, 42, 40), (0.1036, 0.1528, 18, 16, 37, 31)),
            f"(MRR, median, beyond 10) x (i2t, t2i): before {headline(b)}, after {headline(a)}; "
            f"R@1/5/10 before {b['i2t']['R']}/{b['t2i']['R']}, after {a['i2t']['R']}/{a['t2i']['R']}",
        ),
        practice.Check(
            "FINDING: R@1 and MRR disagree on i2t, and the one R@1 hit is a tie",
            (b["i2t"]["R"][0], a["i2t"]["R"][0], r["tie_hit"], b["i2t"]["R1_worst"], b["i2t"]["mrr_worst"])
            == (0.02, 0.0, [0.149, 0.149, 0.149], 0.0, 0.0696),
            f"R@1_i2t {b['i2t']['R'][0]} -> {a['i2t']['R'][0]}; image 1's top two scores and its "
            f"diagonal {r['tie_hit']}; tie broken against: R@1 {b['i2t']['R1_worst']}, MRR {b['i2t']['mrr_worst']}",
        ),
        practice.Check(
            "FINDING: 11 duplicated captions cap a perfect model at R@1 = 0.78",
            (r["distinct"], o["i2t"]["R"][0], o["t2i"]["R"][0], o["i2t"]["mrr"], o["i2t"]["mrr_best"],
             o["i2t"]["mrr_worst"], r["in_training"]) == (39, 0.78, 0.78, 0.89, 1.0, 0.78, 50),
            f"{r['distinct']}/50 distinct captions; oracle R@1 {o['i2t']['R'][0]}, MRR "
            f"{o['i2t']['mrr']} ({o['i2t']['mrr_best']} best, {o['i2t']['mrr_worst']} worst); "
            f"{r['in_training']}/50 eval captions are training captions",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
