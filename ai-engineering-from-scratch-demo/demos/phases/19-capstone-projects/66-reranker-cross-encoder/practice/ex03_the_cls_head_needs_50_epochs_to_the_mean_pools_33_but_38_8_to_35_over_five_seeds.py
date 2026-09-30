"""Exercise 3 — the CLS head needs 50 epochs to the mean-pool's 33 on the lesson's seed, and 38.8 to 35.0 over five seeds.

    Replace mean-pooling with a CLS-token head. Compare convergence on this fixture.

Reading of the exercise: the lesson's `CrossEncoder` is kept as it is. A
forward hook on its last LayerNorm copies the CLS position over every
position, so the lesson's own mean-pool then averages identical vectors and
its head reads the CLS vector alone. This matches `head(x[:, 0])` to 1.8e-7
(float32). Both heads are trained with `train_tiny` for `main()`'s 60 epochs
on the 14 `TRAIN_TRIPLES`. Convergence is the first epoch whose loss drops
below 0.1 and below 0.01. The run is repeated from five inits, the
lesson's `SEED` plus 0 to 4, by swapping the `_set_seed` that
`CrossEncoder()` calls. Held-out recall@1 over 8 paraphrased queries (N = 8)
shows whether faster fitting buys ranking.

**ANSWER: on the lesson's seed the CLS head converges more slowly; over five
seeds the gap is small.** Epoch at which the loss first drops below 0.01:

| init | SEED | +1 | +2 | +3 | +4 | mean |
|---|---:|---:|---:|---:|---:|---:|
| mean-pool | 33 | 32 | 31 | 44 | 35 | 35.0 |
| CLS | 50 | 36 | 31 | 34 | 43 | 38.8 |

On the lesson's seed the CLS head starts at a lower loss (0.209 against
0.536). It then sits near 0.2 for 20 epochs and reaches 0.1 at epoch 36,
where the mean-pool reaches it at 17. Across seeds the mean-pool is faster on
3 inits, the CLS head on 1, and they tie on 1. Both end below 0.003. The
doc's "the difference is small" holds on average; the one seed the lesson
ships shows the biggest gap of the five.

**FINDING: neither head ranks held-out queries better than the bi-encoder.**
Held-out recall@1 over the five inits is 8/40 for the mean-pool and 6/40 for
the CLS head. The bi-encoder alone gets 20/40 (4/8 per init). Fitting the
14 training pairs faster does not transfer to new queries.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "66-reranker-cross-encoder"
EPOCHS = 60
HELD_OUT = {  # one paraphrase per corpus doc, sharing few words with it
    "cancel a broken s3 upload": "d1",
    "big file gets split into pieces": "d2",
    "limit on retries for a bucket": "d3",
    "who is allowed to do what": "d4",
    "opa runtime wrapper": "d5",
    "fusing keyword and vector results": "d6",
    "memory per embedding in float32": "d7",
    "terminate a running job": "d8",
}


def cls_hook(_module, _inputs, out):
    """On the last LayerNorm: copy the CLS position (0) over every position.

    The lesson's forward then mean-pools identical vectors, so its head reads
    the CLS vector alone. Nothing of the reference forward is rewritten.
    """
    return out[:, :1].expand_as(out)


def first_below(losses, level):
    return next((i for i, v in enumerate(losses) if v < level), None)


def run(ref, head, seed):
    ref._set_seed = lambda: (torch.manual_seed(seed), ref.np.random.seed(seed))  # CrossEncoder() calls this
    model = ref.CrossEncoder()
    if head == "cls":
        model.ln2.register_forward_hook(cls_hook)
    losses = ref.train_tiny(model, ref.TRAIN_TRIPLES, epochs=EPOCHS)
    bi = ref.BiEncoder()
    for c in ref.CORPUS:
        bi.add(c)
    hits = sum(ref.rerank(model, q, bi.search(q, 8), 1)[0][0].doc_id == g for q, g in HELD_OUT.items())
    return {"first": losses[0], "last": losses[-1], "to_0.1": first_below(losses, 0.1), "to_0.01": first_below(losses, 0.01),
            "held_r1": hits,
            "bi_r1": sum(bi.search(q, 1)[0].doc_id == g for q, g in HELD_OUT.items())}


@torch.no_grad()
def cls_is_exact(ref):
    """The hooked model's score against the head applied by hand to the raw CLS vector."""
    model, raw = ref.CrossEncoder(), []
    model.ln2.register_forward_hook(lambda _m, _i, out: raw.append(out))
    model.ln2.register_forward_hook(cls_hook)
    ids, tids, _ = ref._batch_encode(ref.TRAIN_TRIPLES)
    return (model(ids, tids) - model.head(raw[0][:, 0]).squeeze(-1)).abs().max().item()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    saved, out = ref._set_seed, {"cls_gap": cls_is_exact(ref)}
    try:
        out.update({h: [run(ref, h, ref.SEED + k) for k in range(5)] for h in ("mean", "cls")})
    finally:
        ref._set_seed = saved
    return out


def verify(result):
    r = result
    col = {h: {k: [run[k] for run in r[h]] for k in ("to_0.1", "to_0.01", "held_r1", "bi_r1", "first", "last")} for h in ("mean", "cls")}
    m, c = col["mean"], col["cls"]
    wins = [(a < b) - (a > b) for a, b in zip(m["to_0.01"], c["to_0.01"])]
    return [
        practice.Check(
            "ANSWER: the CLS head converges more slowly on the lesson's seed, a little more slowly on average",
            all([
                (m["to_0.01"], c["to_0.01"], m["to_0.1"][0], c["to_0.1"][0]) == ([33, 32, 31, 44, 35], [50, 36, 31, 34, 43], 17, 36),
                sorted(wins) == [-1, 0, 1, 1, 1], max(m["last"] + c["last"]) < 0.003, r["cls_gap"] < 1e-5,
                (round(m["first"][0], 3), round(c["first"][0], 3)) == (0.536, 0.209),
            ]),
            f"loss < 0.01 at epoch: mean-pool {m['to_0.01']} (avg {sum(m['to_0.01']) / 5}), CLS {c['to_0.01']} "
            f"(avg {sum(c['to_0.01']) / 5}); < 0.1 on SEED at {m['to_0.1'][0]} vs {c['to_0.1'][0]}; hook exact to {r['cls_gap']:.1e}",
        ),
        practice.Check(
            "FINDING: neither head ranks held-out queries better than the bi-encoder",
            (m["held_r1"], c["held_r1"], sum(m["bi_r1"])) == ([3, 1, 1, 2, 1], [1, 2, 1, 1, 1], 20),
            f"held-out recall@1 per init: mean-pool {m['held_r1']} ({sum(m['held_r1'])}/40), "
            f"CLS {c['held_r1']} ({sum(c['held_r1'])}/40); bi-encoder alone {sum(m['bi_r1'])}/40",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
