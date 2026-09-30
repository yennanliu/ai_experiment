"""Exercise 2 — after ten epochs the margin is 0.113, a negative still outscores a positive, and held-out stays below zero.

    Train the cross-encoder for ten epochs instead of one. Measure the score-margin between positive and negative pairs at each epoch.

Reading of the exercise: an "epoch" is one call of the loop in the lesson's
`train_tiny`, which is one full-batch Adam step over the 14 `TRAIN_TRIPLES`.
For each epoch count from 0 to 10, and for `main()`'s 60, a fresh
`CrossEncoder` (it seeds itself) is trained with `train_tiny(epochs=k)`. The
margin is measured three ways. Mean margin is the mean score of the
label-1.0 pairs minus the mean of the label-0.0 pairs; the graded 0.2-0.5
pairs are left out. Worst gap is the lowest positive minus the highest
negative, and it is above 0 only when every positive outranks every
negative. Held-out margin is the gold document's score minus the mean of the
other 7, averaged over 8 paraphrased queries the model never saw.

**ANSWER: the mean margin grows from 0.011 after one epoch to 0.113 after
ten, but the positives are still not separated from the negatives.**

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean margin | .031 | .011 | .039 | .051 | .060 | .068 | .075 | .081 | .089 | .099 | .113 | 1.002 |
| worst gap | -.164 | -.192 | -.111 | -.051 | -.019 | -.010 | -.007 | -.006 | -.006 | -.007 | -.015 | .971 |
| held-out | -.000 | -.011 | -.005 | -.001 | -.000 | -.002 | -.005 | -.008 | -.010 | -.011 | -.010 | .153 |

The first step shrinks the untrained model's margin (0.031 to 0.011), and
the first step raises the loss from 0.536 to 0.780. The worst gap first turns
positive at epoch 15. A forward hook on one 60-epoch run reads the same
pre-step values at every epoch.

**FINDING: the lesson's "one epoch" is really 60.** The doc lists
`train_tiny(pairs)` as "one pass of supervised training". The function
defaults to `epochs=60`, and `main()` passes 60. One pass leaves the worst
gap at -0.192.

**FINDING: for all ten epochs the held-out margin is at or below zero.** The
model does not prefer the gold document on queries it has not seen until
well past epoch 10 (0.153 at epoch 60). Everything that happens in the ten
epochs is memorisation of the 14 training pairs.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "66-reranker-cross-encoder"
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


@torch.no_grad()
def scores(ref, model, pairs):
    ids, tids, _ = ref._batch_encode([ref.Triple(q, d, 0.0) for q, d in pairs], max_len=model.max_len)
    return model.eval()(ids, tids)


def margins(ref, model):
    """Train: mean(label 1.0) - mean(label 0.0), and the worst pos-neg gap. Held-out: gold - mean(rest)."""
    s = scores(ref, model, [(t.query, t.document) for t in ref.TRAIN_TRIPLES])
    lab = torch.tensor([t.label for t in ref.TRAIN_TRIPLES])
    pos, neg = s[lab == 1.0], s[lab == 0.0]
    texts = {c.doc_id: c.text for c in ref.CORPUS}
    held = []
    for q, g in HELD_OUT.items():
        h = scores(ref, model, [(q, texts[d]) for d in texts])
        gold = list(texts).index(g)
        held.append(h[gold].item() - (h.sum().item() - h[gold].item()) / (len(texts) - 1))
    return round((pos.mean() - neg.mean()).item(), 4), round((pos.min() - neg.max()).item(), 4), round(sum(held) / len(held), 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows, losses = [], None
    for epochs in [*range(11), 60]:
        model = ref.CrossEncoder()  # seeds itself, so every run starts from the same weights
        losses = ref.train_tiny(model, ref.TRAIN_TRIPLES, epochs=epochs) if epochs else []
        rows.append((epochs, *margins(ref, model), round(losses[-1], 4) if losses else None))
    model, seen = ref.CrossEncoder(), []
    model.register_forward_hook(lambda _m, _i, out: seen.append(out.detach()))  # pre-step scores, every epoch
    ref.train_tiny(model, ref.TRAIN_TRIPLES, epochs=60)
    lab = torch.tensor([t.label for t in ref.TRAIN_TRIPLES])
    gaps = [(o[lab == 1.0].min() - o[lab == 0.0].max()).item() for o in seen]
    return {"rows": rows, "separated_at": next(i for i, g in enumerate(gaps) if g > 0), "gaps": [round(g, 3) for g in gaps],
            "doc": parity.doc_text(PHASE, LESSON, "en"),
            "default_epochs": inspect.signature(ref.train_tiny).parameters["epochs"].default}


MEAN = [0.0311, 0.0108, 0.0391, 0.0511, 0.0595, 0.0676, 0.075, 0.0811, 0.0885, 0.0994, 0.113, 1.002]
WORST = [-0.1642, -0.192, -0.1109, -0.051, -0.019, -0.0104, -0.0069, -0.006, -0.006, -0.0073, -0.0145, 0.9714]
HELD = [-0.0003, -0.0106, -0.0046, -0.0013, -0.0004, -0.0018, -0.0047, -0.008, -0.0104, -0.011, -0.0098, 0.1531]


def close(got, want, tol=2e-4):
    return len(got) == len(want) and all(abs(a - b) <= tol for a, b in zip(got, want))


def verify(result):
    r = result
    mean, worst, held = ([row[i] for row in r["rows"]] for i in (1, 2, 3))
    losses = [row[4] for row in r["rows"]]
    return [
        practice.Check(
            "ANSWER: ten epochs take the mean margin from 0.011 to 0.113, with positives still not separated",
            all([close(mean, MEAN), close(worst, WORST), r["separated_at"] == 15, close(r["gaps"][:11], WORST[:11], 6e-4)]),
            f"mean {mean}; worst gap {worst} (epochs 0-10, 60); first separated at epoch {r['separated_at']}",
        ),
        practice.Check(
            "FINDING: the lesson's 'one epoch' is really 60",
            all(["one pass of supervised training" in r["doc"], r["default_epochs"] == 60, close(losses[1:3], [0.5356, 0.7795])]),
            f"doc says 'one pass'; train_tiny defaults to epochs={r['default_epochs']}; loss epoch 1-2 {losses[1:3]}",
        ),
        practice.Check(
            "FINDING: for all ten epochs the held-out margin is at or below zero",
            all([close(held, HELD), max(held[:11]) <= 0 < held[-1]]),
            f"held-out margin epochs 0-10 {held[:11]}, epoch 60 {held[-1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
