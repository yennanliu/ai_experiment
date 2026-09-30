"""Exercise 4 — the answer head keeps the gold doc for 2 of 8 new queries and answers all 4 off-topic ones.

    Add a second cross-encoder head that predicts a binary "is this answer in the document" label. Use both heads at inference; one to rank, one to threshold.

Reading of the exercise: the lesson's `CrossEncoder` keeps its rank head. A
second `nn.Linear` answer head reads the same mean-pooled vector, which a
forward pre-hook on `model.head` hands over. Both heads train together for
`main()`'s 60 full-batch Adam steps at lr 5e-3. The loss is `train_tiny`'s
MSE on the graded label plus BCE on "answer in document". That binary label
is 1 for the five label-1.0 triples. The 0.2-0.5 triples are topical
neighbours without the answer, so they get 0. At inference every corpus doc
is ranked by head 1 and kept only if sigmoid(head 2) > 0.5. It is tested on
the 5 training queries, 8 held-out paraphrases (one per doc) and 4 off-topic
queries, which should come back empty.

**ANSWER: the two-head reranker works only on the queries it was trained
on.**

| queries | gold kept | gold at rank 1 | wrong docs kept | empty answers |
|---|---:|---:|---:|---:|
| 5 training | 5/5 | 2/5 | 6/35 | 0 |
| 8 held-out | 2/8 | 2/8 | 19/56 | 1 |
| 4 off-topic | -- | -- | 9/32 | 0/4 |

"banana bread recipe" gets d3 (retry budgets) as an answer. The threshold
never says "not here" for the queries that need it. The shared loss also
costs the rank head: training-query top-1 is 2/5, against 3/5 for the
single-head model over the same 8 candidates (exercise 1).

**FINDING: the doc's single-head alternative cannot separate off-topic
queries either.** The doc says to log the rank-1 score and treat a low one as
out of domain. With the lesson's own model, the off-topic top-1 scores are
0.679, 0.780, 1.270 and 1.146. The held-out ones run from 0.695 to 1.167.
"who won the world cup" outscores all 8 real queries, so no threshold
rejects every off-topic query without also rejecting all 8 in-domain ones.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
    from torch import nn
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
OFF_TOPIC = ["banana bread recipe", "fluffy clouds drift across a summer afternoon sky",
             "who won the world cup", "how tall is mount everest"]


def two_heads(ref):
    """The lesson's CrossEncoder plus an answer head reading the same pooled vector as its rank head."""
    model = ref.CrossEncoder()  # seeds itself; the answer head is drawn right after
    answer, pooled = nn.Linear(model.head.in_features, 1), []
    model.head.register_forward_pre_hook(lambda _m, args: pooled.append(args[0]))

    def both(ids, tids):
        pooled.clear()
        rank = model(ids, tids)
        return rank, answer(pooled[0]).squeeze(-1)

    return model, answer, both


def train(ref, epochs=60, lr=5e-3):
    """train_tiny's loop (full batch, Adam, MSE on the rank head) plus BCE on answer = (label == 1.0)."""
    model, answer, both = two_heads(ref)
    opt = torch.optim.Adam([*model.parameters(), *answer.parameters()], lr=lr)
    ids, tids, labels = ref._batch_encode(ref.TRAIN_TRIPLES)
    for _ in range(epochs):
        opt.zero_grad()
        rank, logit = both(ids, tids)
        loss = nn.functional.mse_loss(rank, labels) + nn.functional.binary_cross_entropy_with_logits(logit, (labels == 1.0).float())
        loss.backward()
        opt.step()
    return both


@torch.no_grad()
def answer_set(ref, both, query, threshold=0.5):
    """Rank every corpus doc with head 1, keep those head 2 says contain the answer."""
    ids, tids, _ = ref._batch_encode([ref.Triple(query, c.text, 0.0) for c in ref.CORPUS])
    rank, logit = both(ids, tids)
    order = sorted(range(len(ref.CORPUS)), key=lambda i: -rank[i].item())
    return [ref.CORPUS[i].doc_id for i in order if torch.sigmoid(logit[i]) > threshold], ref.CORPUS[order[0]].doc_id


def summarise(ref, both, queries):
    tally = {"gold_kept": 0, "top1": 0, "wrong_kept": 0, "empty": 0}
    for q, g in queries.items():
        kept, top1 = answer_set(ref, both, q)
        row = {"gold_kept": g in kept, "top1": top1 == g, "wrong_kept": len(kept) - (g in kept), "empty": not kept}
        tally = {k: tally[k] + int(v) for k, v in row.items()}
    return tally


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    both = train(ref)
    by_text = {c.text: c.doc_id for c in ref.CORPUS}
    train_q = {t.query: by_text[t.document] for t in ref.TRAIN_TRIPLES if t.label == 1.0}
    out = {"train": summarise(ref, both, train_q), "held": summarise(ref, both, HELD_OUT)}
    out["off_topic"] = {q: answer_set(ref, both, q)[0] for q in OFF_TOPIC}
    single = ref.CrossEncoder()
    ref.train_tiny(single, ref.TRAIN_TRIPLES, epochs=60)
    out["single_top1"] = {q: round(ref.rerank(single, q, ref.CORPUS, 1)[0][1], 3) for q in [*OFF_TOPIC, *HELD_OUT]}
    out["doc_says"] = "Log the rank-1 cross-encoder score" in parity.doc_text(PHASE, LESSON, "en")
    return out


def verify(result):
    r = result
    kept = [len(v) for v in r["off_topic"].values()]
    held = [r["single_top1"][q] for q in HELD_OUT]
    off = [r["single_top1"][q] for q in OFF_TOPIC]
    return [
        practice.Check(
            "ANSWER: the two-head reranker works only on the queries it was trained on",
            (r["train"], r["held"], kept, r["off_topic"]["banana bread recipe"])
            == ({"gold_kept": 5, "top1": 2, "wrong_kept": 6, "empty": 0}, {"gold_kept": 2, "top1": 2, "wrong_kept": 19, "empty": 1},
                [1, 4, 3, 1], ["d3"]),
            f"train {r['train']}; held-out {r['held']}; off-topic answers {r['off_topic']}",
        ),
        practice.Check(
            "FINDING: the doc's single-head alternative cannot separate off-topic queries either",
            (r["doc_says"], off, min(held), max(held), max(off) > max(held)) == (True, [0.679, 0.78, 1.27, 1.146], 0.695, 1.167, True),
            f"rank-1 score, off-topic {off}; held-out {min(held)}..{max(held)}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
