"""Exercise 3 -- a real cross-encoder judge disagrees with the mock on 1 of 45 calls and accepts the same 7 of 8 wrong facts.

    Replace the mock judge with a real model call. Measure the disagreement between the mock and the real judge on the fixture.

Reading of the exercise: the "real model" is local, so the run stays
offline: `cross-encoder/ms-marco-MiniLM-L-6-v2` from the Hugging Face cache,
a trained (query, passage) relevance model, run by a from-scratch BERT
forward (float64) with uncased WordPiece over the checkpoint's own
`vocab.txt` (the fixture is ASCII, so no accent or CJK handling).
transformers is not an installed extra; while authoring (2026-09-29) this
forward was checked against `AutoModelForSequenceClassification`
(transformers 5.17.0) on all 53 pairs scored here: identical token ids,
logits within 3.8e-6. The judge says yes when the logit is above 0, the
boundary of its sigmoid. `supported(claim, context)` scores (claim, context),
`relevant(question, answer)` scores (question, answer). It is passed to the
lesson's `evaluate_pipeline` in place of `MockJudge`, so the lesson's
evaluator makes every call; each call is logged with both judges' verdicts.

**ANSWER: over the lesson's three pipelines the two judges disagree on 1 of
45 calls (2.2%).** The call is the baseline's answer to "how do you stop a
long-running worker": the answer is the worker-pool-sizing doc (d10), the
mock calls it relevant on shared words, and the real model gives -3.799.
Baseline answer relevance falls from 0.75 to 0.5; every other metric value
is unchanged (faithfulness 1.0 for all three, relevance 1.0 for hybrid and
hybrid+rerank). The judges agree on all 33 faithfulness calls and 11 of the
12 relevance calls.

**FINDING: the 33 faithfulness calls could not have disagreed.** Every claim
is a sentence pasted from the retrieved context (Exercise 2), so both judges
say yes, the real one with logits from 8.2 to 10.3.

**FINDING: the real judge is no better at faithfulness.** On 8 claims with
one fact changed (five failed parts, `k = 10`, "ignore the cancellation
signal"), it accepts the same 7 the mock accepts. A relevance model scores
topic match, not entailment.
"""

import math
import pathlib
import re

from harness import parity, practice

try:
    import torch
    from safetensors.torch import load_file
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch and safetensors: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "68-rag-eval-precision-recall"
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HUB = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
PIPES = ("baseline", "hybrid", "hybrid_plus_rerank")
EDITED = [  # one fact changed in a verbatim hybrid claim, judged against that qrel's context
    ("q1", "The abort threshold is configured per bucket at five failed parts."),
    ("q1", "The abort threshold is configured globally at three failed parts."),
    ("q1", "Past that threshold the upload is retried forever."),
    ("q2", "Authorization is centralized in the policy engine."), ("q2", "Authorization is scattered across every service."),
    ("q3", "Production search combines lexical and semantic retrieval through reciprocal rank fusion at k = 10."),
    ("q3", "Production search uses only semantic retrieval."), ("q4", "Long-running jobs ignore the cancellation signal."),
]


def load_model():
    """The cached checkpoint's WordPiece vocabulary and its weights in float64, offline."""
    snaps = sorted((HUB / ("models--" + MODEL.replace("/", "--")) / "snapshots").glob("*/model.safetensors"))
    if not snaps:
        raise practice.Skip(f"needs the {MODEL} checkpoint in {HUB}; download it once with huggingface-cli")
    vocab = {w: i for i, w in enumerate((snaps[0].parent / "vocab.txt").read_text().splitlines())}
    weights = {k.removeprefix("bert."): v.double() for k, v in load_file(str(snaps[0])).items() if v.is_floating_point()}
    return vocab, weights


def wordpiece(text, vocab):
    """Uncased BERT on ASCII text: split off punctuation (`_` included), then greedy longest-match."""
    ids = []
    for word in re.findall(r"[^\W_]+|[^\w\s]|_", text.lower()):
        pieces, i = [], 0
        while i < len(word):
            j = next((j for j in range(len(word), i, -1) if ("##" * (i > 0) + word[i:j]) in vocab), i)
            pieces, i = (pieces + [vocab["##" * (i > 0) + word[i:j]]], j) if j > i else ([vocab["[UNK]"]], len(word))
        ids += pieces
    return ids


def score(a, b, model):
    """BertForSequenceClassification on [CLS] a [SEP] b [SEP]; the logit, > 0 means yes."""
    vocab, w = model
    ia, ib = wordpiece(a, vocab), wordpiece(b, vocab)
    tok, typ, e = [101, *ia, 102, *ib, 102], [0] * (len(ia) + 2) + [1] * (len(ib) + 1), "embeddings."
    n = len(tok)

    def ln(x, p):
        return torch.nn.functional.layer_norm(x, (384,), w[p + ".weight"], w[p + ".bias"], 1e-12)
    def lin(x, p):
        return x @ w[p + ".weight"].T + w[p + ".bias"]

    x = w[e + "word_embeddings.weight"][tok] + w[e + "position_embeddings.weight"][:n]
    x = ln(x + w[e + "token_type_embeddings.weight"][typ], e + "LayerNorm")
    for layer in range(6):
        p = f"encoder.layer.{layer}."
        q, k, v = (lin(x, p + "attention.self." + t).view(n, 12, 32).transpose(0, 1) for t in ("query", "key", "value"))
        att = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(32), -1) @ v
        x = ln(x + lin(att.transpose(0, 1).reshape(n, 384), p + "attention.output.dense"), p + "attention.output.LayerNorm")
        x = ln(x + lin(torch.nn.functional.gelu(lin(x, p + "intermediate.dense")), p + "output.dense"), p + "output.LayerNorm")
    return lin(torch.tanh(lin(x[0], "pooler.dense")), "classifier").item()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    model, mock, calls, out = load_model(), ref.MockJudge(), [], {}

    def judged(kind):  # the MockJudge interface over the model; each call logged with both verdicts
        def call(a, b):
            s = score(a, b, model)
            calls.append((pipe, kind, a, round(s, 3), getattr(mock, kind)(a, b), s > 0))
            return s > 0
        return staticmethod(call)

    real = type("RealJudge", (), {"supported": judged("supported"), "relevant": judged("relevant")})()
    for pipe in PIPES:  # `pipe` labels the calls logged during this evaluation
        res = ref.evaluate_pipeline(getattr(ref, pipe + "_pipeline"), ref.QRELS, judge=real)
        out[pipe] = (res["faithfulness"], res["answer_relevance"])
    by_id = {d.doc_id: d for d in ref.CORPUS}
    ctx = {q.qid: " ".join(by_id[i].text() for i in ref.hybrid_pipeline(q.query, 5)[0]) for q in ref.QRELS}
    return {**out, "calls": calls, "edited": [(mock.supported(c, ctx[q]), score(c, ctx[q], model) > 0) for q, c in EDITED]}


def verify(result):
    r, calls = result, result["calls"]
    sup = [s for _, k, _, s, m, x in calls if k == "supported" and m and x]
    split = [(n, k, a, s) for n, k, a, s, m, x in calls if m != x]
    return [
        practice.Check(
            "ANSWER: the judges disagree on 1 of 45 calls, the baseline's answer to q4",
            (len(calls), split, [r[n] for n in PIPES]) == (45, [("baseline", "relevant", "how do you stop a long-running worker",
                                                                  -3.799)], [(1.0, 0.5), (1.0, 1.0), (1.0, 1.0)]),
            f"{len(calls)} calls; split verdicts {split}; (faithfulness, relevance) with the real judge {[r[n] for n in PIPES]}",
        ),
        practice.Check(
            "FINDING: the 33 faithfulness calls cannot disagree -- every claim is verbatim context",
            (len(sup), min(sup) > 8) == (33, True),
            f"both judges say yes to {len(sup)} supported calls; real logits {min(sup)} to {max(sup)}",
        ),
        practice.Check(
            "FINDING: the real model accepts the same 7 of 8 one-fact edits the mock accepts",
            r["edited"] == [(True, True)] * 4 + [(False, False)] + [(True, True)] * 3,
            f"(mock, real) on the edited claims {r['edited']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
