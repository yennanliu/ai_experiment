"""Exercise 1 -- a real encoder moves the upload doc from dense rank 2 to 1, but BM25 already had it first.

    Replace `mock_embed` with a real model from your provider. Re-run the demo and report how the dense-only ranking changes on the paraphrased query.

Reading of the exercise: the "provider" is a local one, so the run stays
offline: `paraphrase-multilingual-MiniLM-L12-v2` from the Hugging Face cache,
run by a from-scratch BertModel forward (float64) and its Unigram tokenizer
over the checkpoint's own `tokenizer.json` and `model.safetensors`, then mean
pooling as its `1_Pooling/config.json` specifies. transformers is not an
installed extra; while authoring (2026-09-29) this forward was checked against
sentence-transformers on the 7 corpus texts and 3 demo queries: identical
tokens, unit-normalised embeddings within 1.4e-7. The lesson's `mock_embed` is swapped on the
loaded module, so `DenseIndex` and `HybridRetriever` run unchanged, and the
lesson's `main()` queries are re-run at its k_each = k_out = 5.

**ANSWER: on "how do we handle cancelled uploads" the dense-only ranking goes
from d3, d2, d6, d1, d4 (mock) to d2, d1, d3, d5, d4 (MiniLM).** The upload
doc d2 moves from rank 2 to rank 1 with cosine 0.6247 against 0.4517 for the
runner-up, where the mock had d3 ahead of it by 0.4047 to 0.4016. The abort
function d1 moves from 4 to 2 and the off-topic "Search ranking" d6 drops out.
The fused top-1 stays d2, but under the mock d2 and d3 tie exactly
(1/61 + 1/62 each) and d2 wins only on insertion order; MiniLM breaks the tie.

**FINDING: the "paraphrased" query is not a paraphrase of its target.** It
shares `cancelled`, `do` and `uploads` with d2's body ("Cancelled uploads do
not block..."), so BM25 ranks d2 first with 4.85 against 1.69, before any
embedding is involved. The Problem section even says the answer to this query
is "the abort function whose summary mentions cancellation", but d1 contains
no `cancel*` token, and BM25 never returns it.

**FINDING: none of the doc's rank claims matches the run.** It says the three
target docs land at (BM25, dense, RRF) ranks (1, 4, 1), (6, 1, 1) and
(3, 3, 1). With the shipped mock they land at (1, 1, 1), (1, 2, 1) and
(1, 1, 1); a rank of 6 cannot occur at k_each = 5, and BM25 returns only 2
hits for the paraphrase.

Structure: `load_model()` reads the cache (a Skip names the remedy if it is
absent); `pieces()` is Unigram Viterbi; `encode()` is the BERT forward;
`demo()` is the lesson's `main()` without the printing.
"""

from __future__ import annotations

import json
import math
import pathlib

from harness import parity, practice

try:
    import torch
    from safetensors.torch import load_file
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch and safetensors: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "65-hybrid-retrieval-bm25-dense"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HUB = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
PARAPHRASE = "how do we handle cancelled uploads"
QUERIES = {"AbortMultipartOnFail": "d1", PARAPHRASE: "d2", "centralized authorization for service accounts": "d4"}


def load_model():
    """The cached checkpoint's Unigram vocabulary and its BERT weights, offline."""
    if not (snaps := sorted((HUB / ("models--" + MODEL.replace("/", "--"))).glob("snapshots/*/model.safetensors"))):
        raise practice.Skip(f"needs the {MODEL} checkpoint in {HUB}; download it once with huggingface-cli")
    vocab = json.loads((snaps[0].parent / "tokenizer.json").read_text())["model"]["vocab"]
    weights = {k: v if k.startswith("embeddings.word") else v.double() for k, v in load_file(str(snaps[0])).items()}
    return {piece: (i, score) for i, (piece, score) in enumerate(vocab)}, weights, min(s for _, s in vocab) - 10


def pieces(word, vocab, unk):
    """Unigram Viterbi: the highest-scoring split of one metaspace-prefixed word."""
    best = [(0.0, [])] + [(-math.inf, [])] * len(word)
    for i in range(1, len(word) + 1):
        for j in range(max(0, i - 24), i):
            if (hit := vocab.get(word[j:i])) or i - j == 1:
                best[i] = max(best[i], (best[j][0] + (hit[1] if hit else unk), best[j][1] + [word[j:i]]))
    return best[-1][1]


def encode(text, model):
    """BertModel forward in float64, then the checkpoint's mean pooling."""
    vocab, w, unk = model
    split = [p for word in text.split() for p in pieces("▁" + word, vocab, unk)]
    ids = [0] + [vocab[p][0] if p in vocab else 3 for p in split][:126] + [2]

    def ln(x, p):
        return torch.nn.functional.layer_norm(x, (384,), w[p + ".weight"], w[p + ".bias"], 1e-12)

    def lin(x, p):
        return x @ w[p + ".weight"].T + w[p + ".bias"]

    emb = "embeddings."
    x = w[emb + "word_embeddings.weight"][ids].double() + w[emb + "position_embeddings.weight"][: len(ids)]
    x = ln(x + w[emb + "token_type_embeddings.weight"][0], emb + "LayerNorm")
    for layer in range(12):
        p = f"encoder.layer.{layer}."
        q, k, v = (lin(x, p + "attention.self." + t).view(-1, 12, 32).transpose(0, 1) for t in ("query", "key", "value"))
        a = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(32), -1) @ v
        x = ln(x + lin(a.transpose(0, 1).reshape(-1, 384), p + "attention.output.dense"), p + "attention.output.LayerNorm")
        h = torch.nn.functional.gelu(lin(x, p + "intermediate.dense"))
        x = ln(x + lin(h, p + "output.dense"), p + "output.LayerNorm")
    return torch.nn.functional.normalize(x.mean(0), dim=0).tolist()


def demo(ref):
    """The lesson's main() queries through HybridRetriever at k_each = k_out = 5, rounded."""
    retriever = ref.HybridRetriever()
    list(map(retriever.add, ref.CORPUS))
    out = {q: retriever.search(q, k_each=5, k_out=5) for q in QUERIES}
    runs = {q: {m: [(d.doc_id, round(s, 4)) for d, s in hits] for m, hits in r.items()} for q, r in out.items()}
    return runs, out[PARAPHRASE]["fused"][0][1] == out[PARAPHRASE]["fused"][1][1]


def rank(run, gold):
    """1-based rank of `gold` in the bm25, dense and fused lists; None when absent."""
    return tuple(next((i + 1 for i, (d, _) in enumerate(run[m]) if d == gold), None) for m in ("bm25", "dense", "fused"))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    (mock, mock_tie), model, shipped = demo(ref), load_model(), ref.mock_embed
    ref.mock_embed = lambda text, dim=96: encode(text, model)
    try:
        real, real_tie = demo(ref)
    finally:
        ref.mock_embed = shipped
    return {
        "mock": mock[PARAPHRASE], "real": real[PARAPHRASE], "ties": (mock_tie, real_tie),
        "claims": [rank(mock[q], g) for q, g in QUERIES.items()],
        "d2_shares": sorted(set(ref.tokenize(PARAPHRASE)) & set(ref.tokenize(ref.CORPUS[1].body))),
        "d1_cancel": [t for t in ref.tokenize(ref.CORPUS[0].body) if t.startswith("cancel")],
    }


def verify(result):
    r, mock, real = result, result["mock"], result["real"]
    got = ([d for d, _ in mock["dense"]], [d for d, _ in real["dense"]], real["dense"][:2], mock["dense"][:2],
           [d for d, _ in real["fused"][:2]], r["ties"])
    words = (r["d2_shares"], mock["bm25"], r["d1_cancel"])  # the query's overlap with d2, and d1's cancel* tokens
    return [
        practice.Check("ANSWER: MiniLM puts d2 first in the dense list, the mock had it second",
                       got == (["d3", "d2", "d6", "d1", "d4"], ["d2", "d1", "d3", "d5", "d4"], [("d2", 0.6247), ("d1", 0.4517)],
                               [("d3", 0.4047), ("d2", 0.4016)], ["d2", "d3"], (True, False)),
                       f"dense mock {got[0]} -> MiniLM {got[1]}; top cosines {got[2]} (mock {got[3]}); "
                       f"fused top-2 {got[4]}; exact d2/d3 fused tie mock, MiniLM: {got[5]}"),
        practice.Check("FINDING: the 'paraphrased' query shares its words with d2, and d1 never says cancel",
                       words == (["cancelled", "do", "uploads"], [("d2", 4.8541), ("d3", 1.6862)], []),
                       f"shared tokens {words[0]}; BM25 {words[1]}; cancel* in d1: {words[2]}"),
        practice.Check("FINDING: none of the doc's (BM25, dense, RRF) rank claims matches the run",
                       r["claims"] == [(1, 1, 1), (1, 2, 1), (1, 1, 1)],
                       f"doc says (1, 4, 1), (6, 1, 1), (3, 3, 1); measured {r['claims']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
