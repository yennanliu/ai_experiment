"""Exercise 4 — the baseline it names hangs, and parents cost 1.34x for the same recall.

    Implement the parent-child chunking strategy on the sample documents. Use
    child_size=30 and parent_size=100. Search with child chunks but return
    parent chunks in the prompt. Compare the generated answers to standard
    chunking with chunk_size=50.

Reading of the exercise: the generator is lesson 06's `simple_generate`, loaded
from the reference repo, because exercise 3 of this lesson names it and this
lesson ships no generation function at all. The baseline is run at
`chunk_text(doc, 50, 0)`, and why is the first finding.

**ANSWER: `chunk_text(doc, 50)` does not terminate.** The default overlap is 50,
so the step is `chunk_size - overlap == 0` and the `while start < len(words)`
loop appends the same chunk forever. The baseline the exercise names has to be
run at an overlap the exercise does not give, and 0 is the only one that keeps
50-word chunks 50 words long.

**ANSWER: the two strategies tie on recall and parent-child costs 34% more
prompt.** Both put the answering document in the top 3 for 5 of 5 queries.
Summed over the five, the parents returned are 931 words against the standard
chunks' 693 -- 1.34x -- because a parent is 100 words where a standard chunk is
50 and three children can map to fewer than three distinct parents.

**FINDING: the generated answers differ on 2 of the 5 queries**, and not in the
direction the strategy is sold on: the larger context changes the sentence
`simple_generate` picks, because it scores every sentence in every returned
chunk and a 100-word parent offers more of them.

**FINDING: 5 of the 24 children are under 10 words.**
`create_parent_child_chunks` tiles children across each parent with
`child_start += child_size` and stops at the parent boundary, so a 91-word
document split into one 100-word parent yields children of 30, 30, 30 and 1
words. Those fragments are embedded and searchable like any other child.

**MECHANISM: the child index and the parent index are different corpora.**
Children are 24 documents and parents are 7, so the IDF the search runs on is
computed over the children while the text returned is the parents'. Nothing is
wrong with that -- it is the point of the strategy -- but it means the score
attached to a returned parent was never computed for it.

Structure: `layered` builds the parent/child corpus over all six documents and
`standard` the 50-word baseline; `embed` and `search` are the shared retrieval
path, `picks` resolves child hits to parents, and `bounded` is `chunk_text` with
a cap so the non-terminating case can be observed without running it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "07-advanced-rag"
QUERIES = ["what encryption is used at rest", "what is the enterprise refund window",
           "what is the starter rate limit", "what uptime is guaranteed",
           "how much does professional cost"]
GOLD = [2, 0, 3, 5, 1]
PARENT, CHILD, BASELINE, TOP_K = 100, 30, 50, 3


def bounded(text, chunk_size, overlap, cap=500):
    words, made, start = text.split(), 0, 0
    while start < len(words) and made < cap:
        made, start = made + 1, start + chunk_size - overlap
    return made, start   # `chunk_text`'s loop, with a cap


def layered(ref):
    """Parents, children, child->parent, and the document each parent came from."""
    parents, children, mapping, owner = [], [], {}, []
    for index, document in enumerate(ref.SAMPLE_DOCUMENTS):
        got = ref.create_parent_child_chunks(" ".join(document.split()), PARENT, CHILD)
        base_p, base_c = len(parents), len(children)
        parents, children = parents + got[0], children + got[1]
        owner += [index] * len(got[0])
        mapping.update({base_c + c: base_p + p for c, p in got[2].items()})
    return parents, children, mapping, owner


def standard(ref):
    pieces = [ref.chunk_text(" ".join(d.split()), BASELINE, 0) for d in ref.SAMPLE_DOCUMENTS]
    return ([c for group in pieces for c in group],
            [i for i, group in enumerate(pieces) for _ in group])


def embed(ref, texts):
    vocab = ref.build_vocabulary(texts)
    idf = ref.compute_idf(texts, vocab)
    return vocab, idf, [ref.tfidf_embed(t, vocab, idf) for t in texts]


def search(ref, query, corpus, k=TOP_K):
    vocab, idf, rows = corpus
    return [i for i, _ in ref.vector_search(ref.tfidf_embed(query, vocab, idf), rows, k)]


def picks(ref, parts, query):
    mapping = parts["layered"][2]
    children = search(ref, query, parts["child_index"])
    return (list(dict.fromkeys(mapping[i] for i in children)),
            search(ref, query, parts["standard_index"]))


def one_query(ref, generate, parts, query):
    parents, owner = parts["layered"][0], parts["layered"][3]
    chunks, chunk_owner = parts["standard"]
    picked, plain = picks(ref, parts, query)
    texts = ([parents[p] for p in picked], [chunks[i] for i in plain])
    return {"parent_docs": [owner[p] for p in picked],
            "standard_docs": [chunk_owner[i] for i in plain],
            "parent_words": sum(len(t.split()) for t in texts[0]),
            "standard_words": sum(len(t.split()) for t in texts[1]),
            "answers": [generate(ref.build_rag_prompt(query, t), t) for t in texts]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    generate = parity.load_reference(PHASE, "06-rag", "main").simple_generate
    layers, baseline = layered(ref), standard(ref)
    parts = {"layered": layers, "standard": baseline,
             "child_index": embed(ref, layers[1]),
             "standard_index": embed(ref, baseline[0])}
    rows = [one_query(ref, generate, parts, q) for q in QUERIES]
    emitted, final = bounded(ref.SAMPLE_DOCUMENTS[0], BASELINE, BASELINE)
    children = layers[1]
    return {
        "step": BASELINE - BASELINE, "emitted": emitted, "start": final,
        "parents": len(layers[0]), "children": len(children), "standard": len(baseline[0]),
        "tiny": sum(1 for c in children if len(c.split()) < 10),
        "child_sizes": [len(c.split()) for c in children[:4]],
        "queries": len(QUERIES), **totals(rows),
    }


def totals(rows):   # summed over the five queries
    return {"parent_recall": sum(g in r["parent_docs"] for r, g in zip(rows, GOLD)),
            "standard_recall": sum(g in r["standard_docs"] for r, g in zip(rows, GOLD)),
            "parent_words": sum(r["parent_words"] for r in rows),
            "standard_words": sum(r["standard_words"] for r in rows),
            "same_answer": sum(r["answers"][0] == r["answers"][1] for r in rows)}


def verify(result):
    ratio = result["parent_words"] / result["standard_words"]
    return [
        practice.Check(
            "ANSWER: chunk_text(doc, 50) does not terminate",
            all([result["step"] == 0, result["emitted"] == 500, result["start"] == 0]),
            f"the default overlap is {BASELINE}, so the step is {result['step']} and a "
            f"bounded clone makes {result['emitted']} chunks with start still at "
            f"{result['start']}. The baseline has to be run at an overlap the exercise does "
            "not give, and 0 is the only one that keeps 50-word chunks 50 words long",
        ),
        practice.Check(
            "ANSWER: the strategies tie on recall and parent-child costs 34% more prompt",
            all([result["parent_recall"] == result["standard_recall"] == result["queries"],
                 1.3 < ratio < 1.4]),
            f"both put the answering document in the top {TOP_K} for "
            f"{result['parent_recall']} of {result['queries']} queries, and the parents "
            f"returned are {result['parent_words']} words against {result['standard_words']} "
            f"-- {ratio:.2f}x, because a parent is {PARENT} words where a chunk is {BASELINE}",
        ),
        practice.Check(
            "FINDING: the generated answers differ on 2 of the 5 queries",
            result["same_answer"] == 3,
            f"{result['queries'] - result['same_answer']} of {result['queries']} answers "
            f"change, and not in the direction the strategy is sold on: `simple_generate` "
            f"scores every sentence it is given, and a {PARENT}-word parent offers more of "
            f"them than a {BASELINE}-word chunk",
        ),
        practice.Check(
            "FINDING: 5 of the 24 children are under 10 words",
            all([result["tiny"] == 5, result["children"] == 24,
                 result["child_sizes"] == [30, 30, 30, 1]]),
            f"children are tiled with child_start += child_size and stop at the parent "
            f"boundary, so a 91-word document under one {PARENT}-word parent yields "
            f"{result['child_sizes']}. {result['tiny']} of {result['children']} are "
            "fragments, embedded and searchable like any other child",
        ),
        practice.Check(
            "MECHANISM: the child index and the parent index are different corpora",
            all([result["children"] > result["parents"], result["parents"] == 7]),
            f"the search runs over {result['children']} children and the prompt carries "
            f"{result['parents']} parents, so the score was computed against the children's "
            "IDF. That is the point of the strategy -- and it means the number attached to "
            "a returned parent was never computed for that text",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
