"""Exercise 5 — the follow-up fails on a question mark, and history cannot reach retrieval.

    Build a conversation-aware RAG pipeline: maintain a history of the last 3
    exchanges and include them in the prompt alongside the retrieved chunks.
    Test with follow-up questions like "What about enterprise?" after asking
    about pricing.

Reading of the exercise: history is kept as the last three (question, answer)
pairs and injected into the prompt, exactly as written, and the test is the
exercise's own follow-up after its own pricing question. Whether it works is
measured at retrieval, which is where a follow-up either resolves or does not.

**ANSWER: "What about enterprise?" scores 0.0 against every chunk, and the
history cannot help.** `RAGPipeline.query` embeds the question alone and calls
`search` before `build_rag_prompt` is ever reached, so anything added to the
prompt arrives after the decision it would have informed. The retrieval for the
follow-up is byte-identical with and without three turns of history in the
prompt.

**MECHANISM: the question mark.** `build_vocabulary` is `doc.lower().split()`,
so the corpus contains `enterprise` and `enterprise.` and the query contains
`enterprise?` -- a token in neither. Drop the one character and the same
follow-up retrieves three chunks with non-zero scores. One byte is the whole
difference between a dead query and a live one.

**FINDING: alive, its only content word is the one the corpus weighs least.**
`What about enterprise` embeds to a single non-zero dimension: `what`, `about`
and `tiers` are not in the vocabulary at all, and `enterprise` occurs in all
five chunks, so `compute_idf` gives it 1.000 -- the floor. The ranking is pure
term frequency on the corpus's least discriminating word.

**FINDING: the prompt is inert for a second, independent reason.**
`simple_generate` reconstructs the query by splitting the prompt on
`"question:"` and then scans `retrieved_chunks`, not the prompt. Three turns of
history in the prompt leave the answer byte-identical on both follow-ups.

**CONTROL: fix the tokeniser, not the prompt.** Stripping trailing punctuation
on both sides takes the vocabulary from 270 to 258 words and the follow-up --
question mark and all -- from 0.0 everywhere to the pricing chunk at rank 1.
That is the intervention the exercise's own test case needed, and it is nowhere
near the prompt.

Structure: `Conversational` wraps the lesson's pipeline with a three-turn
history and an `inject` flag for the prompt injection the exercise asks for;
`normalised` is the control pipeline.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "06-rag"
FIRST = "what are the pricing tiers"
FOLLOW_UPS = ["What about enterprise?", "What about enterprise"]
TOP_K, HISTORY = 3, 3


class Conversational:
    """The lesson's pipeline plus the last three exchanges."""

    def __init__(self, ref, pipeline):
        self.ref, self.pipeline, self.turns = ref, pipeline, []

    def ask(self, question, inject=False):
        result = self.pipeline.query(question, TOP_K)
        hits = result["retrieved"]
        prompt = result["prompt"]
        if inject and self.turns:
            past = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in self.turns[-HISTORY:])
            prompt = f"Conversation so far:\n{past}\n\n{prompt}"
        answer = self.ref.simple_generate(prompt, [h["chunk"] for h in hits])
        self.turns.append((question, answer))
        return {"indices": [h["index"] for h in hits], "answer": answer, "prompt": prompt,
                "scores": [round(h["score"], 3) for h in hits]}


def strip_punctuation(text):
    return " ".join(word.strip(".,:;!?") for word in text.lower().split())


def normalised(ref, question):
    """The control: the same pipeline with trailing punctuation stripped at tokenisation."""
    pipeline = ref.RAGPipeline()
    pipeline.index([strip_punctuation(d) for d in ref.SAMPLE_DOCUMENTS])
    embedded = ref.tfidf_embed(strip_punctuation(question), pipeline.vocab, pipeline.idf)
    hits = ref.search(embedded, pipeline.embeddings, TOP_K)
    return {"vocab": len(pipeline.vocab), "indices": [i for i, _ in hits],
            "scores": [round(s, 3) for _, s in hits]}


def run(ref, pipeline, question, inject):
    chat = Conversational(ref, pipeline)
    chat.ask(FIRST, inject)
    return chat.ask(question, inject)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pipeline = ref.RAGPipeline()
    pipeline.index(ref.SAMPLE_DOCUMENTS)
    plain = [run(ref, pipeline, q, False) for q in FOLLOW_UPS]
    injected = [run(ref, pipeline, q, True) for q in FOLLOW_UPS]
    normal = normalised(ref, FOLLOW_UPS[0])
    return {
        **weights(ref, pipeline), **arms(plain, injected),
        "normal_vocab": normal["vocab"], "shipped_vocab": len(pipeline.vocab),
        "normal_scores": normal["scores"], "normal_indices": normal["indices"],
        "pricing_chunk": next(i for i, c in enumerate(pipeline.chunks)
                              if "$500 per month" in c),
    }


def weights(ref, pipeline):
    vocab = set(pipeline.vocab)
    live = ref.tfidf_embed(FOLLOW_UPS[1], pipeline.vocab, pipeline.idf)
    return {"nonzero": sum(1 for x in live if x != 0),
            "missing": [w for w in ("what", "about", "tiers") if w not in vocab],
            "enterprise_idf": round(pipeline.idf[pipeline.vocab.index("enterprise")], 3),
            "min_idf": round(min(pipeline.idf), 3),
            "documents_with_enterprise": sum("enterprise" in c.lower().split()
                                             for c in pipeline.chunks),
            "tokens": {w: w in vocab for w in ("enterprise", "enterprise.", "enterprise?")}}


def arms(plain, injected):
    return {"dead_scores": plain[0]["scores"], "live_scores": plain[1]["scores"],
            "dead_indices": plain[0]["indices"], "live_indices": plain[1]["indices"],
            "retrieval_unchanged": [a["indices"] == b["indices"]
                                    for a, b in zip(plain, injected)],
            "answers_unchanged": [a["answer"] == b["answer"]
                                  for a, b in zip(plain, injected)],
            "history_in_prompt": [("Conversation so far" in b["prompt"])
                                  for b in injected]}


def verify(result):
    dead, live = result["dead_scores"], result["live_scores"]
    normal = result["normal_scores"]
    return [
        practice.Check(
            "ANSWER: the follow-up scores 0.0 everywhere, and history cannot reach retrieval",
            all([set(dead) == {0.0}, all(result["retrieval_unchanged"]),
                 all(result["history_in_prompt"])]),
            f"{FOLLOW_UPS[0]!r} scores {dead} against every chunk, so the ranking "
            f"{result['dead_indices']} is `search`'s insertion order. History reaches the "
            f"prompt on {sum(result['history_in_prompt'])} of {len(FOLLOW_UPS)} runs and "
            f"leaves retrieval identical on {sum(result['retrieval_unchanged'])}: `query` "
            "embeds and searches before `build_rag_prompt` is called",
        ),
        practice.Check(
            "MECHANISM: the question mark is the whole difference",
            all([result["tokens"] == {"enterprise": True, "enterprise.": True,
                                      "enterprise?": False},
                 set(live) != {0.0}]),
            f"`build_vocabulary` is doc.lower().split(), so the corpus holds "
            f"{[w for w, ok in result['tokens'].items() if ok]} and the query holds "
            f"'enterprise?', which is in neither. Dropping the character takes the scores "
            f"from {dead} to {live}",
        ),
        practice.Check(
            "FINDING: alive, its only content word is the corpus's lowest-weighted one",
            all([result["nonzero"] == 1, result["missing"] == ["what", "about", "tiers"],
                 result["enterprise_idf"] == result["min_idf"]]),
            f"{FOLLOW_UPS[1]!r} embeds to {result['nonzero']} non-zero dimension: "
            f"{result['missing']} are not in the vocabulary at all, and 'enterprise' "
            f"appears in {result['documents_with_enterprise']} of the chunks, so its IDF is "
            f"{result['enterprise_idf']} -- the floor. The ranking {result['live_indices']} "
            "is pure term frequency on the one word the corpus weighs least",
        ),
        practice.Check(
            "FINDING: the prompt is inert for a second, independent reason",
            all(result["answers_unchanged"]),
            f"`simple_generate` reconstructs the query by splitting the prompt on "
            f"'question:' and then scans retrieved_chunks, not the prompt. The answer is "
            f"byte-identical with and without history on "
            f"{sum(result['answers_unchanged'])} of {len(FOLLOW_UPS)} follow-ups",
        ),
        practice.Check(
            "CONTROL: fix the tokeniser, not the prompt",
            all([set(normal) != {0.0},
                 result["normal_indices"][0] == result["pricing_chunk"],
                 result["normal_vocab"] < result["shipped_vocab"]]),
            f"stripping trailing punctuation on both sides at tokenisation time takes the "
            f"vocabulary {result['shipped_vocab']} -> {result['normal_vocab']} and the "
            f"follow-up as written, question mark included, from {dead} to {normal}, "
            f"returning {result['normal_indices']} with the pricing chunk first. One change "
            "to the tokeniser, none to the prompt",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
