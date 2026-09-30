"""Exercise 2 -- structural keeps 3 of 3 only as one whole-document chunk, and one `#` comment per block drops it to 2.

    Inject a 30 percent fraction of code blocks into the prose fixture. Re-run the table. Explain why every strategy except structural markdown loses recall.

Reading of the exercise: "30 percent" is read by characters. One fenced
Python block goes after each of the prose fixture's three paragraphs, and the
three gold spans are re-located in the new text. Three variants are run
through the lesson's `STRATEGIES` and `eval_recall` on the prose document:
code unrelated to the prose (30.3% of characters), the same code with a
one-line `#` comment opening each block (32.7%), and code that implements
what the prose describes, commented the same way (31.8%).

**ANSWER: with unrelated code the premise holds at k=1, and structural
survives because it never cuts.** Hits@1 of 3 go fixed 2 -> 1, sentence
2 -> 1, recursive 3 -> 1, semantic 2 -> 1; structural stays 3. The prose
fixture has no headings, so `structural_markdown` returns the whole
1,271-character document as a single chunk, and that chunk overlaps every
gold span at any k. The others mix code into prose chunks: all 6 top-1 misses
of fixed, sentence and recursive are won by a chunk that holds a code fence.
At k=3 only semantic still loses (3 -> 2); the other three keep 3/3.

**FINDING: one `#` comment per block reverses the result.**
`structural_markdown` reads a Python comment at line start as a heading, and
it never emits the text before the first heading. The first 290 characters,
the retry-budget paragraph and the fence line after it, land in no chunk, so "when does the retry
budget reset" is lost at every k: structural scores 2/3 at k=1, 3 and 5.

**FINDING: code that documents the prose does not hurt the others.** With
the related blocks, fixed, sentence and recursive score 3/3 at k=1 (up from
2, 2, 3), semantic keeps 2, and structural drops to 1.

Structure: `inject` builds the document; `table` is hits@1,3,5 per strategy;
`top1_misses` names what beat the gold chunk; `structural_view` measures the
characters the structural splitter drops.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "64-chunking-strategies-advanced"
KS = (1, 3, 5)
FENCE = "```python\n{}\n```"
UNRELATED = [
    "def load_rows(path):\n    with open(path) as fh:\n        rows = [line.split(',') for line in fh]\n"
    "    return [r for r in rows if r and r[0]]",
    "def by_date(rows):\n    return sorted(rows, key=lambda r: r[2])\n\n"
    "def top(rows, n=10):\n    return by_date(rows)[:n]",
    "for row in top(load_rows('data.csv')):\n    print(' | '.join(cell.ljust(12) for cell in row))",
]
COMMENTS = ["# load the csv", "# newest first", "# print a table"]
RELATED = [
    "# retry budget\nbudget = RetryBudget(per_minute=64)\nif budget.spend():\n    client.retry(request)\n"
    "else:\n    client.cool_down()",
    "# multipart abort\nif upload.failed_parts >= bucket.abort_threshold:\n    storage.abort(upload.key)\n"
    "    storage.release(upload.key)",
    "# permission check\ndef authorize(principal, resource, action):\n"
    "    return policy.evaluate(principal, resource, action)",
]


def inject(ref, bodies):
    """One fenced block after each of the prose fixture's three paragraphs."""
    blocks = [FENCE.format(b) for b in bodies]
    paras = ref.PROSE_DOC.split("\n\n")
    text = "\n\n".join(p + "\n\n" + b for p, b in zip(paras, blocks))
    queries = [(q, ref._locate(text, span)) for q, span in ref.PROSE_QUERIES]
    return {"doc_id": "prose", "text": text, "queries": queries}, sum(map(len, blocks)) / len(text)


def table(ref, doc):
    return {n: [round(v * 3) for v in ref.eval_recall(fn, [doc], KS).values()]
            for n, fn in ref.STRATEGIES.items()}


def top1_misses(ref, doc):
    """Per strategy: (k=1 misses, how many of them a code-bearing chunk won)."""
    out = {}
    for name, fn in ref.STRATEGIES.items():
        idx = ref.DenseIndex()
        for c in fn(doc["doc_id"], doc["text"]):
            idx.add(c)
        tops = [(idx.search(q, 1)[0], span) for q, span in doc["queries"]]
        misses = [c for c, span in tops if not c.overlaps(*span)]
        out[name] = (len(misses), sum("```" in c.text for c in misses))
    return out


def structural_view(ref, doc):
    chunks = ref.structural_markdown(doc["doc_id"], doc["text"])
    covered = set().union(*(range(c.start, c.end) for c in chunks))
    lost = [q for q, (a, b) in doc["queries"] if not covered & set(range(a, b))]
    return {"chunks": len(chunks), "length": len(doc["text"]), "first": chunks[0].start, "uncovered": len(doc["text"]) - len(covered),
            "whole_doc": len(chunks) == 1 and chunks[0].end - chunks[0].start == len(doc["text"]), "lost": lost}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base = ref.build_fixture()[0]
    plain, f_plain = inject(ref, UNRELATED)
    commented, f_comm = inject(ref, [c + "\n" + b for c, b in zip(COMMENTS, UNRELATED)])
    related, f_rel = inject(ref, RELATED)
    return {"fractions": [round(f, 3) for f in (f_plain, f_comm, f_rel)],
            "base": table(ref, base), "plain": table(ref, plain), "commented": table(ref, commented),
            "related": table(ref, related), "misses": top1_misses(ref, plain),
            "s_plain": structural_view(ref, plain), "s_comm": structural_view(ref, commented)}


def verify(result):
    r, sp, sc = result, result["s_plain"], result["s_comm"]
    at1 = {arm: [h[0] for h in r[arm].values()] for arm in ("base", "plain", "commented", "related")}
    return [
        practice.Check(
            "ANSWER: unrelated code costs every strategy a k=1 hit except structural, which is one chunk",
            (r["fractions"][0], at1["base"], list(r["plain"].values()), sp["chunks"], sp["whole_doc"], sp["length"])
            == (0.303, [2, 2, 3, 2, 3], [[1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 2, 2], [3, 3, 3]], 1, True, 1271)
            and list(r["misses"].values())[:3] == [(2, 2)] * 3,
            f"code {r['fractions'][0]:.1%}; hits@1,3,5 before {r['base']} after {r['plain']}; structural "
            f"chunks {sp['chunks']} (whole {sp['length']}-char doc: {sp['whole_doc']}); misses won by code {r['misses']}",
        ),
        practice.Check(
            "FINDING: a `#` comment per block is a heading to structural, which drops the preamble",
            (r["commented"]["structural"], sc["first"], sc["uncovered"], sc["lost"])
            == ([2, 2, 2], 290, 290, ["when does the retry budget reset"]),
            f"code {r['fractions'][1]:.1%}; structural {r['commented']['structural']}; first chunk at "
            f"{sc['first']}, {sc['uncovered']} chars in no chunk, unreachable: {sc['lost']}",
        ),
        practice.Check(
            "FINDING: code that documents the prose does not hurt the others",
            at1["related"] == [3, 3, 3, 2, 1],
            f"code {r['fractions'][2]:.1%}; hits@1,3,5 {r['related']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
