"""Exercise 4 -- the centroid-sentence summary costs 4 of 45 top-1 hits, because it repeats words the chunk already has.

    Add a `summary` field per chunk: a one-sentence centroid description. Re-run the eval with the summary appended to the chunk body. Measure the recall lift.

Reading of the exercise: "a one-sentence centroid description" is read
extractively, since the lesson has no generator: the summary is the chunk's
own sentence whose `mock_embed` vector is nearest the mean of its sentences'
vectors. The lesson's `Chunk` has no `summary` field, so each chunk gets one
as an attribute, and the text the lesson's `DenseIndex` embeds becomes body +
" " + summary. Gold offsets are unchanged, so `eval_recall` scores it as is. A
second, abstractive-shaped summary is run beside it: "This section covers a,
b, c." naming the chunk's three most frequent content words. The lift is
counted in hits over 5 strategies x 9 queries.

**ANSWER: the lift is negative: top-1 hits fall from 39 to 35 of 45.**
Sentence goes 7 -> 6, recursive 9 -> 8, semantic 6 -> 5, structural 9 -> 8,
and fixed stays 8. At k=3 and k=5 every strategy is 9/9 before and after, so
k=1 is the only place a lift could show.

**FINDING: a centroid sentence adds no words, only weight.** The summary is
a sentence the chunk already contains, so for a one-sentence chunk it doubles
every count, and the unit-normalised embedding is unchanged: 22 of the 50
chunks embed identically, including 17 of the 20 semantic chunks. Elsewhere
it tilts the chunk towards its most typical sentence and away from the less
typical ones.

**FINDING: the keyword summary gains 1 of 45.** It lifts sentence 7 -> 8
and leaves every other strategy where it was, 40 of 45 in all.

Structure: `centroid_summary` and `keyword_summary` write the summary;
`summarised` wraps a lesson strategy; `unchanged` counts chunks whose vector
the summary does not move.
"""

from __future__ import annotations

import collections
import dataclasses
import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "64-chunking-strategies-advanced"
KS = (1, 3, 5)
ORDER = ("fixed", "sentence", "recursive", "semantic", "structural")
STOP = set("the a an is of to and in for at by it its that so when every one each are be with on as this".split())


def centroid_summary(ref, text):
    """The chunk's own sentence nearest the mean of its sentence embeddings."""
    sents = [s for _, _, s in ref.split_sentences(text)] or [text]
    vecs = [ref.mock_embed(s) for s in sents]
    centroid = [sum(col) / len(vecs) for col in zip(*vecs)]
    return max(sents, key=lambda s: ref.cosine(ref.mock_embed(s), centroid))


def keyword_summary(ref, text):
    """A templated sentence naming the chunk's three most frequent content words."""
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2]
    top = [w for w, _ in collections.Counter(words).most_common(3)]
    return "This section covers " + ", ".join(top) + "."


def summarised(ref, fn, summary):
    """Chunks with a `summary` field; the text the index embeds is body + summary."""

    def run(doc_id, text):
        out = []
        for c in fn(doc_id, text):
            s = summary(ref, c.text)
            chunk = dataclasses.replace(c, text=c.text + " " + s)
            chunk.summary = s
            out.append(chunk)
        return out

    return run


def hits(ref, fn, fixture):
    return [round(v * 9) for v in ref.eval_recall(fn, fixture, KS).values()]


def unchanged(ref, fn, fixture):
    """Chunks whose embedding the centroid summary leaves exactly as it was."""
    same = total = 0
    for d in fixture:
        for c in fn(d["doc_id"], d["text"]):
            total += 1
            after = ref.mock_embed(c.text + " " + centroid_summary(ref, c.text))
            same += max(abs(a - b) for a, b in zip(after, ref.mock_embed(c.text))) < 1e-12
    return same, total


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fixture = ref.build_fixture()
    table = {}
    for name in ORDER:
        fn = ref.STRATEGIES[name]
        table[name] = {"base": hits(ref, fn, fixture),
                       "centroid": hits(ref, summarised(ref, fn, centroid_summary), fixture),
                       "keywords": hits(ref, summarised(ref, fn, keyword_summary), fixture),
                       "unchanged": unchanged(ref, fn, fixture)}
    first = summarised(ref, ref.STRATEGIES["sentence"], centroid_summary)("prose", ref.PROSE_DOC)[0]
    return {"table": table, "field": first.text.endswith(" " + first.summary)}


def columns(table):
    col = {arm: [table[n][arm] for n in ORDER] for arm in ("base", "centroid", "keywords")}
    deep = sorted({tuple(h[1:]) for rows in col.values() for h in rows})
    return {arm: [h[0] for h in rows] for arm, rows in col.items()}, deep


def verify(result):
    at1, deep = columns(result["table"])
    same = [result["table"][n]["unchanged"] for n in ORDER]
    return [
        practice.Check(
            "ANSWER: appending the centroid sentence lowers top-1 hits from 39 to 35 of 45",
            (at1["base"], at1["centroid"], deep) == ([8, 7, 9, 6, 9], [8, 6, 8, 5, 8], [(9, 9)]),
            f"hits@1 {ORDER}: base {at1['base']} ({sum(at1['base'])}), centroid {at1['centroid']} "
            f"({sum(at1['centroid'])}); hits@3,5 across all arms {deep}",
        ),
        practice.Check(
            "FINDING: a centroid sentence adds no words -- 22 of 50 chunks embed identically",
            (same, result["field"]) == ([(1, 8), (1, 6), (1, 7), (17, 20), (2, 9)], True),
            f"unchanged/total per strategy {same}; summary field set: {result['field']}",
        ),
        practice.Check(
            "FINDING: the keyword summary gains 1 of 45, on sentence alone",
            at1["keywords"] == [8, 8, 9, 6, 9],
            f"hits@1 with keywords {at1['keywords']} ({sum(at1['keywords'])})",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
