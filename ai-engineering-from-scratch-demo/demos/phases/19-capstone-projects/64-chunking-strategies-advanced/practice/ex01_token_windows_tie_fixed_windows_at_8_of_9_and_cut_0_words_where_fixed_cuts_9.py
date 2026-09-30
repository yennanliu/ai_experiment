"""Exercise 1 -- token windows tie fixed windows at 8 of 9 and cut 0 words where fixed cuts 9.

    Add a sixth strategy: token-window using `tiktoken` instead of character counts. Compare against fixed-window on the same fixture.

Reading of the exercise: the sixth strategy is `token_window`, the lesson's
`fixed_window` with the window counted in `cl100k_base` tokens instead of
characters; each chunk keeps character offsets (summed token byte lengths --
the fixture is ASCII), so the lesson's `eval_recall` scores it unchanged.
"Compare" is run at a matched budget: the lesson's 400/80-character window is
converted with the fixture's own ratio (2,189 characters, 413 tokens), which
gives 75/15 tokens. Then both are swept over 18 matched budgets, 40-125
tokens, to see whether the unit or the size decides recall.

**ANSWER: at the matched budget the two tie.** Both make 8 chunks and hit
8/9 queries at k=1 and 9/9 at k=3 and k=5. The one measurable difference is the boundaries:
fixed-window cuts 9 chunk boundaries inside a word, the token window 0.

**FINDING: recall@3 and recall@5 cannot separate any two strategies here.**
`eval_recall` builds one index per document, and neither windowing gives any
document more than 3 chunks, so k=3 returns the whole document.

**FINDING: window size moves recall@1 more than the unit does.** Over the 18
matched budgets the token window wins 2, fixed-window wins 5, and 11 tie;
each ranges over 7-9 of 9 hits as the size changes.

Structure: `token_window` is the new strategy; `mid_word` counts boundaries
between two alphanumerics; `sweep` runs both units at matched budgets.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import tiktoken
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs tiktoken: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "64-chunking-strategies-advanced"
KS, CHAR_SIZE = (1, 3, 5), 400
SWEEP = range(40, 130, 5)


def token_window(ref, enc, doc_id, text, size, overlap):
    """`fixed_window`, counted in tokens; chunks carry character offsets."""
    ids = enc.encode(text)
    offsets = [0]
    for tok in ids:
        offsets.append(offsets[-1] + len(enc.decode_single_token_bytes(tok)))
    out, i = [], 0
    while i < len(ids):
        end = min(i + size, len(ids))
        a, b = offsets[i], offsets[end]
        out.append(ref.Chunk(doc_id, "token", a, b, text[a:b]))
        if end == len(ids):
            break
        i += size - overlap
    return out


def mid_word(chunks, text):
    cuts = [p for c in chunks for p in (c.start, c.end) if 0 < p < len(text)]
    return sum(text[p - 1].isalnum() and text[p].isalnum() for p in cuts)


def hits(ref, fn, fixture):
    n = sum(len(d["queries"]) for d in fixture)
    return [round(v * n) for v in ref.eval_recall(fn, fixture, KS).values()]


def describe(ref, fn, fixture):
    chunks = [(fn(d["doc_id"], d["text"]), d["text"]) for d in fixture]
    return {"hits": hits(ref, fn, fixture), "chunks": sum(len(c) for c, _ in chunks),
            "mid_word": sum(mid_word(c, t) for c, t in chunks),
            "max_per_doc": max(len(c) for c, _ in chunks)}


def sweep(ref, enc, fixture, ratio):
    wins = {"token": 0, "fixed": 0, "tie": 0}
    ranges = {"token": set(), "fixed": set()}
    for size in SWEEP:
        chars = round(size * ratio)
        tok = hits(ref, lambda d, t, s=size: token_window(ref, enc, d, t, s, s // 5), fixture)[0]
        fix = hits(ref, lambda d, t, c=chars: ref.fixed_window(d, t, c, c // 5), fixture)[0]
        wins["token" if tok > fix else "fixed" if fix > tok else "tie"] += 1
        ranges["token"].add(tok)
        ranges["fixed"].add(fix)
    return wins, {k: (min(v), max(v)) for k, v in ranges.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    enc = tiktoken.get_encoding("cl100k_base")
    fixture = ref.build_fixture()
    n_chars = sum(len(d["text"]) for d in fixture)
    n_tokens = sum(len(enc.encode(d["text"])) for d in fixture)
    size = round(CHAR_SIZE * n_tokens / n_chars)
    tok = lambda d, t: token_window(ref, enc, d, t, size, size // 5)  # noqa: E731
    wins, ranges = sweep(ref, enc, fixture, n_chars / n_tokens)
    return {"ascii": all(d["text"].isascii() for d in fixture), "chars": n_chars,
            "tokens": n_tokens, "size": size, "overlap": size // 5,
            "fixed": describe(ref, ref.STRATEGIES["fixed"], fixture),
            "token": describe(ref, tok, fixture), "wins": wins, "ranges": ranges}


def verify(result):
    r, fx, tk = result, result["fixed"], result["token"]
    return [
        practice.Check(
            "ANSWER: at a matched 75/15-token budget the token window ties fixed-window",
            (r["ascii"], r["chars"], r["tokens"], r["size"], r["overlap"]) == (True, 2189, 413, 75, 15)
            and fx["hits"] == tk["hits"] == [8, 9, 9] and fx["chunks"] == tk["chunks"] == 8
            and (fx["mid_word"], tk["mid_word"]) == (9, 0),
            f"{r['chars']} chars / {r['tokens']} tokens -> {r['size']}/{r['overlap']}; hits@1,3,5 "
            f"fixed {fx['hits']} token {tk['hits']}; chunks {fx['chunks']}/{tk['chunks']}; "
            f"mid-word cuts {fx['mid_word']} vs {tk['mid_word']}",
        ),
        practice.Check(
            "FINDING: recall@3 and @5 are saturated -- no document gets more than 3 chunks",
            fx["max_per_doc"] == tk["max_per_doc"] == 3,
            f"most chunks in one document: fixed {fx['max_per_doc']}, token {tk['max_per_doc']}",
        ),
        practice.Check(
            "FINDING: window size moves recall@1 more than the unit does",
            r["wins"] == {"token": 2, "fixed": 5, "tie": 11}
            and r["ranges"] == {"token": (7, 9), "fixed": (7, 9)},
            f"over {len(SWEEP)} budgets: {r['wins']}; hits@1 ranges {r['ranges']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
