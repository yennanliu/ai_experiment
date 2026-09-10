r"""Exercise 1 — alternation order beats the pattern.

    **Easy.** Extend `tokenize` to keep URLs as single tokens. Test:
    `tokenize("Visit https://example.com today.")` should produce one URL token.

Reading of the exercise: the extension is one alternative added to `WORD_RE`,
and the exercise's own test cannot tell a working one from a broken one. Two
things decide the answer and neither is the URL pattern. First, placement:
`re` alternation is ordered and first-match-wins at each position, so a URL
alternative appended after `[A-Za-z]+` never fires -- "https" is matched as a
word before the URL branch is ever tried -- and the tokenizer's output stays
byte-identical to the lesson's on all 12 sentences here. The lesson says "add
patterns before the general ones" in one clause and does not say why; this is
why. Second, greed: the textbook `https?://\S+` scores 5 of 12 whole URLs
because `\S+` eats the sentence-final stop, the comma, the closing bracket and
the closing quote, and the sentence the exercise names for its test is one of
the 5 it gets right -- the URL there is followed by a space. So the test as
written passes on the buggy pattern. Trimming trailing punctuation takes it to
11 of 12; the last one is a balanced parenthesis inside the path, which no
trailing-character rule can decide and a one-line balance pass over the token
stream can. The cost of not fixing this is measured against the lesson's own
pipeline rather than asserted: sharded, `http://` and `https://` collapse to
the same token stream, because `stem_step_1a` and `lemmatize(_, "NOUN")` both
strip the trailing "s" of the shard "https".

Structure: `CORPUS` is 12 sentences each labelled with the URL span a human
would draw, so "kept whole" is measured against ground truth rather than
against another regex. `toks` prepends a candidate alternative to the lesson's
own `WORD_RE.pattern` -- the base pattern is imported, never copied -- `after`
appends it instead, `balanced` is the repair pass, and `score` counts labelled
spans recovered. `shards` runs the two schemes through the lesson's
`preprocess` to price the failure downstream.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "01-text-processing"

GREEDY = r"https?://\S+"
TRIM = r"https?://[^\s<>\"]*[^\s<>\"'.,;:!?)\]}]"
PROBE = "Visit https://example.com today."

CORPUS = (
    (PROBE, "https://example.com"),
    ("Visit https://example.com.", "https://example.com"),
    ("See https://example.com, then stop.", "https://example.com"),
    ("(see https://example.com)", "https://example.com"),
    ("Read https://example.com/a/b?q=1&r=2 now", "https://example.com/a/b?q=1&r=2"),
    ("Mirror: http://example.org/index.html!", "http://example.org/index.html"),
    ("Docs at https://a.io; mirrors elsewhere.", "https://a.io"),
    ('"https://example.com" is the source', "https://example.com"),
    ("Try https://example.com/path_(v2) later", "https://example.com/path_(v2)"),
    ("https://example.com starts the line", "https://example.com"),
    ("End of line https://example.com", "https://example.com"),
    ("Ask [https://example.com] for details.", "https://example.com"),
)

PATTERNS = (("lesson", None), ("greedy", GREEDY), ("trim", TRIM))

toks = lambda ref, url, text: re.findall(                                          # noqa: E731
    (url + "|" if url else "") + ref.WORD_RE.pattern, text)
after = lambda ref, text: re.findall(ref.WORD_RE.pattern + "|" + GREEDY, text)     # noqa: E731
urls = lambda row: [t for t in row if t.startswith("http")]                        # noqa: E731
score = lambda got: sum(gold in row for row, (_, gold) in zip(got, CORPUS))        # noqa: E731
misses = lambda got: [i for i, (row, (_, g)) in enumerate(zip(got, CORPUS), 1) if g not in row]
covers = lambda got: all("".join(row) == re.sub(r"\s", "", text)                   # noqa: E731
                        for row, (text, _) in zip(got, CORPUS))


def balanced(tokens) -> list:
    """Re-attach a ')' that closes a '(' the URL itself opened."""
    out = []
    for token in tokens:
        opened = out and out[-1].startswith("http") and out[-1].count("(") > out[-1].count(")")
        if token == ")" and opened:
            out[-1] += ")"
        else:
            out.append(token)
    return out


def shards(ref) -> dict:
    """What sharding a URL costs in the lesson's own pipeline."""
    runs = {scheme: ref.preprocess(f"{scheme}://example.com") for scheme in ("http", "https")}
    return {key: [runs[s][key] for s in ("http", "https")] for key in ("tokens", "stems", "lemmas")}


def tokenized(ref) -> dict:
    """Every arm's tokens for every sentence, plus the balance-repaired arm."""
    arms = {name: [toks(ref, pattern, text) for text, _ in CORPUS] for name, pattern in PATTERNS}
    arms["repaired"] = [balanced(row) for row in arms["trim"]]
    return arms


def tally(arms) -> dict:
    return {key: {name: fn(rows) for name, rows in arms.items()}
            for key, fn in (("kept", score), ("missed", misses), ("covers", covers))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    arms = tokenized(ref)
    appended = [after(ref, text) for text, _ in CORPUS]
    stops = sum(len(a) != len(b) for a, b in zip(arms["greedy"], arms["repaired"]))
    return dict(tally(arms), probe=arms["repaired"][0], shattered=arms["lesson"][0],
                dead=appended == arms["lesson"], one=len(urls(arms["repaired"][0])),
                greedy_urls=urls(arms["greedy"][1]), paren=urls(arms["trim"][8])[0],
                stops=stops, n=len(CORPUS), pipeline=shards(ref))


def verify(result):
    kept, pipe, n = result["kept"], result["pipeline"], result["n"]
    return [
        practice.Check(
            "ANSWER: the probe sentence yields one URL token, where the lesson yields seven",
            result["one"] == 1 and result["probe"] == ["Visit", "https://example.com", "today", "."],
            f"{PROBE!r} tokenizes to {result['probe']} -- 4 tokens, exactly 1 a URL. The lesson's "
            f"WORD_RE gives {result['shattered']}: {len(result['shattered'])} tokens, 0 URLs, the "
            f"URL itself split across 7"),
        practice.Check(
            "MECHANISM: placement decides it -- appended, the URL alternative never fires",
            result["dead"] and kept["lesson"] == 0,
            f"appending the same {GREEDY!r} after the word rule leaves the output byte-identical to "
            f"the lesson's on all {n} sentences, for {kept['lesson']}/{n} URLs kept whole: `re` "
            f"alternation is ordered, so `[A-Za-z]+` matches 'https' before the URL branch is tried"),
        practice.Check(
            "FINDING: the exercise's own test passes on the standard broken pattern",
            kept["greedy"] < n and 1 not in result["missed"]["greedy"],
            f"{GREEDY!r} keeps {kept['greedy']}/{n} whole and misses sentences "
            f"{result['missed']['greedy']} -- the probe is not among them, because its URL is "
            f"followed by a space. One sentence later `\\S+` returns "
            f"{result['greedy_urls'][0]!r}, stop included, and across the corpus it destroys "
            f"{result['stops']} punctuation boundaries"),
        practice.Check(
            "FINDING: trimming the tail gets all but the balanced paren, which needs a second pass",
            kept["trim"] == n - 1 and result["missed"]["trim"] == [9] and kept["repaired"] == n,
            f"refusing a trailing .,;:!?)]}}\"' takes it to {kept['trim']}/{n}; the miss is "
            f"{result['paren']!r}, a '(' the path itself opened. No trailing-character rule can "
            f"decide that -- ')' closes the URL in sentence 4 and closes nothing in sentence 9 -- "
            f"but re-attaching a ')' only when the URL has an unmatched '(' reaches {n}/{n}"),
        practice.Check(
            "CONTROL: character coverage holds for all four arms, so the obvious invariant ranks none",
            all(result["covers"].values()),
            f"every arm satisfies ''.join(tokens) == the input with whitespace removed: "
            f"{result['covers']} -- the lesson's shattering and greedy's over-eating both preserve "
            f"it, so a round-trip test would call the 0/{n} arm and the {n}/{n} arm equally correct"),
        practice.Check(
            "FINDING: unsharded, http and https differ; sharded, the lesson's pipeline erases that",
            pipe["tokens"][0] != pipe["tokens"][1] and pipe["stems"][0] == pipe["stems"][1]
            and pipe["lemmas"][0] == pipe["lemmas"][1],
            f"as tokens the two schemes differ in one place, {pipe['tokens'][0][0]!r} vs "
            f"{pipe['tokens'][1][0]!r}; as stems and as lemmas both read "
            f"{pipe['stems'][0]} -- stem_step_1a strips the final 's' of the shard 'https', and "
            f"lemmatize(_, 'NOUN') strips it again, so the scheme is erased twice over"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
