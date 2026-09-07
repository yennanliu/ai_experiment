"""Exercise 3 — receipt pair f1 vs cer.

    **(Hard)** Use PaddleOCR on a set of 20 receipts, extract line items, and compute F1 against hand-labelled ground truth for {item_name, price} pairs.

Reading of the exercise: the first clause is not runnable in this repo and the
second is exactly computable, so this file establishes the first as a measurement
rather than routing around it. `importlib.import_module` raises
ModuleNotFoundError for `paddleocr`, `pytesseract`, `easyocr` and `transformers`
alike -- none is in any dependency group -- and the lesson ships zero image or
PDF files, so the 20 receipts do not exist either; `code/main.py` never mentions
an engine, it is a CRNN trainer. What remains is the scorer, and it is the
interesting half. Against a hand-labelled 20-line corpus corrupted by the classic
confusions (o/0, l/1, i/1, s/5, b/8, z/2) at a 50% per-character rate, pair F1 is
0.2000 while CER is 0.1350: 86.5% of characters are right and 80% of the pairs
score zero, because exact-match pair F1 is the AND of its fields. Price F1 is
1.0000 throughout -- prices are digits and every confusion class maps a letter
*to* a digit -- so pair F1 equals name F1 exactly. WER on the same text is
0.4000, 3.0x the CER, and it is unbounded above: splitting `milk` into four
characters scores 4.0000, which no F1 can. The confusion fold is idempotent but
not injective (36 symbols to 30 classes, 14 ordered pairs merged), which makes it
a legitimate scoring equivalence -- pair F1 1.0000 -- and an illegitimate
post-processor: emitting its output as corrected text scores 0.0000 and destroys
the 4 pairs that were already right. Pure stdlib; nothing is imported from the
lesson's `code/`, which is why this exercise runs at T0.

Structure: `edit` is the stdlib Levenshtein over any sequence, so it serves both
CER (characters) and WER (words); the module-level `f1` lambda is exact-match F1
over multisets, `fold` applies the confusion classes, and `lines`, `names`,
`prices` and `folded` are the four projections of the corpus that the scorers
need; `probe_engines` records what each OCR import actually raises and counts the
lesson's receipt files and engine mentions; `corrupt` applies the confusion map
at a seeded per-character rate; `score` reports pair, name and price F1 with
corpus CER and WER; `fold_probe` measures idempotence, non-injectivity and the
two-sided versus one-sided verdicts; `metric_probe` checks the triangle
inequality and identity over the corpus names and hand-works one line; and the
`absent`, `listing` and `ladder` lambdas format what the checks report. At 146
code lines this sits above D14's 120-line target and 4 clear of the ceiling:
five checks over three corruption rates and 6,840 distance triples.
"""

from __future__ import annotations

import collections
import importlib
import itertools
import random

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "19-ocr-document-understanding"

ENGINES = ("paddleocr", "pytesseract", "easyocr", "transformers")
WORDS = ("paddle", "tesseract", "easyocr", "receipt", "donut", "trocr")
CONFUSE = {"o": "0", "l": "1", "i": "1", "s": "5", "b": "8", "z": "2"}
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
PICTURES = (".png", ".jpg", ".jpeg", ".tif", ".pdf")
RATES, RATE, SEED = (0.3, 0.5, 0.6), 0.5, 19
HAND = ("soda 1.50", "50da 1.5o")            # reference, then a hand-corrupted hypothesis
RECEIPTS = [("soda", "1.50"), ("bread", "3.25"), ("milk", "2.10"), ("soap", "4.99"),
            ("salt", "0.89"), ("oil", "7.40"), ("ice", "1.05"), ("beans", "2.35"),
            ("zucchini", "3.60"), ("lime", "0.55"), ("onion", "1.20"), ("basil", "2.75"),
            ("olives", "5.10"), ("sugar", "1.95"), ("butter", "4.20"), ("lentils", "3.05"),
            ("spice", "6.50"), ("iodine", "8.15"), ("biscuit", "2.60"), ("zest", "0.75")]
f1 = lambda pred, truth: sum((collections.Counter(pred)             # noqa: E731 - exact-match F1
                              & collections.Counter(truth)).values()) / len(truth)
fold = lambda text: "".join(CONFUSE.get(char, char) for char in text)           # noqa: E731
lines = lambda rows: [f"{name} {price}" for name, price in rows]                # noqa: E731
names = lambda rows: [name for name, _ in rows]                                 # noqa: E731
prices = lambda rows: [price for _, price in rows]                              # noqa: E731
folded = lambda rows: [(fold(name), fold(price)) for name, price in rows]       # noqa: E731
absent = lambda raised: all(why.startswith("ModuleNotFoundError") for why in raised.values())  # noqa: E731
listing = lambda raised: "; ".join(f"{name} -> {why}" for name, why in raised.items())  # noqa: E731
ladder = lambda table: ", ".join(f"{r:.0%}->{table[r]['pair']:.4f}" for r in RATES)     # noqa: E731


def edit(hyp, ref) -> int:
    prev = list(range(len(ref) + 1))
    for row, left in enumerate(hyp, 1):
        cur = [row]
        for col, right in enumerate(ref, 1):
            cur.append(min(prev[col] + 1, cur[-1] + 1, prev[col - 1] + (left != right)))
        prev = cur
    return prev[-1]


def probe_engines() -> dict:
    raised = {}
    for name in ENGINES:
        try:
            raised[name] = "imported" if importlib.import_module(name) else "empty module"
        except ModuleNotFoundError as exc:
            raised[name] = f"{type(exc).__name__}: {exc}"
    lesson = parity.lesson_dir(PHASE, LESSON)
    source = (lesson / "code" / "main.py").read_text(encoding="utf-8").lower()
    return {"raised": raised, "mentions": sum(source.count(word) for word in WORDS),
            "images": sum(1 for path in lesson.rglob("*") if path.suffix.lower() in PICTURES)}


def corrupt(text, rng, rate) -> str:
    return "".join(CONFUSE[char] if char in CONFUSE and rng.random() < rate else char for char in text)


def score(hyp, truth) -> dict:
    hyp_lines, refs = lines(hyp), lines(truth)
    return {"pair": f1(hyp, truth), "name": f1(names(hyp), names(truth)),
            "price": f1(prices(hyp), prices(truth)),
            "cer": sum(map(edit, hyp_lines, refs)) / sum(map(len, refs)),
            "wer": sum(edit(got.split(), want.split()) for got, want in zip(hyp_lines, refs))
                   / (2 * len(truth))}


def fold_probe(hyp) -> dict:
    return {"idempotent": all(fold(fold(text)) == fold(text) for row in RECEIPTS + hyp for text in row),
            "merged": sum(1 for one, two in itertools.permutations(ALPHABET, 2) if fold(one) == fold(two)),
            "both": f1(folded(hyp), folded(RECEIPTS)), "one_sided": f1(folded(hyp), RECEIPTS),
            "classes": len({fold(char) for char in ALPHABET}), "symbols": len(ALPHABET),
            "already": sum(map(tuple.__eq__, hyp, RECEIPTS)),
            "kept": sum(map(tuple.__eq__, folded(hyp), RECEIPTS))}


def metric_probe() -> dict:
    items, triples = names(RECEIPTS), list(itertools.permutations(names(RECEIPTS), 3))
    return {"triples": len(triples), "split": edit(list("milk"), ["milk"]),
            "violations": sum(1 for one, two, tri in triples if edit(one, tri) > edit(one, two) + edit(two, tri)),
            "reflexive": all(edit(item, item) == 0 for item in items),
            "hand": (edit(HAND[1], HAND[0]), len(HAND[0]), edit(HAND[1].split(), HAND[0].split()))}


def solve():
    hyps = {rate: [(corrupt(name, rng, rate), corrupt(price, rng, rate))
                   for rng in [random.Random(SEED)] for name, price in RECEIPTS] for rate in RATES}
    return {"engines": probe_engines(), "scores": {r: score(hyps[r], RECEIPTS) for r in RATES},
            "fold": fold_probe(hyps[RATE]), "metric": metric_probe(),
            "sample": hyps[RATE][2:4], "truth": RECEIPTS[2:4], "count": len(RECEIPTS)}


def verify(result):
    eng, main = result["engines"], result["scores"][RATE]
    fold_r, met, count = result["fold"], result["metric"], result["count"]
    return [
        practice.Check(
            "ANSWER: PaddleOCR is not installed and the 20 receipts do not exist",
            absent(eng["raised"]),
            f"`importlib.import_module` on all {len(ENGINES)} candidate engines: {listing(eng['raised'])}. "
            f"The lesson tree holds {eng['images']} image or PDF files and `code/main.py` mentions an "
            f"engine or receipt {eng['mentions']} times -- it trains a CRNN. So the corpus is "
            f"hand-labelled here: {count} single-item receipts"),
        practice.Check(
            "ANSWER: the same 20 lines score pair F1 0.2000, CER 0.1350 and WER 0.4000",
            main["pair"] == 0.2 and main["cer"] < 0.15 and met["split"] > 1,
            f"the confusion map {CONFUSE} at a {RATE:.0%} per-character rate, seed {SEED}: pair F1 "
            f"{main['pair']:.4f} ({int(main['pair'] * count)}/{count} exact pairs) against CER "
            f"{main['cer']:.4f} -- {1 - main['cer']:.1%} of characters correct -- and WER "
            f"{main['wer']:.4f}, {main['wer'] / main['cer']:.1f}x the CER. WER is unbounded above 1.0 "
            f"where F1 is not: 'milk' split into {met['split']} characters costs {met['split']} word edits "
            f"over 1 reference word, WER {float(met['split']):.4f}. Sample {result['sample']}"),
        practice.Check(
            "MECHANISM: exact-match pair F1 is the AND of its fields, so it equals the weaker one",
            main["pair"] == main["name"] and main["price"] == 1.0,
            f"name F1 {main['name']:.4f}, price F1 {main['price']:.4f}, pair F1 {main['pair']:.4f}: prices "
            f"are digits and '.', and every confusion class maps a letter *to* a digit, so no price is ever "
            f"touched and the pair score collapses onto the name. Rate ladder {ladder(result['scores'])}; "
            f"the hand line {HAND} gives CER {met['hand'][0]}/{met['hand'][1]} = "
            f"{met['hand'][0] / met['hand'][1]:.4f} but WER {met['hand'][2]}/2 = {met['hand'][2] / 2:.4f}"),
        practice.Check(
            "CONTROL: the fold is idempotent but not injective — a scorer, never a post-processor",
            fold_r["idempotent"] and fold_r["classes"] < fold_r["symbols"],
            f"`fold(fold(x)) == fold(x)` on all {2 * count} corpus strings, and it collapses "
            f"{fold_r['symbols']} symbols to {fold_r['classes']} classes, making {fold_r['merged']} "
            f"ordered symbol pairs equal. Applied to both sides it scores pair F1 {fold_r['both']:.4f}; "
            f"emitted as corrected text it scores {fold_r['one_sided']:.4f} and takes the "
            f"{fold_r['already']} already-correct pairs down to {fold_r['kept']}"),
        practice.Check(
            "CONTROL: the distance underneath every rate above is a metric on this corpus",
            met["violations"] == 0 and met["reflexive"],
            f"over the {count} item names, {met['violations']} of {met['triples']:,} ordered triples "
            f"violate edit(a,c) <= edit(a,b) + edit(b,c) and every name has edit(a,a) == 0, so CER and "
            f"WER are normalised metrics and the gap between them above is a real gap"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
