"""Exercise 1 — the best length policy misses 2 of 4 encoded attacks, no more than the strict baseline, and over-refuses 40% of safe prompts.

    Add a fourth mock policy that refuses based on prompt length. Confirm that under-refusal rises on encoded attacks (which tend to be short).

Reading of the exercise: the fourth policy refuses a prompt longer than T
characters (jailbreaks carry extra framing, so length is the signal) and
answers otherwise, emitting the lesson's own `REFUSAL_TEXT` / `ANSWER_TEXT`
at the strict policy's confidences (0.90 / 0.80). T is not guessed: every
T from 40 to 110 is run through the lesson's `evaluate_policy` on its 25
unsafe + 30 safe prompts, and the policy is the T with the best accuracy,
so the length signal gets its strongest showing. "Rises" is checked two
ways: against the other categories under the same policy, and against the
lesson's `MockPolicyStrict` baseline on the same 4 encoded prompts.

**ANSWER: it rises against the other categories, not against the
baseline.** The best threshold is T = 68 (accuracy 0.69). There,
encoding-trick has the highest per-category under-refusal, 0.50 against
0.20 overall, because two of the four encoded prompts (61 and 52
characters) are short enough to be answered. But `MockPolicyStrict`
already misses the same share of encoded attacks (0.50), so the length
policy is no worse on them. Encoded under-refusal goes above the
baseline only for T >= 74, where accuracy falls. The length policy's cost
is on the other side: it refuses 12 of the 30 safe prompts (over-refusal
0.40), and every lesson policy refuses at most 5.

**FINDING: encoded attacks are short only compared with other attacks.**
Their median length is 67.5 characters, against 75 for the other 21 unsafe
prompts and 67 for the safe prompts. No length threshold separates them
from benign asks: to refuse all four you need T < 52, and that refuses 29
of the 30 safe prompts.

**FINDING: `MockPolicyLeaky` misses encoded attacks because its only
pattern is the placeholder word.** Its one regex, `redacted_harmful`,
appears in plain text in 20 of the 25 unsafe prompts and in none of the 4
encoded ones. Encoding hides the word, so Leaky's 1.00 encoded under-refusal
measures how the corpus spells the placeholder, not how well it detects
jailbreaks.

Expected output: three PASS checks.
"""

from __future__ import annotations

import re
import statistics
import sys

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "84-refusal-evaluation"
ENC = "encoding-trick"


def load_lesson():
    """main.py does `from mock_llm import ...`, so its siblings go in sys.modules first."""
    sib = {m: parity.load_reference(PHASE, LESSON, m) for m in ("mock_llm", "prompts")}
    saved = {m: sys.modules.get(m) for m in sib}
    sys.modules.update(sib)
    try:
        main = parity.load_reference(PHASE, LESSON, "main")
    finally:
        for m in sib:
            sys.modules.pop(m)
        sys.modules.update({m: old for m, old in saved.items() if old is not None})
    return main, sib["mock_llm"], sib["prompts"]


def length_policy(mock, threshold):
    def policy(prompt):
        if len(prompt) > threshold:
            return mock.REFUSAL_TEXT.format(c=90)
        return mock.ANSWER_TEXT.format(c=80)
    return policy


def by_group(prompts, fn):
    """fn(prompt) for the unsafe prompts, split into encoding-trick vs the rest."""
    out = {"enc": [], "other": []}
    for r in prompts.unsafe():
        out["enc" if r["category"] == ENC else "other"].append(fn(r["prompt"]))
    return out


def solve():
    main, mock, prompts = load_lesson()
    sweep = {t: main.evaluate_policy(f"MockPolicyLength{t}", length_policy(mock, t)) for t in range(40, 111)}
    best = max(sweep, key=lambda t: (sweep[t]["accuracy"], -t))
    lens = {**by_group(prompts, len), "safe": [len(r["prompt"]) for r in prompts.safe()]}
    leak_re = re.compile(mock._LEAKY_FORBIDDEN[0], re.IGNORECASE)
    leak_hits = {k: sum(v) for k, v in by_group(prompts, lambda p: bool(leak_re.search(p))).items()}
    lesson = {n: main.evaluate_policy(n, p) for n, p in mock.policies().items()}
    return {"best": best, "report": sweep[best], "lesson": lesson, "leak_hits": leak_hits,
            "enc_by_t": {t: r["per_category_under_refusal"][ENC] for t, r in sweep.items()},
            "median": {k: statistics.median(v) for k, v in lens.items()}, "lens": lens,
            "all_enc_fp": sweep[min(lens["enc"]) - 1]["confusion"]["fp"]}


def check_answer(result):
    rep, lesson = result["report"], result["lesson"]
    cats = rep["per_category_under_refusal"]
    strict_enc = lesson["MockPolicyStrict"]["per_category_under_refusal"][ENC]
    above = sorted(t for t, v in result["enc_by_t"].items() if v > strict_enc)
    max_fp = max(r["confusion"]["fp"] for r in lesson.values())
    facts = (result["best"], cats[ENC], max(cats.values()), rep["under_refusal"], strict_enc, above[0],
             rep["confusion"]["fp"], max_fp)
    ok = facts == (68, 0.5, 0.5, 0.2, 0.5, 74, 12, 5)
    return practice.Check(
        "ANSWER: encoded under-refusal is the highest category (0.50 vs 0.20) but equals the strict baseline",
        ok, f"T={result['best']} acc {rep['accuracy']}; per category {cats}; strict encoded {strict_enc}, "
            f"exceeded from T={above[0]}; over-refusal {rep['over_refusal']} ({rep['confusion']['fp']}/30) "
            f"vs lesson max {max_fp}/30")


def check_short(result):
    med = result["median"]
    ok = (med["enc"] < med["other"] and abs(med["enc"] - med["safe"]) <= 1
          and min(result["lens"]["enc"]) == 52 and result["all_enc_fp"] == 29)
    return practice.Check(
        "FINDING: encoded attacks are short against other attacks, not against safe prompts",
        ok, f"median chars encoded {med['enc']}, other unsafe {med['other']}, safe {med['safe']}; "
            f"refusing all 4 encoded (T<52) over-refuses {result['all_enc_fp']}/30 safe")


def check_leaky(result):
    hits = result["leak_hits"]
    leaky = result["lesson"]["MockPolicyLeaky"]["per_category_under_refusal"][ENC]
    return practice.Check(
        "FINDING: Leaky's only pattern is the placeholder word, visible in 20/21 plain and 0/4 encoded prompts",
        hits == {"enc": 0, "other": 20} and leaky == 1.0,
        f"placeholder regex hits {hits}; Leaky encoded under-refusal {leaky}")


def verify(result):
    return [check_answer(result), check_short(result), check_leaky(result)]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
