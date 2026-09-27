"""Exercise 5 — on a 5% compute budget a 23.7% gate covers 21% of traffic and leaks 79% of harmful outputs; a 1% gate covers all.

    Read Constitutional Classifiers v2 methodology (if available). Explain why
    ~1% compute overhead is a qualitatively different safety story than 23.7%.

Reading of the exercise: overhead matters through what an operator does
with it. With a fixed safety-compute budget b, a gate costing o per request
can inspect min(1, b/o) of traffic; the rest is ungated. That model is run
on the lesson's toy, with the reference `critique` as the output gate over
20,000 seeded base-model responses, at the two overheads the lesson quotes.

**ANSWER: at 23.7% the gate is a sampling decision; at ~1% it is an
invariant.** Always on, a 23.7% gate costs 19.2% of serving capacity
(1 - 1/1.237), a 1% gate 1.0%. On a 1% / 5% / 10% budget the 23.7% gate
covers 4.2% / 21.1% / 42.2% of requests, and in the toy lets through
95.4% / 78.5% / 57.9% of the harmful responses a full gate would stop (92.7%
of base responses carry a harmful token), within a point of 1 - coverage; the 1% gate covers 100% at every
budget and leaks 0. Partial coverage makes defence a lottery an attacker can
retry: at 21.1% coverage the expected tries to reach an ungated request is
1.27. At full coverage the attacker has to beat the classifier itself, and
"no universal jailbreak" becomes a claim about the classifier rather than
about luck.

**FINDING: the lesson misdates v1.** It says "v1 (2023) had 23.7% compute
overhead"; the 23.7% figure is from Sharma et al., "Constitutional
Classifiers" (arXiv:2501.18837, submitted 31 January 2025), which the lesson
does not cite.

Structure: `coverage()` is the budget model; `leak()` gates a seeded sample of
base responses with the reference critique at that coverage.
"""

from __future__ import annotations

import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "05-constitutional-ai-rlaif"
BUDGETS, N = (0.01, 0.05, 0.10), 20000


def coverage(budget, overhead):
    return min(1.0, budget / overhead)


def leak(ref, responses, cover, seed):
    """Share of harmful responses that reach the user when a fraction `cover` is gated."""
    rng = random.Random(seed)
    harmful = [r for r in responses if ref.critique(r, ref.PRINCIPLES[1])]
    passed = sum(1 for _ in harmful if rng.random() >= cover)
    return round(passed / len(harmful), 3), round(len(harmful) / len(responses), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    v1 = re.search(r"v1 \((\d{4})\) had ([\d.]+)% compute overhead", doc)
    v2 = re.search(r"v2 \((\d{4})\) is ~([\d.]+)%", doc)
    old, new = float(v1.group(2)) / 100, float(v2.group(2)) / 100
    saved, ref.random = ref.random, random.Random(9)
    try:
        responses = [ref.base_model_sample() for _ in range(N)]
    finally:
        ref.random = saved
    return {
        "overheads": (old, new), "v1_year": v1.group(1), "cites_paper": "2501.18837" in doc,
        "capacity": [round(1 - 1 / (1 + o), 3) for o in (old, new)],
        "cover": {o: [round(coverage(b, o), 3) for b in BUDGETS] for o in (old, new)},
        "leak": {o: [leak(ref, responses, coverage(b, o), i)[0] for i, b in enumerate(BUDGETS)]
                 for o in (old, new)},
        "harmful_share": leak(ref, responses, 1.0, 0)[1],
        "tries": round(1 / (1 - coverage(0.05, old)), 2),
    }


def verify(result):
    r = result
    old, new = r["overheads"]
    return [
        practice.Check(
            "ANSWER: at 23.7% the gate is a sampling decision; at ~1% it is an invariant",
            ((old, new), r["capacity"], r["harmful_share"], r["tries"]) ==
            ((0.237, 0.01), [0.192, 0.01], 0.927, 1.27)
            and (r["cover"], r["leak"]) == ({old: [0.042, 0.211, 0.422], new: [1.0, 1.0, 1.0]},
                                            {old: [0.954, 0.785, 0.579], new: [0.0, 0.0, 0.0]})
            and all(abs(lk + c - 1) < 0.01 for lk, c in zip(r["leak"][old], r["cover"][old])),
            f"capacity cost always-on {r['capacity']}; coverage at budgets {BUDGETS}: "
            f"{r['cover']}; harmful responses leaked {r['leak']} (of {r['harmful_share']:.1%} "
            f"harmful); tries to an ungated request at 5% budget {r['tries']}",
        ),
        practice.Check(
            "FINDING: the lesson misdates v1",
            r["v1_year"] == "2023" and not r["cites_paper"],
            f"lesson dates v1 to {r['v1_year']}; cites arXiv:2501.18837 (Jan 2025): {r['cites_paper']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
