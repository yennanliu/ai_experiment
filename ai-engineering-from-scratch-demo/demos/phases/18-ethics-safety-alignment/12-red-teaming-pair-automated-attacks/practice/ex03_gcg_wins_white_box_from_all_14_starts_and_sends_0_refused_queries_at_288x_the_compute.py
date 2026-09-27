"""Exercise 3 — GCG wins white-box from all 14 starts and sends 0 refused queries, at 288x the compute.

    Read Chao et al. 2023 Figure 5 (PAIR vs GCG comparison). Describe two scenarios where GCG is preferred despite PAIR's efficiency advantage.

Reading of the exercise: the lesson does not reproduce Figure 5, and no
number from it is quoted here. The comparison comes from the lesson's own
"Why PAIR is efficient" section: GCG needs white-box access and spends
optimizer steps, while PAIR is black-box and query-efficient. Each scenario
is then built as a runnable model on the lesson's code. The toy GCG gets
white-box access to the semantic filter (its term lists, read from source).
Its loss is the number of filter terms a prompt still trips, and it runs
greedy coordinate descent over single-character swaps from a 7-character
vocabulary. Each step applies the swap that lowers the loss most. Its cost is
counted in local loss evaluations. PAIR's cost is counted in target queries,
and in the refusals the defender gets to see.

**ANSWER 1: a white-box worst-case evaluation, where PAIR's attacker cannot
find a way in.** Started from each of the reference's 14 distinct templates,
the toy GCG reaches loss 0, and all 14 of its outputs pass the semantic
filter. Under the same filter, PAIR with paraphrase and with roleplay spends
its whole 20-query budget, is refused 20 times, and fails. PAIR's result is
bounded by what its attacker thinks to say. GCG's result is bounded only by
the filter. For a pre-release open-weights evaluation that wants a lower
bound on robustness, that difference decides the choice. (The white-box loss
is faithful: on all 14 templates, loss > 0 exactly when the target refuses.)

**ANSWER 2: a monitored or rate-limited target with a local copy.** GCG does
its whole search offline and sends one query, which is not refused: 0
refusals on either filter, and its strings also pass the keyword filter 14
of 14 times. PAIR shows the defender every failed attempt: 2 refusals before
paraphrase wins on the keyword filter, and 20 per failed strategy on the
semantic filter. When refusals trigger alerts or bans, and open weights let
the search run locally, the white-box attack is also the quiet one.

**FINDING: the price is compute, 288x PAIR's slowest win.** GCG makes a mean
of 2.14 swaps and 864 loss evaluations per start. PAIR's slowest winning run
on either filter takes 3 queries. That is the lesson's "orders of magnitude"
gap, although here GCG's evaluations are local and free of refusals.

Structure: `terms()` is the white-box read; `gcg()` is the greedy coordinate
search; `refusals()` counts what a PAIR run shows the target.
"""

from __future__ import annotations

import ast
import inspect

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "12-red-teaming-pair-automated-attacks"
STRATEGIES = ("paraphrase", "roleplay", "encoded")
SUBS = "013457@"                        # the toy GCG's token vocabulary: one char swaps
REFUSAL = "I cannot help with that."


def terms(ref):
    """White-box access: every term list the semantic filter checks, read from its source."""
    tree = ast.parse(inspect.getsource(ref.semantic_filter_target))
    lists = [ast.literal_eval(n.value) for n in ast.walk(tree)
             if isinstance(n, ast.Assign) and isinstance(n.value, ast.List)]
    return [*ref.FORBIDDEN_TERMS, *(t for lst in lists for t in lst)]


def loss(vocab, prompt):
    """The gradient stand-in: how many filter terms the prompt still trips."""
    low = prompt.lower()
    return sum(t in low for t in vocab)


def gcg(vocab, prompt, steps=12):
    """Greedy coordinate search: apply the single swap that lowers the loss most, repeat.

    Returns (final prompt, swaps made, local loss evaluations). No target queries.
    """
    evals = 0
    for swaps in range(steps):
        if loss(vocab, prompt) == 0:
            return prompt, swaps, evals
        cands = [prompt[:i] + c + prompt[i + 1:] for i in range(len(prompt)) for c in SUBS]
        evals += len(cands)
        prompt = min(cands, key=lambda p: loss(vocab, p))
    return prompt, steps, evals


def refusals(ref, target, name):
    """(refused queries a PAIR run shows the defender, success, queries)."""
    succ, k, hist = ref.pair_loop(target, getattr(ref, f"attacker_{name}"))
    return sum(r == REFUSAL for _, r in hist), succ, k


def gcg_runs(ref, vocab):
    """GCG from every distinct reference template: its outputs, mean swaps, mean evaluations."""
    pool = list(dict.fromkeys(getattr(ref, f"attacker_{s}")([None] * j)
                              for s in STRATEGIES for j in range(5)))
    runs = [gcg(vocab, p) for p in pool]
    mean = [round(sum(r[i] for r in runs) / len(runs), 2) for i in (1, 2)]
    return pool, [p for p, _, _ in runs], *mean


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    vocab, sem, kw = terms(ref), ref.semantic_filter_target, ref.keyword_filter_target
    pool, found, swaps, evals = gcg_runs(ref, vocab)
    pair = {t: {s: refusals(ref, fn, s) for s in STRATEGIES}
            for t, fn in (("keyword", kw), ("semantic", sem))}
    worst = max(k for runs in pair.values() for _, succ, k in runs.values() if succ)
    return {
        "faithful": all((loss(vocab, p) > 0) == (sem(p) == REFUSAL) for p in pool),
        "pool": len(pool),
        "gcg_sem": sum(sem(p) != REFUSAL for p in found),
        "gcg_kw": sum(kw(p) != REFUSAL for p in found),
        "swaps": swaps, "evals": evals, "example": found[0],
        "pair": pair, "pair_worst": worst, "ratio": round(evals / worst),
    }


def verify(result):
    pair = result["pair"]
    return [
        practice.Check(
            "ANSWER 1: white-box worst case -- GCG breaks the semantic filter from every start",
            (result["faithful"], result["pool"], result["gcg_sem"]) == (True, 14, 14)
            and [pair["semantic"][s] for s in STRATEGIES[:2]] == [(20, False, 20)] * 2,
            f"GCG from all {result['pool']} templates passes {result['gcg_sem']}; PAIR "
            f"(refusals, success, queries) on semantic: {pair['semantic']}",
        ),
        practice.Check(
            "ANSWER 2: monitored target -- GCG searches offline and sends one clean query",
            (pair["keyword"]["paraphrase"], result["gcg_sem"], result["gcg_kw"])
            == ((2, True, 3), 14, 14),
            f"PAIR (refusals, success, queries): {pair}; GCG: 0 refusals, 1 query, and its "
            f"strings pass the keyword filter {result['gcg_kw']}/{result['pool']}",
        ),
        practice.Check(
            "FINDING: GCG pays in local compute, 288x PAIR's slowest winning run",
            [result[k] for k in ("evals", "swaps", "pair_worst", "ratio")]
            == [864, 2.14, 3, 288],
            f"mean {result['swaps']} swaps and {result['evals']:.0f} loss evaluations per start, "
            "vs "
            f"{result['pair_worst']} queries for PAIR's slowest win; "
            f"e.g. {result['example']!r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
