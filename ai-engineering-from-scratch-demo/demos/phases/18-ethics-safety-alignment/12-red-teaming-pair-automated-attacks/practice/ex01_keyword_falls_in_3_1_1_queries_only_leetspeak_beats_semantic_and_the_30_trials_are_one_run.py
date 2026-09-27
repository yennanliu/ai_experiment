"""Exercise 1 — the keyword filter falls in 3 / 1 / 1 queries, only leetspeak beats the semantic filter, and the 30 trials are one run.

    Run `code/main.py`. Measure mean-queries-to-success for the three built-in
    attacker strategies. Explain which target-defense assumption each exploits.

Reading of the exercise: "measure" means reading the table `main()` prints for
both targets, and then checking what the table is made of. "Which assumption
each exploits" is answered by taking each strategy's templates apart. For
every template the solution records which filter clause refuses it, and it
tests whether the strategy's trick (paraphrase, framing, leetspeak) is what
got the template through.

**ANSWER: against the keyword filter, paraphrase needs 3 queries and roleplay
and encoded need 1 each (ASR 1.0 for all three).** Against the semantic
filter only encoded succeeds, in 1 query. Paraphrase and roleplay print
mean-queries 20.0 there, but 20 is the budget cap on runs that failed (ASR
0.0), so their mean-queries-to-success is undefined, not 20.

**FINDING: the 30 trials are one run, and no strategy reads the target's
reply.** Every attacker indexes its template list by `len(history)`. A
history full of refusals gives the same next prompt as a history full of
compliance, so the loop has no in-context feedback, which is the mechanism
the lesson credits for PAIR's efficiency. Nothing after `random.seed(31)`
calls `random`, and 30 repeated runs produce 1 distinct outcome. The printed
ASRs are 30 copies of one deterministic run.

**FINDING: the lesson says "~5-15 iterations"; the code takes 1-3.**

**FINDING: against the keyword filter each strategy wins on vocabulary, not
on its trick.** The keyword filter assumes its 4-term list covers every way
to ask. Paraphrase exploits that gap directly: 1 of its 5 templates avoids
all four terms, and that template is query 3. Roleplay wins because 4 of its
5 templates happen to contain no listed term; the keyword filter never looks
at framing. Encoded passes 4 of 4, but with the leetspeak undone it still
passes 4 of 4. Only against the semantic filter does the trick carry weight:
de-leeted, the encoded templates pass 0 of 4 instead of 3. The assumption
encoding breaks is that matching exact substrings is enough.

**FINDING: the semantic filter stops roleplay by its verbs, never by its cue
list.** All 5 roleplay templates already contain a risk verb or a forbidden
term. The roleplay-cue clause decides none of them, so on this attacker pool
it is dead code.

Structure: `printed()` parses `main()`'s table; `pool()` enumerates a
strategy's templates through the reference attacker itself;
`local_lists()` reads the semantic filter's term lists from its source.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "12-red-teaming-pair-automated-attacks"
STRATEGIES = ("paraphrase", "roleplay", "encoded")
ROW = r"attacker=(\w+)\s+ASR=([\d.]+)\s+mean-queries=([\d.]+)"
DELEET = str.maketrans("0345", "oeas")
REFUSAL = "I cannot help with that."


def printed(ref):
    """main()'s own table: {target: {attacker: (ASR, mean-queries)}}."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.main()
    blocks = re.split(r"-- target: ([\w-]+) --", out.getvalue())[1:]
    return {t: {a: (float(r), float(q)) for a, r, q in re.findall(ROW, body)}
            for t, body in zip(blocks[::2], blocks[1::2])}


def pool(ref, name):
    """Every distinct template a strategy emits: the attacker reads only len(history)."""
    fn = getattr(ref, f"attacker_{name}")
    return list(dict.fromkeys(fn([None] * j) for j in range(20)))


def passing(ref, prompts):  # [keyword, semantic] count of prompts let through
    targets = (ref.keyword_filter_target, ref.semantic_filter_target)
    return [sum(t(p) != REFUSAL for p in prompts) for t in targets]


def local_lists(fn):
    """A function's local list literals, read from its source: {name: [terms]}."""
    tree = ast.parse(inspect.getsource(fn))
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.List)}


def roleplay_refusals(ref):
    """(roleplay templates with a risk verb or forbidden term, ones only the cue list catches)."""
    lists = local_lists(ref.semantic_filter_target)
    verbs, cues = ref.FORBIDDEN_TERMS + lists["risk_adj"], lists["roleplay_cues"]
    tripped = [(any(t in p.lower() for t in verbs), any(c in p.lower() for c in cues))
               for p in pool(ref, "roleplay")]
    return sum(v for v, _ in tripped), sum(c and not v for v, c in tripped)


def feedback_blind(ref, name):
    """Same prompt whether the history says 'refused' or 'complied'."""
    fn = getattr(ref, f"attacker_{name}")
    return all(fn([("p", REFUSAL)] * j) == fn([("p", "ok")] * j) for j in range(10))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pools = {s: pool(ref, s) for s in STRATEGIES}
    claim = re.search(r"succeed in (~[\d-]+) iterations", parity.doc_text(PHASE, LESSON))
    return {
        "table": printed(ref),
        "passes": {s: [len(p), *passing(ref, p)] for s, p in pools.items()},
        "roleplay": roleplay_refusals(ref),
        "deleet": passing(ref, [p.translate(DELEET) for p in pools["encoded"]]),
        "blind": [feedback_blind(ref, s) for s in STRATEGIES],
        "outcomes": len({repr(ref.pair_loop(ref.semantic_filter_target,
                                            ref.attacker_paraphrase)) for _ in range(30)}),
        "uses_random": "random." in inspect.getsource(ref).split("random.seed(31)")[1],
        "claim": claim.group(1), "terms": ref.FORBIDDEN_TERMS,
    }


def verify(result):
    table, passes, claim = result["table"], result["passes"], result["claim"]
    kw, sem = table["keyword-filter"], table["semantic-filter"]
    return [
        practice.Check(
            "ANSWER: 3 / 1 / 1 queries on the keyword filter; only encoded beats semantic, in 1",
            kw == {"paraphrase": (1.0, 3.0), "roleplay": (1.0, 1.0), "encoded": (1.0, 1.0)}
            and sem == {"paraphrase": (0.0, 20.0), "roleplay": (0.0, 20.0),
                        "encoded": (1.0, 1.0)},
            f"main() prints (ASR, mean-queries): {table}; the 20s are the budget cap",
        ),
        practice.Check(
            "FINDING: the 30 trials are one run, and no strategy reads the target's reply",
            result["blind"] == [True] * 3 and result["outcomes"] == 1
            and not result["uses_random"],
            f"attackers blind to responses: {result['blind']}; distinct outcomes over 30 "
            f"runs: {result['outcomes']}; random used after seed(31): {result['uses_random']}",
        ),
        practice.Check(
            f"FINDING: the lesson says {claim} iterations; the code takes 1-3",
            claim == "~5-15" and sorted({q for _, q in kw.values()}) == [1.0, 3.0],
            f"lesson: {claim}; keyword-filter (ASR, mean-queries) {kw}",
        ),
        practice.Check(
            "FINDING: against the keyword filter each strategy wins on vocabulary, not its trick",
            passes == {"paraphrase": [5, 1, 0], "roleplay": [5, 4, 0], "encoded": [4, 4, 3]}
            and result["deleet"] == [4, 0] and len(result["terms"]) == 4,
            f"[templates, passing keyword, passing semantic]: {passes}; encoded with "
            f"leetspeak undone passes {result['deleet']}; keyword list {result['terms']}",
        ),
        practice.Check(
            "FINDING: the semantic filter stops roleplay by its verbs, never by its cue list",
            result["roleplay"] == (5, 0),
            f"roleplay (with a risk verb or forbidden term, cue-only): {result['roleplay']} of 5",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
