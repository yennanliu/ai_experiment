"""Exercise 2 — 18 rules guard one true positive each, and leave-one-out credits 13 of 31 to no rule.

    Compute per-rule contribution: for each rule, count how many true positives would be lost if it were removed. Sort rules by marginal contribution.

Reading of the exercise: leave-one-out over the lesson's 53 rules. For each
rule, the lesson's `Detector` is rebuilt from `all_rules()` without it and
run over the 50 taxonomy fixtures from lesson 82 and the 25 benign prompts.
A true positive is a fixture whose verdict category equals its label. Each
rule gets the TPs lost, the TPs gained (removal can change which category
wins the argmax), and the wrong-category labels removed. Rules are sorted
by net marginal contribution (lost minus gained), ties broken by the
wrong labels they cause, then by name.

**ANSWER: 18 rules each protect exactly one true positive, 33 protect
none, and 2 cost one.** The 18 at +1 are single-rule catches such as
`begin-with-bien-sur` (pi-08), `html-comment` (cs-04) and `leet-letters`
(et-03). The two at -1 are `override-claim` and `from-now-on-unchained`:
removing either turns a wrong label into a right one (cs-06 becomes
context-smuggling, mt-06 multi-turn-ramp). The net contributions sum to
16, while the detector has 31 true positives.

**FINDING: leave-one-out credits 13 of the 31 true positives to no rule.**
14 TPs are fired by two or more rules of their own category, so no single
removal loses 13 of them. Removing both rules of each of the 13 two-rule
pairs loses that TP every time, so the zeros are redundancy, not dead
weight. The effect runs the other way too: removing `ignore-prior` or
`ignore-family` alone changes nothing, but removing both lets the CSV
fixture cs-08 fall through to `csv-note-field` and become a true positive.

**FINDING: 5 rules never fire on any of the lesson's 75 prompts.** They are
`ignore-previous`, `disregard-prior`, `sure-here-the`, `morse-instruction`
and `act-as-x`. `disregard` is the paraphrase the lesson's Problem section
names as the one a single regex misses, and no fixture uses it. Four rules
cause a wrong label that dropping them removes: `no-warning` (net 0, labels
io-07 prefix-injection), `leet-letters` (net +1, but labels the role-play
fixture rp-09 an encoding trick), and the two at -1.

Expected output: three PASS checks.
"""

from __future__ import annotations

import sys

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "83-prompt-injection-detector"


def load():
    """main.py does `from rules import ...` and `from benign import ...`; register both for that import."""
    deps = {name: parity.load_reference(PHASE, LESSON, name) for name in ("rules", "benign")}
    saved = {name: sys.modules.get(name) for name in deps}
    sys.modules.update(deps)
    try:
        return deps["rules"], deps["benign"], parity.load_reference(PHASE, LESSON, "main")
    finally:
        for name, module in saved.items():
            sys.modules.pop(name) if module is None else sys.modules.__setitem__(name, module)


def outcomes(main, rules, fixtures, benign):
    """(true-positive fixture ids, wrong-category fires) for one rule set."""
    detector = main.Detector(rules)
    labelled = [(f["id"], f["category"], f["prompt"]) for f in fixtures] + [(f"b{i}", "benign", p) for i, p in enumerate(benign)]
    verdicts = [(pid, truth, detector.analyze(str(prompt)).category) for pid, truth, prompt in labelled]
    return ({pid for pid, truth, got in verdicts if got == truth != "benign"},
            {pid for pid, truth, got in verdicts if got not in ("benign", truth)})


def without(rules, *names):
    return [r for r in rules if r["name"] not in names]


def ablate(main, rules, fixtures, benign):
    """Leave-one-out: TPs lost, TPs gained, wrong fires removed, per rule, sorted by net marginal TPs."""
    base_tp, base_wrong = outcomes(main, rules, fixtures, benign)
    rows = []
    for rule in rules:
        tp, wrong = outcomes(main, without(rules, rule["name"]), fixtures, benign)
        lost, gained = sorted(base_tp - tp), sorted(tp - base_tp)
        rows.append({"name": rule["name"], "category": rule["category"], "lost": lost, "gained": gained,
                     "net": len(lost) - len(gained), "fp_removed": len(base_wrong - wrong)})
    return base_tp, sorted(rows, key=lambda r: (-r["net"], r["fp_removed"], r["name"]))


def own_category_fires(main, rules, fixtures):
    """Per fixture, the fired rules whose category is the fixture's own."""
    detector, category = main.Detector(rules), {r["name"]: r["category"] for r in rules}
    verdicts = {f["id"]: (f["category"], detector.analyze(str(f["prompt"])).fired) for f in fixtures}
    return {pid: [n for n in names if category[n] == truth] for pid, (truth, names) in verdicts.items()}


def fired_anywhere(main, rules, prompts):
    return {name for p in prompts for name in main.Detector(rules).analyze(str(p)).fired}


def pair_losses(main, rules, fixtures, benign, own, base_tp):
    """For each TP backed by exactly two in-category rules, remove both: is the TP then lost?"""
    doubles = {pid: tuple(names) for pid, names in own.items() if pid in base_tp and len(names) == 2}
    return {pid: pid not in outcomes(main, without(rules, *pair), fixtures, benign)[0] for pid, pair in doubles.items()}


def solve():
    rules_mod, benign_mod, main = load()
    fixtures, benign, rules = main.load_taxonomy(), benign_mod.prompts(), rules_mod.all_rules()
    base_tp, rows = ablate(main, rules, fixtures, benign)
    fired = fired_anywhere(main, rules, [f["prompt"] for f in fixtures] + benign)
    own = own_category_fires(main, rules, fixtures)
    ignore_pair = outcomes(main, without(rules, "ignore-prior", "ignore-family"), fixtures, benign)[0] - base_tp
    return {"base_tp": sorted(base_tp), "rows": rows, "n_prompts": len(fixtures) + len(benign),
            "dead": [r["name"] for r in rules if r["name"] not in fired],
            "backed": {pid: len(names) for pid, names in own.items() if pid in base_tp},
            "pairs": pair_losses(main, rules, fixtures, benign, own, base_tp), "ignore_pair_gain": sorted(ignore_pair)}


def check_answer(result):
    rows, net = result["rows"], [r["net"] for r in result["rows"]]
    top = [r["name"] for r in rows if r["net"] == 1]
    return practice.Check(
        "ANSWER: 18 rules each protect exactly one true positive, 33 protect none, and 2 cost one",
        (net.count(1), net.count(0), net.count(-1), sum(net), len(result["base_tp"])) == (18, 33, 2, 16, 31)
        and net == sorted(net, reverse=True),
        f"net marginal TPs +1 x{net.count(1)} ({', '.join(top[:4])}, ...), 0 x{net.count(0)}, -1 x{net.count(-1)} "
        f"({', '.join(r['name'] for r in rows if r['net'] < 0)}); sum {sum(net)} against {len(result['base_tp'])} TPs",
    )


def check_redundancy(result):
    backed, rows = result["backed"], result["rows"]
    orphans = len(result["base_tp"]) - sum(len(r["lost"]) for r in rows)
    return practice.Check(
        "FINDING: leave-one-out credits 13 of 31 true positives to no rule, because two same-category rules back each",
        (orphans, sum(n >= 2 for n in backed.values()), len(result["pairs"])) == (13, 14, 13)
        and all(result["pairs"].values()) and result["ignore_pair_gain"] == ["cs-08"],
        f"{orphans} TPs lost by no single removal; {sum(n >= 2 for n in backed.values())} TPs fired by >=2 in-category rules; "
        f"removing each of the {len(result['pairs'])} two-rule pairs loses its TP; removing ignore-prior + "
        f"ignore-family together gains {result['ignore_pair_gain']} (neither alone does)",
    )


def check_dead_and_costly(result):
    costly = {r["name"]: (r["net"], r["fp_removed"]) for r in result["rows"] if r["fp_removed"]}
    return practice.Check(
        "FINDING: 5 rules never fire on the lesson's 75 prompts, 'disregard' among them, and dropping 4 rules removes a wrong-category label",
        result["dead"] == ["ignore-previous", "disregard-prior", "sure-here-the", "morse-instruction", "act-as-x"]
        and set(costly) == {"no-warning", "leet-letters", "override-claim", "from-now-on-unchained"},
        f"dead on {result['n_prompts']} prompts: {result['dead']}; (net TP, wrong fires removed) if dropped: {costly}",
    )


def verify(result):
    return [check(result) for check in (check_answer, check_redundancy, check_dead_and_costly)]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
