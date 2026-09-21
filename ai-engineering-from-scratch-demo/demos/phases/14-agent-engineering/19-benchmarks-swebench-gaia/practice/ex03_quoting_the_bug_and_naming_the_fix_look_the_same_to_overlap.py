"""Exercise 3 — quoting the bug and naming the fix look the same to overlap.

    Read the SWE-bench+ paper (Aleithan et al., Oct 2024). Implement their
    solution-leakage check: pattern-match the issue text against the diff.

Reading of the exercise: the paper's finding is that 32.67% of successful
patches had the solution present in the issue text. Implementing the check
means deciding what "present" means, and the deciding detail is that a good
bug report *quotes the broken code*. Tokens on a diff's removed lines are
therefore expected in the issue text; only tokens that appear on **added**
lines and nowhere in the pre-image are evidence that the reporter wrote the
fix.

**ANSWER: the added-only check flags 4 of 12 issues -- 33.3%, against the
paper's 32.67%.** It scores every identifier and literal introduced by the
diff, flags an issue when the text names at least one of them, and agrees
with the hand-labelled ground truth on **12** of **12**: precision **1.00**,
recall **1.00**.

**FINDING: the naive check reproduces nothing.** Matching the issue text
against the whole diff, removed lines included, flags **10** of **12** --
**83.3%** -- at precision **0.40**. All **6** false positives are reports
that quoted the failing call, which is what a good report does. One line of
filtering separates a 33.3% headline from an 83.3% one, and only the first
is a measurement of anything.

**FINDING: leakage is invisible to the harness by construction.** Replacing
every task's `description` with its own patch -- total leakage -- leaves all
**12** verdicts byte-identical, because `run_task` reads `description` **0**
times. Contamination cannot be measured by the thing being contaminated; it
needs a second pass over the corpus.

**FINDING: the other 31.08% needs no text at all.** SWE-bench+ also found
weakly-covered tasks. A task with **1** FAIL_TO_PASS test and **0**
PASS_TO_PASS tests is resolved by a patch that satisfies that one test and
corrupts everything else, because `ptp_broke` over an empty list is **0**.
The shipped harness returns `resolved=True` for exactly that patch.

Structure: `introduced()` is the added-only token set; `flag()` is the check;
`ISSUES` carries the hand labels it is scored against.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "19-benchmarks-swebench-gaia"
TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")
STOP = frozenset("""if is in the a an and or not to of for def return self value values
result none true false it this that when then with""".split())

# (issue text, diff, leaks) -- `leaks` is the hand label the checks are scored on.
ISSUES = (
    ("timeout is ignored; pass timeout=self.timeout to urlopen",
     "- resp = urlopen(url)\n+ resp = urlopen(url, timeout=self.timeout)", True),
    ("mean(xs) raises ZeroDivisionError on total / len(xs) for empty xs",
     "- return total / len(xs)\n+ return total / len(xs) if xs else 0.0", False),
    ("the fix is to use math.isclose instead of == on floats",
     "- if a == b:\n+ if math.isclose(a, b):", True),
    ("parse_date fails for '2024-02-30' and returns a bogus datetime",
     "- return datetime(y, m, d)\n+ return _safe_date(y, m, d)", False),
    ("just add a retries=3 keyword to fetch and loop on ConnectionError",
     "- def fetch(url):\n+ def fetch(url, retries=3):", True),
    ("sorting is unstable; sorted(rows) gives different orders across runs",
     "- return sorted(rows)\n+ return sorted(rows, key=_row_key)", False),
    ("normalize_headers calls name.lower() and mangles Content-Type",
     "- name.lower()\n+ name.title()", False),
    ("cache never expires -- I think expires_at should be compared to now()",
     "- if key in cache:\n+ if key in cache and cache[key].expires_at > now():", True),
    ("output is polluted during test runs and should not be",
     "- print(msg)\n+ logger.debug(msg)", False),
    ("ratio(num, den) raises ZeroDivisionError when den is 0",
     "- return num / den\n+ return num / den if den else float('inf')", False),
    ("slice is off by one: items[:n] drops the last element",
     "- return items[:n]\n+ return items[:n + 1]", False),
    ("reading the config file fails with UnicodeDecodeError on Windows",
     "- open(path)\n+ open(path, encoding='utf-8')", False),
)


def tokens(text):
    return {t.lower() for t in TOKEN.findall(text)} - STOP


def side(diff, added):
    return tokens("\n".join(line for line in diff.split("\n")
                            if line.startswith("+") is added))


def introduced(diff):
    """Tokens the patch adds and the pre-image never had."""
    return side(diff, True) - side(diff, False)


def flag(issue, diff, added_only=True):
    signal = introduced(diff) if added_only else tokens(diff)
    return bool(tokens(issue) & signal)


def rates(calls, hits):
    true_pos = sum(p and t for p, t in calls)
    return {"precision": round(true_pos / hits, 2) if hits else 0.0,
            "recall": round(true_pos / sum(t for _, t in calls), 2),
            "agree": sum(p == t for p, t in calls),
            "false_pos": [i[:28] for (i, _, t), (p, _) in zip(ISSUES, calls) if p > t]}


def score(added_only):
    calls = [(flag(issue, diff, added_only), leaks) for issue, diff, leaks in ISSUES]
    hits = sum(p for p, _ in calls)
    return {"flagged": hits, "rate": round(100 * hits / len(ISSUES), 1),
            **rates(calls, hits)}


def build_task(ref, description, ptp):
    return ref.Task(tid="t", description=description, state_before={"ok": 0},
                    patch=lambda state: {"ok": 1}, pass_to_pass=ptp,
                    fail_to_pass=[("t", lambda state: state["ok"] == 1)])


def leak_blind(ref):
    """Verdicts with honest descriptions, and with the patch pasted in as the text."""
    green = [("p", lambda state: True)]
    honest = [ref.run_task(build_task(ref, issue, green)).resolved
              for issue, _, _ in ISSUES]
    leaked = [ref.run_task(build_task(ref, diff, green)).resolved
              for _, diff, _ in ISSUES]
    return {"honest": honest, "leaked": leaked, "identical": honest == leaked}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    weak = ref.run_task(build_task(ref, "ratio() divides by zero", []))
    return {"total": len(ISSUES), "added_only": score(True), "paper_rate": 32.67,
            "labelled": sum(leaks for _, _, leaks in ISSUES),
            "whole_diff": score(False), "blind": leak_blind(ref),
            "weak": {"resolved": weak.resolved, "ptp_total": weak.ptp_total,
                     "ftp": (weak.ftp_passed, weak.ftp_total)}}


def verify(result):
    tight, naive, blind = result["added_only"], result["whole_diff"], result["blind"]
    return [
        practice.Check(
            "ANSWER: the added-only check flags 4 of 12 -- 33.3% against 32.67%",
            all([tight["flagged"] == 4, tight["rate"] == 33.3, tight["agree"] == 12,
                 tight["precision"] == 1.0, tight["recall"] == 1.0,
                 result["labelled"] == 4]),
            f"scoring only the identifiers the diff introduces flags "
            f"{tight['flagged']}/{result['total']} -- {tight['rate']}% against the paper's "
            f"{result['paper_rate']}% -- agreeing with the hand labels "
            f"{tight['agree']}/{result['total']}, precision {tight['precision']}",
        ),
        practice.Check(
            "FINDING: the naive check reproduces nothing",
            all([naive["flagged"] == 10, naive["rate"] == 83.3,
                 naive["precision"] == 0.4, len(naive["false_pos"]) == 6]),
            f"matching against the whole diff, removed lines included, flags "
            f"{naive['flagged']}/{result['total']} -- {naive['rate']}% -- at precision "
            f"{naive['precision']}, and all {len(naive['false_pos'])} false positives "
            f"quoted the failing call. One line of filtering separates "
            f"{tight['rate']}% from {naive['rate']}%",
        ),
        practice.Check(
            "FINDING: leakage is invisible to the harness by construction",
            all([blind["identical"] is True, all(blind["honest"]),
                 len(blind["leaked"]) == 12]),
            f"replacing every description with its own patch leaves all "
            f"{len(blind['leaked'])} verdicts identical ({blind['identical']}): run_task "
            "reads description zero times. Contamination cannot be measured by the thing "
            "being contaminated -- it needs a second pass over the corpus",
        ),
        practice.Check(
            "FINDING: the other 31.08% needs no text at all",
            all([result["weak"]["resolved"] is True,
                 result["weak"]["ptp_total"] == 0,
                 result["weak"]["ftp"] == (1, 1)]),
            f"a task with {result['weak']['ftp'][1]} FAIL_TO_PASS test and "
            f"{result['weak']['ptp_total']} PASS_TO_PASS tests is resolved "
            f"({result['weak']['resolved']}) by a patch that satisfies the one test and "
            "zeroes the rest, because ptp_broke over an empty list is 0",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
