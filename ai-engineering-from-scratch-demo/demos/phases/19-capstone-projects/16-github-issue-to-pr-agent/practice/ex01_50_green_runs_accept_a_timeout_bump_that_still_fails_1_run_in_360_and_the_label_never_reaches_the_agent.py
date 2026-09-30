"""Exercise 1 -- 50 green runs accept a timeout bump that still fails 1 run in 360, and the label never reaches the agent.

    Add a "fix flaky test" mode: the label `@agent stabilize-flake TestX` runs the test 50 times in-sandbox and proposes a minimal change that stabilizes it.

Reading of the exercise: the mode has three parts, each built beside the
lesson's code. (1) Trigger: the label is parsed into a mode and a test name,
and the lesson's TypeScript router (`code/ts/src/router.ts`) is checked for
whether an `issues` delivery with action `labeled` (the action GitHub sends
when a label is added, docs.github.com/en/webhooks/webhook-events-and-payloads#issues,
read 2026-09-29) would ever be dispatched. (2) The 50-run probe: `TestX` is a
seeded stand-in for the classic flake -- it waits a fixed 80 ms for an async
job whose latency is lognormal (median 40 ms, sigma 0.5) -- so it fails when
the job is slow. (3) The minimal change: candidate patches are tried smallest
diff first, each re-run 50 times, and the first one that goes 50/50 green is
proposed; a rerun decorator is refused because it changes the verdict, not
the test. True residual rates come from 20,000 runs of each patch. The
lesson's own flake, `run_verify`, is measured the same way.

**ANSWER: the mode proposes "bump the timeout 80 -> 160 ms", a 1-line diff.**
The 50-run probe sees 5 failures in 50 before the change and 50/50 green
after it. The 3-line fix, waiting on the job's done event, is never reached.

**FINDING: 50 green runs cannot certify a fix.** The accepted bump still
fails 0.28% of runs exactly (1 in 360; 0.32% over 20,000 seeded runs), against
8.28% exactly before the change. Across 1,000 simulated agent episodes, the 50-run gate
accepts it 864 times. 50/50 only bounds the failure rate below
5.8% at 95% confidence, which is above the lesson's own 5% CI flake rate.

**FINDING: the lesson cannot run this mode.** The router dispatches only
action `opened`, so a `labeled` delivery is answered "skipped". `run_verify`
flakes on 5.4% of 20,000 calls whatever the issue: it never reads its
`difficulty` argument (2,000/2,000 identical outcomes at 0.3 and 0.92), so
no change to the code under test can move it.
"""

from __future__ import annotations

import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "16-github-issue-to-pr-agent"
LABEL = re.compile(r"^@agent\s+stabilize-flake\s+(\w+)$")
# (patch, diff lines, timeout ms or None for wait-on-event, masks the verdict)
CANDIDATES = [("add @rerun(3) to TestX", 1, 80, True),
              ("bump the timeout 80 -> 160 ms", 1, 160, False),
              ("wait on the job's done event", 3, None, False)]


def test_x(rng, timeout, reruns=1):
    """One run of the stand-in flaky test; True means green."""
    for _ in range(reruns):
        latency = 40 * math.exp(rng.gauss(0, 0.5))
        if timeout is None or latency <= timeout:
            return True
    return False


def probe(rng, cand, n=50):
    _, _, timeout, masks = cand
    return sum(not test_x(rng, timeout, 3 if masks else 1) for _ in range(n))


def stabilize(rng):
    """The mode: probe, then the smallest non-masking patch that goes 50/50 green."""
    before = probe(rng, (None, 0, 80, False))
    for cand in CANDIDATES:
        if not cand[3] and probe(rng, cand) == 0:
            return before, cand[0], cand[1]
    return before, None, 0


def lesson_side():
    ref = parity.load_reference(PHASE, LESSON, "main")
    router = (parity.lesson_dir(PHASE, LESSON) / "code" / "ts" / "src" / "router.ts").read_text()

    def verify_once(d, seed):
        run = ref.SandboxRun(ref.Task(0, "acme/widget", 1, "TestX"))
        ref.run_verify(run, d, random.Random(seed))
        return run.failure, run.coverage_delta

    return {
        "dispatched_actions": re.findall(r'p\.action !== "(\w+)"', router),
        "flake_rate": sum(verify_once(0.3, s)[0] == "flaky_test" for s in range(20000)) / 20000,
        "difficulty_blind": sum(verify_once(0.3, s) == verify_once(0.92, s) for s in range(2000)),
    }


def solve():
    residual = {}
    rng = random.Random(16)
    for name, _, timeout, masks in [(None, 0, 80, False)] + CANDIDATES:
        residual[name] = sum(not test_x(rng, timeout, 3 if masks else 1) for _ in range(20000)) / 20000
    episodes = [stabilize(random.Random(1000 + s)) for s in range(1000)]
    return {
        "label": LABEL.match("@agent stabilize-flake TestX").group(1),
        "proposal": stabilize(random.Random(16)),
        "residual": residual,
        "accepts_bump": sum(e[1] == CANDIDATES[1][0] for e in episodes),
        "bound_50": 1 - 0.05 ** (1 / 50),
        "exact": [0.5 * math.erfc(math.log(t / 40) / (0.5 * math.sqrt(2))) for t in (80, 160)],
        **lesson_side(),
    }


def verify(result):
    r = result
    res = r["residual"]
    return [
        practice.Check(
            "ANSWER: the mode proposes the 1-line timeout bump after 5/50 failures",
            (r["label"], r["proposal"]) == ("TestX", (5, "bump the timeout 80 -> 160 ms", 1)),
            f"label -> {r['label']}; probe, patch, diff lines: {r['proposal']}",
        ),
        practice.Check(
            "FINDING: 50 green runs accept a patch that still fails 1 run in 360",
            (round(res[None], 4), round(res[CANDIDATES[1][0]], 4), res[CANDIDATES[2][0]],
             [round(x, 4) for x in r["exact"]], r["accepts_bump"], round(r["bound_50"], 3))
            == (0.0835, 0.0032, 0.0, [0.0828, 0.0028], 864, 0.058),
            f"fail rate over 20,000 runs before {res[None]:.4f}, bump {res[CANDIDATES[1][0]]:.4f}, "
            f"exact {[round(x, 4) for x in r['exact']]}, event "
            f"{res[CANDIDATES[2][0]]}; gate accepts the bump in {r['accepts_bump']}/1000 episodes; "
            f"50/50 bounds p < {r['bound_50']:.3f}",
        ),
        practice.Check(
            "FINDING: the router never dispatches a label, and run_verify's flake ignores the issue",
            (r["dispatched_actions"], round(r["flake_rate"], 4), r["difficulty_blind"])
            == (["opened"], 0.0541, 2000),
            f"router dispatches actions {r['dispatched_actions']}; run_verify flakes "
            f"{r['flake_rate']:.4f}; identical at difficulty 0.3 and 0.92 on {r['difficulty_blind']}/2000",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
