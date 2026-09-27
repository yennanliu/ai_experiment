"""Exercise 3 — a patient U beats the reference's first-come audit budget every time; hidden, paced audits cut it to 5% under Trusted Editing at H = 0.05.

    The 2024 paper assumes U can see the protocol. Design a protocol that
    hides from U which outputs are being audited. What does this protocol
    require from the deployment infrastructure?

Reading of the exercise: the question is what U gains from knowing which
outputs get audited, and what it takes to deny it that. So the reference's
own audit rule is played as the paper's game: 1000 tasks per episode; a
backdoor H audits is caught and ends the deployment (blue wins); the first
backdoor that ships unaudited is a red win. Every run_protocol branch spends
H first-come (`used_h / n < h_budget`), so the moment the budget runs out is
a public fact. The visible red team is patient: it behaves until the budget
is spent, then attacks every task. The designed protocol keeps the same
budget but decides each audit with a secret coin, paced at H / (expected flag
rate). Its red team has no signal, so it starts attacking at a random task.
2000 seeded episodes per cell, reusing the reference's `t_edit`.

**ANSWER: hide the audit decision *and* pace it; hiding alone does not
help an unflagged audit.** Red-team win rate:

| protocol | H | first-come (visible) | hidden, paced |
|---|---:|---:|---:|
| upfront | 0.02 | 1.000 | 0.9865 |
| upfront | 0.10 | 1.000 | 0.8995 |
| Trusted Editing | 0.02 | 1.000 | 0.6210 |
| Trusted Editing | 0.05 | 0.5175 | 0.0495 |
| Trusted Editing | 0.10 | 0.000 | 0.0000 |

The designed protocol: T edits every output. Each output T is unsure about
goes to H with probability q = H / 0.05, where 0.05 is t_edit's over-reject
rate on benign work. The coin is drawn after the output is committed, from
randomness U never sees, and it stops at the hard budget.

**FINDING: the reference's budget has a public exhaustion point.** Upfront
auditing spends it on exactly tasks 1-20 or 1-100, so the patient attack
lands on task 21 or 101 and always ships. Under Trusted Editing at H = 0.02,
benign over-rejects alone spend the 20 audits (0.05 per task), and the attack
lands on task 404 on average. A patient U that counts its tasks wins every
episode there. At H = 0.05 the budget runs out inside 1000 tasks in about
half the episodes, and U wins 0.5175.

**FINDING: hiding helps only when H is flag-directed.** Under upfront
auditing, a hidden 10% audit still lets one attack through with probability
0.8995 (1 - H), so secrecy cannot rescue it. Under Trusted Editing, T routes H onto
the 30% of backdoors it fails to remove, so paced secrecy leaves U about 1 - q: 0.621 against
0.60 at H = 0.02.

What the protocol requires from infrastructure. An audit coin drawn after U's
output is committed, from a source U cannot read. Audited and unaudited
outputs must be indistinguishable to U: same path, same latency, no verdict
fed back into its context. U instances must not see counters, their own task
index or the audit logs. The pacing needs a measured flag-rate estimate
(here, the 0.05 over-reject rate) and a hard budget cap. And a caught
backdoor must actually halt the deployment, which is the only reason
catching one counts as a win.

Structure: `episode()` plays one deployment against a schedule; `game()` seeds
both the coin and the reference's module-level random (restored after).
"""

from __future__ import annotations

import inspect
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "10-ai-control-subversion"
N, EPISODES = 1000, 2000
CELLS = (("upfront", 0.02), ("upfront", 0.10), ("trusted_ed", 0.02),
         ("trusted_ed", 0.05), ("trusted_ed", 0.10))


def step(ref, proto, attack, t_skill):
    """(eligible for audit, backdoored): upfront audits everything; TE only T's low-confidence."""
    if proto == "upfront":
        return True, attack
    _, bd, low_confidence = ref.t_edit("x", attack, t_skill)
    return low_confidence, bd


def episode(ref, rng, proto, h, hidden, rates):
    """('caught' | 'shipped' | 'none', task index where it ended). rates = (flag rate, t_skill)."""
    (flag_rate, t_skill), budget, used, start = rates, h * N, 0, rng.randrange(N)
    q = min(1.0, h / {"upfront": 1.0, "trusted_ed": flag_rate}[proto])
    for i in range(N):
        eligible, bd = step(ref, proto, i >= start if hidden else used >= budget, t_skill)
        if not eligible:
            continue
        audit = used < budget and (rng.random() < q if hidden else True)
        used += audit
        if bd:
            return ("caught" if audit else "shipped"), i + 1
    return "none", N


def game(ref, proto, h, hidden, rates, seed=0):
    """(red-team win rate, mean task the episode ended on)."""
    rng, saved = random.Random(seed), ref.random
    ref.random = random.Random(seed + 1)
    try:
        out = [episode(ref, rng, proto, h, hidden, rates) for _ in range(EPISODES)]
    finally:
        ref.random = saved
    return sum(o == "shipped" for o, _ in out) / EPISODES, round(statistics.mean(i for _, i in out))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    flag_rate = inspect.signature(ref.t_edit).parameters["overreject"].default
    rates = flag_rate, inspect.signature(ref.run_protocol).parameters["t_skill"].default
    table = {(p, h): (game(ref, p, h, False, rates), game(ref, p, h, True, rates))
             for p, h in CELLS}
    return {"table": table, "flag_rate": flag_rate, "t_skill": rates[1],
            "first_come": inspect.getsource(ref.run_protocol).count("used_h / n < h_budget")}


def verify(result):
    table = result["table"]
    wins = {cell: (vis[0], hid[0]) for cell, (vis, hid) in table.items()}
    return [
        practice.Check(
            "ANSWER: hide the audit decision and pace it; hiding alone does not help an unflagged audit",
            wins == {("upfront", 0.02): (1.0, 0.9865), ("upfront", 0.1): (1.0, 0.8995),
                     ("trusted_ed", 0.02): (1.0, 0.621), ("trusted_ed", 0.05): (0.5175, 0.0495),
                     ("trusted_ed", 0.1): (0.0, 0.0)},
            f"red-team win rate (first-come, hidden paced): {wins}",
        ),
        practice.Check(
            "FINDING: the reference's budget has a public exhaustion point",
            result["first_come"] == 3 and table[("upfront", 0.02)][0][1] == 21
            and table[("upfront", 0.1)][0][1] == 101 and table[("trusted_ed", 0.02)][0][1] == 404
            and result["flag_rate"] == 0.05,
            f"run_protocol spends H first-come in {result['first_come']} branches; patient attack "
            f"lands at task {table[('upfront', 0.02)][0][1]} / {table[('upfront', 0.1)][0][1]} "
            f"(upfront) and {table[('trusted_ed', 0.02)][0][1]} (TE, H = 0.02)",
        ),
        practice.Check(
            "FINDING: hiding helps only when H is flag-directed",
            abs(wins[("upfront", 0.1)][1] - 0.9) < 0.01 and result["t_skill"] == 0.7
            and abs(wins[("trusted_ed", 0.02)][1] - (1 - 0.02 / result["flag_rate"])) < 0.03,
            f"hidden upfront at H = 0.10 {wins[('upfront', 0.1)][1]} (1 - H = 0.90); hidden TE at "
            f"H = 0.02 {wins[('trusted_ed', 0.02)][1]} (1 - q = {1 - 0.02 / result['flag_rate']:.2f}); T misses {1 - result['t_skill']:.0%}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
