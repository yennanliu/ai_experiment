"""Exercise 2 — second order helps only when exactly one agent is left with one box.

    Implement second-order ToM (agent A models what B thinks about C). Does it
    improve over first-order? On what tasks?

Reading of the exercise: A second-order agent rebuilds each other agent B's
belief window -- everything B has observed about everyone else, which is
public -- runs B's own first-order rule on it, and treats B's choice as
claimed whenever that rule leaves B a single box. It is compared with
first-order on the lesson's task and on the one variation that gives it
something to infer: a prime in which some agents never announced a box.

**ANSWER: on the lesson's task it improves nothing -- it is first-order on
800 of 800 trials, primed or not, at 3x3 and 5x5.** Primed, every agent is already down to
its own box, so there is nothing left to predict; unprimed, no agent's rule
ever narrows to one box, so there is no prediction to make. The task where
it helps is the one where one agent knows more than the others: with one
silent agent, first-order duplicates 1.04 per trial at 3x3, 1.615 at 4x4 and
2.525 at 5x5, and second-order 0.00 at all three. The silent agent heard
everyone and is left with one box; the others heard it say nothing and have
two. Reasoning about *its* beliefs is what tells them which of the two is
taken. With two silent agents, nobody is down to one box and second-order
is first-order again, at 2.65.

**FINDING: a partial prime makes first-order worse than no ToM.** With one
agent silent, first-order duplicates 1.04, 1.615 and 2.525 per trial at 3x3,
4x4 and 5x5, against the zeroth-order agents' 0.965, 1.40 and 1.97. The announced
agents all avoid the same announced boxes and pile onto the unannounced one.

Li et al.'s second order is narrower than the exercise's: it is "what others
believe about *my* mental state", and GPT-4 scored 64.3% on it against 60.0%
first-order -- the one row of their Table 2 where second order is not lower.

Structure: `second_order()` is the chooser; ex01's `trial()` runs it.
"""

from __future__ import annotations

import pathlib

from harness import practice

HERE = pathlib.Path(__file__).resolve().parent
SIM = practice.load_module(next(HERE.glob("ex01_*.py")))


def options(agent, world):
    """The reference's first-order option set, computed from `agent`'s own window."""
    available = sorted(world.boxes_with_tokens)
    recent = {box for _, box in agent.observations[-(len(world.boxes_with_tokens) + 2):]}
    return [b for b in available if b not in recent] or available


def second_order(agent, world, rng, agents):
    if agent.collected or not world.boxes_with_tokens:
        return -1
    claimed = {options(b, world)[0] for b in agents
               if b is not agent and not b.collected and len(options(b, world)) == 1}
    mine = options(agent, world)
    return rng.choice([b for b in mine if b not in claimed] or mine)


def silent(quiet):
    """A prime in which the agents in `quiet` never announce a box."""
    def prime(agents, n_boxes):
        for i, agent in enumerate(agents):
            for j, other in enumerate(agents):
                if i != j and j not in quiet:
                    agent.observe(other.name, j % n_boxes)
    return prime


def solve():
    ref = SIM.reference()
    same = all(SIM.trial(ref, n, n, True, s, prime=p) ==
               SIM.trial(ref, n, n, True, s, prime=p, chooser=second_order)
               for n in (3, 5) for p in (True, False) for s in range(SIM.TRIALS))
    cases = {(3, (2,)): None, (4, (3,)): None, (5, (4,)): None, (5, (3, 4)): None}
    for n, quiet in cases:
        cases[(n, quiet)] = {"first": SIM.bench(ref, n, True, prime=silent(quiet))[1],
                             "second": SIM.bench(ref, n, True, prime=silent(quiet),
                                                 chooser=second_order)[1],
                             "zeroth": SIM.bench(ref, n, False)[1]}
    return {"same": same, "cases": cases}


def verify(result):
    c = result["cases"]
    one = [c[(3, (2,))], c[(4, (3,))], c[(5, (4,))]]
    return [
        practice.Check(
            "ANSWER: nothing on the lesson's task; everything when one agent is left with one box",
            all([result["same"], [x["first"] for x in one] == [1.04, 1.615, 2.525],
                 all(x["second"] == 0.0 for x in one),
                 c[(5, (3, 4))]["first"] == c[(5, (3, 4))]["second"] == 2.65]),
            f"primed or not, second order equals first order on 800 of 800 trials; with "
            f"one silent agent first order duplicates {[x['first'] for x in one]} at 3x3, "
            f"4x4, 5x5 and second order {[x['second'] for x in one]}; with two silent "
            f"agents both give {c[(5, (3, 4))]['first']}",
        ),
        practice.Check(
            "FINDING: a partial prime makes first-order worse than no ToM",
            all(x["first"] > x["zeroth"] for x in one),
            f"one silent agent: first order {[x['first'] for x in one]} duplications per "
            f"trial against zeroth order {[x['zeroth'] for x in one]} at 3x3, 4x4, 5x5 -- "
            "the announced agents avoid the same boxes and pile onto the silent one's",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
