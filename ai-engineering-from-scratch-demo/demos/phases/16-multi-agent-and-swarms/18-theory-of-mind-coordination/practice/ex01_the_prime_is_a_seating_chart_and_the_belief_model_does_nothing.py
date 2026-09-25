"""Exercise 1 — the prime is a seating chart and the belief model does nothing.

    Run `code/main.py`. Confirm first-order ToM reduces duplication rate by ~7x.
    Does the gap persist when you scale to 5 agents and 5 boxes?

Reading of the exercise: "~7x" is the lesson's own ~35% against ~5%, so it is
checked against what the bench prints; then the ToM condition is taken apart
into its two ingredients -- the turn-0 "preference prime" and the first-order
avoidance rule -- to see which one the gap belongs to.

**ANSWER: it is not 7x; it is 0.965 duplications per trial against 0.00, and
the gap persists -- 1.97 against 0.00 at 5 agents and 5 boxes.** A ratio
with zero in the denominator is not a ratio. Both conditions also complete
200 of 200 trials, where the lesson expects ~60% and ~95%.

**FINDING: without the prime, first-order ToM is zeroth-order, trial for
trial.** Remove the prime and the ToM agents produce the same completions,
duplications and turns as the zeroth-order agents on 400 of 400 trials (3
and 5 agents). After a collision the boxes a loser saw others target last
turn are exactly the boxes that were just collected, which are no longer
available -- so the avoidance rule never removes an option, and the same
seed draws the same boxes.

**FINDING: the prime is a fixed assignment by index.** Each agent is told
every other agent "prefers" box `j % n_boxes`, so agent i's only unclaimed
box is box i, and on every seed the ToM agents' turn-0 choices are exactly
[0, 1, ..., n-1]. An agent with no model of anyone that simply takes box i
matches the ToM condition on every trial at both sizes. The measured
"coordination effect" is a seating chart handed out before turn 0.

**FINDING: the ToM agent the lesson describes is not in the module.** Build
It promises a `ToMAgent` with own beliefs and per-other-agent belief models;
the module has `Agent` with a boolean `tom` and a flat list of (name, box)
observations, and no `other_models` anywhere.

Structure: `trial()` is `run_trial` with four seams -- the prime, the choice
rule, per-turn belief flips and a per-turn log -- and is checked equal to the
reference on all 800 shipped trials before anything else is measured.
Exercises 2-5 load this file for it.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "18-theory-of-mind-coordination"
TRIALS = 200


def reference():
    return parity.load_reference(PHASE, LESSON, "main")


def default_prime(agents, n_boxes):
    for i, agent in enumerate(agents):
        for j, other in enumerate(agents):
            if i != j:
                agent.observe(other.name, j % n_boxes)


def first_order(agent, world, rng, agents):  # the reference rule, as a chooser
    return agent.choose_target(world, rng)


def flip(agents, n_boxes, rng, count):
    """Rewrite `count` random beliefs to a random box -- the hallucination seam."""
    holders = [a for a in agents if a.observations and not a.collected]
    for _ in range(count if holders else 0):
        agent = rng.choice(holders)
        k = rng.randrange(len(agent.observations))
        agent.observations[k] = (agent.observations[k][0], rng.randrange(n_boxes))


def resolve(agents, world, commitments):
    """The reference's collision count and first-in-order-wins resolution."""
    choices = list(commitments.values())
    for name, box in commitments.items():
        if box in world.boxes_with_tokens:
            world.boxes_with_tokens.discard(box)
            next(a for a in agents if a.name == name).collected = True
    return sum(choices.count(b) - 1 for b in set(choices))


def setup(ref, n_agents, n_boxes, tom, prime):
    agents = [ref.Agent(f"agent-{i}", tom=tom) for i in range(n_agents)]
    if tom and prime:
        (prime if callable(prime) else default_prime)(agents, n_boxes)
    return agents


def play_turn(agents, world, rng, chooser, log):  # everyone sees everyone else's pick
    picks = {a.name: chooser(a, world, rng, agents) for a in agents if not a.collected}
    commitments = {name: box for name, box in picks.items() if box >= 0}
    log.append([commitments.get(a.name, -1) for a in agents])
    for observer in agents:
        observer.observations.extend((o, b) for o, b in commitments.items() if o != observer.name)
    return resolve(agents, world, commitments)


def trial(ref, n_agents, n_boxes, tom, seed, max_turns=10, prime=True,
          chooser=first_order, flips=0, log=None):
    rng, world = random.Random(seed), ref.World.new(n_boxes)
    agents, log = setup(ref, n_agents, n_boxes, tom, prime), [] if log is None else log
    duplications = turns = 0
    for turns in range(1, max_turns + 1):
        flip(agents, n_boxes, rng, flips)
        duplications += play_turn(agents, world, rng, chooser, log)
        if all(a.collected for a in agents):
            break
    return sum(a.collected for a in agents), duplications, turns


def bench(ref, n, tom, **kw):
    runs = [trial(ref, n, n, tom, seed, **kw) for seed in range(TRIALS)]
    return (sum(c == n for c, _, _ in runs), round(sum(d for _, d, _ in runs) / TRIALS, 3),
            round(sum(t for _, _, t in runs) / TRIALS, 3))


def by_index(agent, world, rng, agents):  # no model of anyone: take your own number
    return int(agent.name.split("-")[1]) % world.n_boxes


def matches(ref, left, right, sizes=(3, 5)):
    return all(trial(ref, n, n, left[0], s, **left[1]) == trial(ref, n, n, right[0], s, **right[1])
               for n in sizes for s in range(TRIALS))


def first_turns(ref, n):
    logs = [[] for _ in range(TRIALS)]
    for seed, log in enumerate(logs):
        trial(ref, n, n, True, seed, log=log)
    return {tuple(log[0]) for log in logs}


def solve():
    ref = reference()
    sizes = (3, 5)
    parity_ok = all(trial(ref, n, n, tom, s) == ref.run_trial(n, n, tom, s)
                    for n in sizes for tom in (False, True) for s in range(TRIALS))
    source = parity.lesson_dir(PHASE, LESSON).joinpath("code", "main.py").read_text(encoding="utf-8")
    return {
        "parity": parity_ok,
        "bench": {n: {"zeroth": bench(ref, n, False), "tom": bench(ref, n, True)} for n in sizes},
        "noprime_is_zeroth": matches(ref, (True, {"prime": False}), (False, {})),
        "index_is_tom": matches(ref, (False, {"chooser": by_index}), (True, {})),
        "first_turn": {n: first_turns(ref, n) for n in sizes},
        "tom_agent": hasattr(ref, "ToMAgent"), "other_models": "other_models" in source,
    }


def verify(result):
    b3, b5 = result["bench"][3], result["bench"][5]
    return [
        practice.Check(
            "ANSWER: not 7x -- 0.965 against 0.00, and the gap persists at 5x5",
            all([result["parity"], b3["zeroth"] == (200, 0.965, 1.855),
                 b3["tom"] == (200, 0.0, 1.0), b5["zeroth"][1] == 1.97, b5["tom"][1] == 0.0]),
            f"(full completions, duplications/trial, turns) 3x3: zeroth {b3['zeroth']}, ToM "
            f"{b3['tom']}; 5x5: zeroth {b5['zeroth']}, ToM {b5['tom']} -- a zero denominator",
        ),
        practice.Check(
            "FINDING: without the prime, first-order ToM is zeroth-order, trial for trial",
            result["noprime_is_zeroth"],
            "on 400 of 400 trials the unprimed ToM agents match the zeroth-order agents "
            "exactly: the boxes a loser saw targeted last turn are the ones just collected",
        ),
        practice.Check(
            "FINDING: the prime is a fixed assignment by index",
            all([result["index_is_tom"], result["first_turn"][3] == {(0, 1, 2)},
                 result["first_turn"][5] == {(0, 1, 2, 3, 4)}]),
            f"turn-0 ToM choices over all seeds: {sorted(result['first_turn'][3])} and "
            f"{sorted(result['first_turn'][5])}; taking box i matches ToM on every trial",
        ),
        practice.Check(
            "FINDING: the ToM agent the lesson describes is not in the module",
            not result["tom_agent"] and not result["other_models"],
            "no ToMAgent class and no other_models state -- Agent has a bool and a flat "
            "list of (name, box) observations",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
