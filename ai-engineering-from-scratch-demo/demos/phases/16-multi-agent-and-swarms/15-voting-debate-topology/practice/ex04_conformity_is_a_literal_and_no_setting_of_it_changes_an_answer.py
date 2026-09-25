"""Exercise 4 — conformity is a literal, and no setting of it changes an answer.

    Read the AgentVerse paper (ICLR 2024). Identify which emergent behavior
    your implementation exhibits most strongly. Can you elicit the opposite
    behavior by a prompt change?

Reading of the exercise: the implementation is the lesson's `run_graph`, the
only topology in which agents see each other. "A prompt change" has to be
read as a change to the one parameter standing in for the prompt, since the
simulated agents have no prompt; "the opposite behavior" is taken two ways,
as no conformity and as active contrarianism.

**ANSWER: conformity, the only behavior it has -- and the opposite can be
elicited, but no setting of the conformity knob changes a single answer.**
AgentVerse names volunteer, conformity and destructive behaviors. Volunteer
behavior (time, resource and assistance contributions) cannot occur: a
`SimAgent` has one method, `answer`, and every agent answers the same
question, so there is no work to take on. Conformity is hardcoded -- a
dissenter moves to the current majority when `rng.random() < 0.4`, a literal
inside `run_graph`. Over 200 seeds at N=5, setting it to 0.0, 0.4 or 1.0
gives the same 200 final answers and accuracy 0.905 at every setting.
Only the contrarian rule -- a majority member defects to the runner-up with
probability 0.4 -- moves anything: accuracy falls to 0.555, with 85 answers
flipped right-to-wrong and 15 wrong-to-right.

**FINDING: conformity is invisible to the vote it is supposed to distort.**
Moving a dissenter to the argmax never moves the argmax, so in this harness
conformity cannot be the groupthink risk the lesson warns about; it only
raises agreement. Its opposite, which spreads votes *away* from the leader,
is the behavior with teeth -- and 85 of its 100 flips go the wrong way,
because the leader is usually right.

**FINDING: the paper's conformity example is agreement with a correct
critic.** In AgentVerse's Minecraft case Charlie drifts to off-task crafting,
Alice and Bob criticise it, and Charlie "acknowledges his mistake and
re-focuses on the mutual tasks". The lesson's gloss -- an agent aligning with
a critic "even when the critic is wrong" -- is the sycophancy reading, not
the paper's example; and the paper's third behavior, destructive (agents
harming others or destroying a village library for materials), goes
unmentioned.

Structure: `run()` is the reference `run_graph` with its update rule passed
in; it reproduces the reference on every seed before a rule is varied.
"""

from __future__ import annotations

import collections
import inspect
import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "15-voting-debate-topology"
N, TRIALS = 5, 200


def conform(prob):
    return lambda pos, lead, runner, rng: lead if pos != lead and rng.random() < prob else pos


def contrarian(prob):
    return lambda pos, lead, runner, rng: (runner if pos == lead and runner
                                           and rng.random() < prob else pos)


def run(ref, agents, rng, rule, rounds=2):
    positions = [a.answer("RIGHT", rng) for a in agents]
    for _ in range(rounds - 1):
        lead = ref.majority(positions)
        runner = next((k for k, _ in collections.Counter(positions).most_common()
                       if k != lead), None)
        positions = [rule(p, lead, runner, rng) for p in positions]
    return ref.majority(positions)


def finals(ref, rule):
    return [run(ref, ref.make_agents(N, False, t), random.Random(t * 31 + 7), rule)
            for t in range(TRIALS)]


def accuracy(answers):
    return sum(a == "RIGHT" for a in answers) / TRIALS


def flips(before, after):
    """(right-to-wrong, wrong-to-right) between two lists of finals."""
    pairs = list(zip(before, after))
    return (sum(a == "RIGHT" != b for a, b in pairs), sum(b == "RIGHT" != a for a, b in pairs))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reference = [ref.run_graph(ref.make_agents(N, False, t), "RIGHT",
                               random.Random(t * 31 + 7)).final_answer for t in range(TRIALS)]
    knob = {p: finals(ref, conform(p)) for p in (0.0, 0.4, 1.0)}
    contra = finals(ref, contrarian(0.4))
    doc = parity.doc_text(PHASE, LESSON)
    to_wrong, to_right = flips(knob[0.4], contra)
    return {
        "reproduces": knob[0.4] == reference,
        "identical": all(f == knob[0.4] for f in knob.values()),
        "accs": {p: accuracy(f) for p, f in knob.items()}, "contra": accuracy(contra),
        "to_wrong": to_wrong, "to_right": to_right,
        "methods": [m for m in vars(ref.SimAgent) if not m.startswith("_")],
        "literal": "rng.random() < 0.4" in inspect.getsource(ref.run_graph),
        "gloss": "even when the critic is wrong" in doc,
        "destructive": "destructive" in doc.lower(),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: conformity, and no setting of the knob changes an answer",
            all([result["reproduces"], result["identical"], result["literal"],
                 "answer" in result["methods"], result["contra"] < result["accs"][0.4]]),
            f"conformity at 0.0/0.4/1.0 gives identical finals, accuracy {result['accs']}; "
            f"the contrarian rule gives {result['contra']}, flipping "
            f"{result['to_wrong']} right-to-wrong and {result['to_right']} wrong-to-right",
        ),
        practice.Check(
            "FINDING: conformity is invisible to the vote it is supposed to distort",
            result["to_wrong"] > 5 * result["to_right"],
            f"contrarian flips go the wrong way {result['to_wrong']} times of "
            f"{result['to_wrong'] + result['to_right']}: the leader is usually right",
        ),
        practice.Check(
            "FINDING: the paper's conformity example is agreement with a correct critic",
            result["gloss"] and not result["destructive"],
            "the lesson glosses conformity as aligning with a critic 'even when the critic "
            "is wrong'; AgentVerse's example is Charlie re-focusing after correct "
            "criticism, and its destructive behaviors go unmentioned",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
