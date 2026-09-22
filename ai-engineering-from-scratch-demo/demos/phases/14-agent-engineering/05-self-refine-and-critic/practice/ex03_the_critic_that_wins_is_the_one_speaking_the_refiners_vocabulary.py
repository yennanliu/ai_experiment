"""Exercise 3 — the critic that wins is the one speaking the refiner's words.

    Implement a "generator-critic on different models" variant: big model
    generates, small model critiques. Does it beat same-model?

Reading of the exercise: the toy has one scripted generator, so "a different
model critiques" can only vary the *critique text*. That turns out to be the
whole experiment, because `generate` selects its next output by searching the
last critique for `germany` and `everest`. Four critics are run through the
lesson's own loop -- the shipped self-critic, the same judgement reworded, a
rubber stamp, and the external verifier -- scored separately for accuracy and
for convergence.

**ANSWER: a different critic wins only by changing vocabulary, not by being
righter.** The shipped self-critic scores **3/3** on the trajectory's three
outputs and never converges. The same judgement reworded to name the entities
scores **3/3** and converges in **3** iterations, identical to CRITIC.
Accuracy held constant, the outcome flips on two words.

**FINDING: the self-critic is not the thing that fails.** `feedback_self`
flags the Germany error, then the Everest error, then passes the corrected
text -- **3** of **3** correct verdicts. What fails is the coupling:
`generate` matches `germany`/`everest` in the critique, and the self-critic
says `capital` and `continent`. The demo's closing line blames unreliable
self-evaluation for a keyword miss.

**FINDING: a rubber stamp is the fastest run in the lesson.** A critic that
always answers `looks good to me` converges in **1** iteration with **2**
known-wrong facts still in the output, against CRITIC's **3**. Iterations-to-
convergence rewards the worst critic most.

**FINDING: the coupling is measurable as shared words.** Counting critiques
that contain a word `generate` keys on: shipped **0**, rubber stamp **0**,
reworded **2**, external **2**. Every arm with a non-zero count ends with a
clean output and every arm with zero does not -- that count, and not the
critic's provenance, predicts the outcome in all four arms.

Structure: `arm()` rebinds `ref.feedback_self` under `try/finally` and runs
the lesson's `run_loop`; `accuracy()` scores each critic against the same
three outputs.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "05-self-refine-and-critic"
TOPIC, TRIGGERS = "world facts", ("germany", "everest")


def reworded(text):
    """The shipped judgement, with the entity named instead of described."""
    if "Germany" in text and "Paris" in text:
        return "germany: that is the wrong capital", False
    if "Europe" in text and "Everest" in text:
        return "everest: that is the wrong continent", False
    return "no issues", True


def rubber_stamp(text):
    return "looks good to me", True


def arm(ref, critic):
    shipped = ref.feedback_self
    ref.feedback_self = critic
    try:
        history = ref.run_loop(TOPIC, use_critic=False, max_iters=6)
    finally:
        ref.feedback_self = shipped
    return history


def wrong_facts(ref, text):
    lowered = text.lower()
    return [fact for fact in ref.KNOWN_WRONG_FACTS
            if all(word in lowered for word in fact.split() if len(word) > 4)]


def accuracy(critic, outputs):
    """Correct verdicts over the trajectory: fail, fail, pass."""
    expected = [False, False, True]
    return sum(critic(text)[1] == want for text, want in zip(outputs, expected))


def coupling(critic, outputs):
    return sum(any(word in critic(text)[0].lower() for word in TRIGGERS)
               for text in outputs)


def outcomes(ref, runs):
    return ({name: (len(h) if h[-1].verified else 0) for name, h in runs.items()},
            {name: len(wrong_facts(ref, h[-1].output)) for name, h in runs.items()})


def scores(critics, outputs):
    return ({name: accuracy(critic, outputs) for name, critic in critics.items()},
            {name: coupling(critic, outputs) for name, critic in critics.items()})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    outputs = [attempt.output for attempt in ref.run_loop(TOPIC, True, 6)]
    critics = {"shipped": ref.feedback_self, "reworded": reworded,
               "stamp": rubber_stamp, "external": ref.verify_external}
    runs = {name: arm(ref, critics[name]) for name in ("shipped", "reworded", "stamp")}
    runs["external"] = ref.run_loop(TOPIC, True, 6)
    converged, errors_left = outcomes(ref, runs)
    acc, link = scores(critics, outputs)
    return {
        "outputs": len(outputs), "converged": converged, "errors_left": errors_left,
        "accuracy": acc, "coupling": link,
        "shipped_critiques": [ref.feedback_self(text)[0] for text in outputs],
    }


def verify(result):
    won, acc, link = result["converged"], result["accuracy"], result["coupling"]
    clean = {name: won[name] > 0 and result["errors_left"][name] == 0 for name in won}
    return [
        practice.Check(
            "ANSWER: rewording wins where the same judgement did not",
            all([won["shipped"] == 0, won["reworded"] == 3, won["external"] == 3,
                 acc["shipped"] == acc["reworded"] == 3,
                 result["errors_left"]["reworded"] == 0]),
            f"the shipped self-critic scores {acc['shipped']}/3 on the trajectory and "
            f"never converges; the same judgement reworded scores {acc['reworded']}/3 "
            f"and converges in {won['reworded']} iterations, matching CRITIC's "
            f"{won['external']}. Accuracy held constant, the outcome flips on two words",
        ),
        practice.Check(
            "FINDING: the self-critic is not the thing that fails",
            all([acc["shipped"] == 3, result["outputs"] == 3,
                 all("germany" not in c.lower() and "everest" not in c.lower()
                     for c in result["shipped_critiques"])]),
            f"feedback_self returns {result['shipped_critiques']} over the three outputs "
            f"-- {acc['shipped']} of 3 verdicts correct, including the pass on the "
            "corrected text. It says 'capital' and 'continent' where generate looks for "
            "'germany' and 'everest', so the demo blames self-evaluation for a miss",
        ),
        practice.Check(
            "FINDING: a rubber stamp is the fastest run in the lesson",
            all([won["stamp"] == 1, result["errors_left"]["stamp"] == 2,
                 acc["stamp"] == 1, won["stamp"] < won["external"]]),
            f"a critic that always answers 'looks good to me' converges in "
            f"{won['stamp']} iteration with {result['errors_left']['stamp']} known-wrong "
            f"facts still in the output, against CRITIC's {won['external']}, while "
            f"scoring {acc['stamp']}/3. Iterations-to-convergence rewards it most",
        ),
        practice.Check(
            "FINDING: shared words predict convergence in all four arms",
            all([link["shipped"] == 0, link["stamp"] == 0, link["reworded"] == 2,
                 link["external"] == 2,
                 all((link[name] > 0) == clean[name] for name in link)]),
            f"counting critiques that contain a word generate keys on: "
            f"{link}. Every arm with a non-zero count ends clean and every arm with "
            f"zero does not ({clean}) -- the critic's provenance predicts nothing and "
            "its vocabulary predicts everything",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
