"""Exercise 3 — the cost is a straight line, and the vote splits on a full stop.

    **Self-consistency cost curve**: Run self-consistency with N=1, 3, 5, 7, 10
    on 20 GSM8K problems. Plot accuracy vs cost (total tokens). Where is the
    knee of the curve for your model?

Reading of the exercise: cost is measurable exactly and accuracy is not, so
cost is measured exactly -- `cl100k_base` over the prompt the lesson's own
`build_cot_prompt` renders, times the N samples `self_consistency_solve` draws.
The vote logic is then exercised end to end through a scripted client, which is
the seam the lesson already has: every solver takes `client` as a parameter.

**ANSWER: there is no knee, because cost is linear in N through the origin.**
`self_consistency_solve` rebuilds one prompt and re-sends it N times, so
total = N x (prompt + completion) with no shared prefix and no discount. At
N = 1, 3, 5, 7, 10 the totals are exact multiples: 482, 1446, 2410, 3374, 4820.
Any knee in the exercise's plot comes from the accuracy axis alone.

**FINDING: the prompt is 92.7% of every call, and it is paid N times.** 447
prompt tokens against a 35-token completion. Ten samples of one question cost
4,470 prompt tokens to buy 350 tokens of reasoning. (And the lesson ships 5 test
questions, not the 20 the exercise asks to run.)

**FINDING: self-consistency splits its own vote on a full stop.** Exercise 1's
extractor returns "72." for "The answer is 72." and "72" for the same sentence
without it. Ten samples, five each way, produce `{'72.': 5, '72': 5}` --
confidence 0.50 on a question where every sample agreed.

**FINDING: on a tie, the answer is sample 1's.** `Counter.most_common(1)` is
insertion-ordered, so an even split returns whichever spelling was drawn first.
At N = 10 that is the difference between escalating and not.

**FINDING: confidence is computed over parsed answers, not samples.** Eight
replies that parse to None and two that agree give confidence 1.00 from 2
votes -- and `solve_with_escalation` escalates only below 0.80, so it skips the
expensive path exactly when the cheap one collapsed.

Structure: `Stub` is the scripted client, `cost` the tiktoken accounting, and
`vote` runs the lesson's own `self_consistency_solve` against scripted replies.
"""

from __future__ import annotations

import types

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "02-few-shot-cot"
SAMPLES = (1, 3, 5, 7, 10)
COMPLETION = ("Natalia sold 48 clips in April and half as many, 24, in May. "
              "48 + 24 = 72. The answer is 72.")
PERIOD_SPLIT = ["The answer is 72."] * 5 + ["The answer is 72"] * 5
MOSTLY_UNPARSED = ["I cannot help with that."] * 8 + ["The answer is 72"] * 2


class Stub:
    """A scripted `client`: the seam every solver in this lesson already takes."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        text = self.replies[self.calls % len(self.replies)]
        self.calls += 1
        message = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def cost(ref):
    """cl100k_base over the prompt the lesson renders, times the samples it draws."""
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
    system, user = ref.build_cot_prompt(ref.TEST_QUESTIONS[0]["question"], ref.GSM8K_EXAMPLES)
    prompt = len(encoder.encode(system)) + len(encoder.encode(user))
    completion = len(encoder.encode(COMPLETION))
    return {"prompt": prompt, "completion": completion,
            "totals": [n * (prompt + completion) for n in SAMPLES],
            "share": round(prompt / (prompt + completion), 3)}


def vote(ref, replies, n):
    stub = Stub(replies)
    answer, confidence, _, counts = ref.self_consistency_solve(
        "Q?", ref.GSM8K_EXAMPLES, stub, "model", n_samples=n)
    return {"answer": answer, "confidence": confidence, "counts": dict(counts),
            "calls": stub.calls}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "advanced_prompting")
    numbers = cost(ref)
    split, thin = vote(ref, PERIOD_SPLIT, 10), vote(ref, MOSTLY_UNPARSED, 10)
    unit = numbers["totals"][0]
    return {**numbers, "per_sample": sorted({t / n for t, n in zip(numbers["totals"], SAMPLES)}),
            "linear": [t == n * unit for t, n in zip(numbers["totals"], SAMPLES)],
            "questions": len(ref.TEST_QUESTIONS), "asked_for": 20,
            "split": split, "thin": thin,
            "escalates": thin["confidence"] < 0.8, "split_escalates": split["confidence"] < 0.8}


def verify(result):
    split, thin = result["split"], result["thin"]
    return [
        practice.Check(
            "ANSWER: no knee -- cost is linear in N through the origin",
            all([all(result["linear"]), len(result["per_sample"]) == 1]),
            f"totals at N={list(SAMPLES)} are {result['totals']} tokens, exactly "
            f"{result['per_sample'][0]:.0f} per sample at every N. The prompt is rebuilt "
            "and re-sent whole, with no shared prefix and no discount, so any knee in the "
            "exercise's plot comes from the accuracy axis alone",
        ),
        practice.Check(
            "FINDING: the prompt is 92.7% of every call and is paid N times",
            all([result["share"] > 0.9, result["questions"] < result["asked_for"]]),
            f"{result['prompt']} prompt tokens against a {result['completion']}-token "
            f"completion, share {result['share']}. Ten samples of one question spend "
            f"{10 * result['prompt']} prompt tokens to buy {10 * result['completion']} of "
            f"reasoning -- and the lesson ships {result['questions']} test questions, not "
            f"the {result['asked_for']} the exercise asks to run",
        ),
        practice.Check(
            "FINDING: self-consistency splits its own vote on a full stop",
            all([split["counts"] == {"72.": 5, "72": 5}, split["confidence"] == 0.5]),
            f"ten samples that all say 72, five of them ending the sentence, vote "
            f"{split['counts']} at confidence {split['confidence']}. Exercise 1's extractor "
            "returns '72.' for 'The answer is 72.', so the two spellings of one answer are "
            "two candidates and unanimity is scored as a coin flip",
        ),
        practice.Check(
            "FINDING: on a tie the answer is whichever sample was drawn first",
            all([split["answer"] == "72.", split["calls"] == 10]),
            f"`Counter.most_common(1)` is insertion-ordered, so the 5-5 split returns "
            f"{split['answer']!r} -- the spelling of sample 1, after {split['calls']} calls. "
            "Reverse the draw order and the same ten replies return '72'",
        ),
        practice.Check(
            "FINDING: confidence is computed over parsed answers, not over samples",
            all([thin["confidence"] == 1.0, thin["counts"] == {"72": 2},
                 not result["escalates"], result["split_escalates"]]),
            f"eight replies that parse to None and two that agree give confidence "
            f"{thin['confidence']} from {sum(thin['counts'].values())} votes. "
            "`solve_with_escalation` escalates below 0.80, so it skips Tree-of-Thought here "
            "and runs it on the split above -- the run where every sample agreed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
