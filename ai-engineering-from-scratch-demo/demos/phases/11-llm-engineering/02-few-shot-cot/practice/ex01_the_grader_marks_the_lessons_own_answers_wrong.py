"""Exercise 1 — the grader marks the lesson's own reference answers wrong.

    **Measure the gap**: Take 10 GSM8K problems. Solve each with zero-shot,
    few-shot, zero-shot CoT, and few-shot CoT. Record accuracy for each. Which
    technique gives the biggest lift on your model?

Reading of the exercise: no key is available, so the gap between the four
techniques cannot be measured -- but the instrument that would measure it can,
and it is the same `extract_answer` plus `str(answer) == str(expected)` that
`run_comparison` uses for all four arms. It is graded on replies whose correct
score is known by construction, starting with the lesson's own exemplars.

**ANSWER: the comparison cannot run, because the grader fails 5 of the 5
reference answers the lesson writes into its own prompts.** `build_cot_prompt`
emits `A: {reasoning} The answer is {answer}.` Feed that exact line back and
`extract_answer` returns `'18.'`, `'3.'`, `'70000.'`, `'624.'`, `'5.'` -- the
sentence-final full stop included -- and `str('18.') == str('18')` is False.

**MECHANISM: `([\\d,]+\\.?\\d*)` makes the decimal point optional and the digits
after it optional too.** So the group matches "18" and then swallows the "."
that ends the sentence. Every system prompt in the lesson orders the model to
"End with: 'The answer is [number]'", and a model that ends a sentence scores 0.

**FINDING: the damage is technique-specific.** Over 16 labelled reply shapes the
grader gets 8 right -- all six few-shot-CoT shapes fail and all six
zero-shot shapes pass. The failures are CoT-shaped: the trailing full stop, and
self-correction -- `re.search` returns the *first* "the answer is", so
"Initially the answer is 60, but the answer is 72" is graded as 60.

**FINDING: one of the five test questions is exemplar #4, verbatim.**
`GSM8K_EXAMPLES[3]["question"] == TEST_QUESTIONS[4]["question"]`, answer
included. At the default `num_examples=3` it is outside the slice; at 4 or more
the few-shot arms are shown the answer to a question they are being scored on.

**CONTROL: two edits fix the instrument.** Make the decimal non-optional once
started, `([\\d,]+(?:\\.\\d+)?)`, and take the last match rather than the first.
The lesson's own exemplars then score 5 of 5 and the corpus 16 of 16.

Structure: `SHAPES` is the labelled reply corpus, `graded` the lesson's own
grading step, and `repaired` the two-edit control.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "02-few-shot-cot"
# (reply, the expected answer string, the technique whose format this is)
SHAPES = [
    ("The answer is 72.", "72", "few_shot_cot"), ("The answer is 72", "72", "zero_shot"),
    ("72", "72", "zero_shot"), ("The answer is $10.", "10", "few_shot_cot"),
    ("She sold 48 + 24 = 72 clips.", "72", "zero_shot_cot"),
    ("Initially the answer is 60, but the answer is 72.", "72", "zero_shot_cot"),
    ("Step 1: 48. Step 2: 24. The answer is 72.", "72", "few_shot_cot"),
    ("The answer is 1,000.", "1000", "few_shot_cot"), ("#### 72", "72", "zero_shot"),
    ("The answer is 5", "5", "zero_shot"), ("Total: 624 pages", "624", "zero_shot_cot"),
    ("The answer is 42.", "42", "few_shot_cot"), ("= 70000", "70000", "zero_shot"),
    ("Let me think step by step. 12 * 52 = 624. The answer is 624.", "624", "zero_shot_cot"),
    ("The answer is 3.", "3", "few_shot_cot"), ("I make it 18 dollars.", "18", "zero_shot"),
]
REPAIRED = [r"[Tt]he answer is[:\s]*\$?([\d,]+(?:\.\d+)?)", r"#### ([\d,]+(?:\.\d+)?)"]


def exemplar_lines(ref):
    """The exact strings `build_cot_prompt` writes into the prompt."""
    return [(f"{e['reasoning']} The answer is {e['answer']}.", e["answer"])
            for e in ref.GSM8K_EXAMPLES]


def graded(ref, reply, expected):
    """`run_comparison`'s scoring step, verbatim: extract, then compare as strings."""
    return str(ref.extract_answer(reply)) == str(expected)


def repaired(text):
    """The control: decimals only when they have digits, and the last match wins."""
    for pattern in REPAIRED:
        found = re.findall(pattern, text)
        if found:
            return found[-1].replace(",", "")
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None


def by_technique(ref):
    """How many of each technique's reply shapes the lesson's grader scores right."""
    tally = {}
    for reply, expected, technique in SHAPES:
        hit, total = tally.get(technique, (0, 0))
        tally[technique] = (hit + graded(ref, reply, expected), total + 1)
    return tally


SELF_CORRECT = "Initially the answer is 60, but the answer is 72."


def as_shipped(ref, lines):
    return {"exemplars": [ref.extract_answer(line) for line, _ in lines],
            "expected": [want for _, want in lines],
            "exemplars_right": sum(graded(ref, line, want) for line, want in lines),
            "corpus_right": sum(graded(ref, r, w) for r, w, _ in SHAPES),
            "corpus": len(SHAPES), "by_technique": by_technique(ref),
            "first_match": ref.extract_answer(SELF_CORRECT)}


def repaired_tally(lines):
    return {"fixed_exemplars": sum(repaired(line) == want for line, want in lines),
            "fixed_corpus": sum(repaired(r) == w for r, w, _ in SHAPES),
            "fixed_first": repaired(SELF_CORRECT)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "advanced_prompting")
    lines = exemplar_lines(ref)
    prompt = ref.build_cot_prompt("Q?", ref.GSM8K_EXAMPLES, 4)[1]
    return {**as_shipped(ref, lines), **repaired_tally(lines), "default_examples": 3,
            "leaked": [i for i, e in enumerate(ref.GSM8K_EXAMPLES)
                       if e["question"] in [t["question"] for t in ref.TEST_QUESTIONS]],
            "leaked_in_prompt": [t["question"][:24] in prompt for t in ref.TEST_QUESTIONS]}


def verify(result):
    cot = result["by_technique"].get("few_shot_cot", (0, 0))
    zero = result["by_technique"].get("zero_shot", (0, 0))
    return [
        practice.Check(
            "ANSWER: the grader fails 5 of the 5 answers the lesson puts in its own prompt",
            all([result["exemplars_right"] == 0, len(result["expected"]) == 5]),
            f"`build_cot_prompt` writes 'A: {{reasoning}} The answer is {{answer}}.' Feeding "
            f"that line back gives {result['exemplars']} against expected "
            f"{result['expected']} -- {result['exemplars_right']} of 5 scored correct, "
            "because str('18.') == str('18') is False",
        ),
        practice.Check(
            "MECHANISM: the capture group swallows the sentence-final full stop",
            all([str(result["exemplars"][0]).endswith("."), result["exemplars"][0] != "18"]),
            r"([\d,]+\.?\d*) makes the point optional and the digits after it optional too, "
            f"so it matches '18' and then takes the '.' as well -> {result['exemplars'][0]!r}. "
            "Every system prompt in the lesson orders 'End with: The answer is [number]'",
        ),
        practice.Check(
            "FINDING: the failures are concentrated in the CoT-shaped replies",
            all([result["corpus_right"] < result["corpus"], result["first_match"] == "60",
                 cot[0] < zero[0]]),
            f"{result['corpus_right']} of {result['corpus']} labelled reply shapes graded "
            f"right, by technique {result['by_technique']}. `re.search` returns the first "
            f"match, so 'Initially the answer is 60, but the answer is 72' grades as "
            f"{result['first_match']!r} -- and self-correction is what CoT produces",
        ),
        practice.Check(
            "FINDING: one test question is few-shot exemplar #4, answer included",
            all([result["leaked"] == [3], any(result["leaked_in_prompt"])]),
            f"GSM8K_EXAMPLES[{result['leaked'][0]}]['question'] is TEST_QUESTIONS[4]"
            f"['question'] verbatim. At the default num_examples="
            f"{result['default_examples']} it sits outside examples[:3]; at 4 or more the "
            "few-shot arms are shown the answer to a question they are scored on",
        ),
        practice.Check(
            "CONTROL: two edits to the extractor score the exemplars 5 of 5",
            all([result["fixed_exemplars"] == 5, result["fixed_corpus"] == result["corpus"],
                 result["fixed_first"] == "72"]),
            r"requiring digits after the point, ([\d,]+(?:\.\d+)?), and taking the last "
            f"match instead of the first scores the lesson's exemplars "
            f"{result['fixed_exemplars']} of 5, the corpus {result['fixed_corpus']} of "
            f"{result['corpus']}, and the self-correcting reply {result['fixed_first']!r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
