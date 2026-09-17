<!-- generated:start -->
# 11-llm-engineering / 02-few-shot-cot

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/02-few-shot-cot/) · upstream spec
`phases/11-llm-engineering/02-few-shot-cot/docs/en.md`

```bash
uv run demo practice run 02-few-shot-cot --ex 1
uv run demo explain 02-few-shot-cot --ex 1
uv run pytest demos/phases/11-llm-engineering/02-few-shot-cot
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Measure the gap: Take 10 GSM8K problems. Solve each with zero-shot, few-shot, zero-shot CoT,… | code | T1 | `ex01_the_grader_marks_the_lessons_own_answers_wrong.py` |
| 2 | Example selection experiment: For the same 10 problems, compare random example selection vs h… | code | T1 | `ex02_the_pool_is_five_and_quality_selection_retrieves_the_test.py` |
| 3 | Self-consistency cost curve: Run self-consistency with N=1, 3, 5, 7, 10 on 20 GSM8K problems.… | code | T1 | `ex03_the_cost_is_a_straight_line_and_the_vote_splits_on_a_full_stop.py` |
| 4 | Build a ReAct loop: Extend the pipeline with a calculator tool. When the model generates a ma… | code | T1 | `ex04_the_answer_line_is_checked_before_the_action_line.py` |
| 5 | ToT for creative tasks: Adapt the Tree-of-Thought solver for a creative writing task: "Write… | code | T1 | `ex05_the_beam_discards_its_best_node_and_returns_a_number.py` |
<!-- generated:end -->

## Answers

Every solver in this lesson takes `client` as a parameter, so the lesson already
has the seam these solutions use: a scripted client, and no key. What that
exposes is that the four techniques the lesson compares are never actually
compared — they are read through one `extract_answer` and one
`str(answer) == str(expected)`, and that pair marks the lesson's own reference
answers wrong. The same bug then splits self-consistency's vote, decides which
runs escalate, and turns a Tree-of-Thought over six-word stories into a number.

All five exercises are **T1**: `advanced_prompting.py` does
`from openai import OpenAI` at module scope, so importing it needs
`uv sync --extra llm`. None of them needs a key. CI runs at `DEMO_TIER=T0` and
skips the lesson.

### 1 — the grader marks the lesson's own answers wrong

`build_cot_prompt` writes this line into every few-shot prompt:

```python
example_text += f"A: {ex['reasoning']} The answer is {ex['answer']}.\n\n"
```

Feed that exact line back through the grader `run_comparison` uses:

| exemplar | `extract_answer` | expected | graded |
|---|---|---|---|
| Janet's ducks | `'18.'` | `'18'` | ❌ |
| the robe | `'3.'` | `'3'` | ❌ |
| Josh's house | `'70000.'` | `'70000'` | ❌ |
| James's letters | `'624.'` | `'624'` | ❌ |
| Wendi's chickens | `'5.'` | `'5'` | ❌ |

**ANSWER: 0 of 5.** The gap the exercise asks you to measure cannot be measured,
because the instrument fails the answers the lesson itself supplies.

**MECHANISM.** The capture group is `([\d,]+\.?\d*)` — the decimal point is
optional *and* the digits after it are optional, so the group matches `18` and
then takes the full stop as well. Every system prompt in the lesson ends with
"End with: 'The answer is [number]'", and a model that writes a sentence scores
zero.

**FINDING: the damage is technique-specific.** Over 16 labelled reply shapes:

| technique | graded right |
|---|---|
| `zero_shot` | **6 / 6** |
| `zero_shot_cot` | 2 / 4 |
| `few_shot_cot` | **0 / 6** |

`re.search` also returns the *first* match, so `"Initially the answer is 60, but
the answer is 72."` grades as **60**. Self-correction is exactly what
chain-of-thought produces, so the arm the exercise expects to win is the arm the
grader punishes.

**FINDING: `GSM8K_EXAMPLES[3]` is `TEST_QUESTIONS[4]`.** Same question, answer
attached. At the default `num_examples=3` it sits outside `examples[:3]`; at 4
or more, the few-shot arms are shown the answer to a question they are scored
on.

**CONTROL: two edits.** Require digits after the point, `([\d,]+(?:\.\d+)?)`,
and take the last match rather than the first. The exemplars then score 5/5, the
corpus 16/16, and the self-correcting reply resolves to 72.

### 2 — the pool is five, and picking well retrieves the test question

`build_cot_prompt` takes `examples[:num_examples]` — a slice, not a choice. So
there is no selection hook, and every strategy is a reordering of the pool.

| exemplar | reasoning chars | backtracking markers | question marks |
|---|---:|---:|---:|
| 1 Janet | 175 | 0 | 0 |
| 2 robe | 109 | 0 | 0 |
| 3 Josh | 233 | 0 | 0 |
| 4 James | 200 | 0 | 0 |
| **5 Wendi** | **1,356** | **5** | **2** |

Exemplar 5 is the only one that models visible self-correction — "Wait, let me
re-read", "Actually", "Hmm", "She needs to give negative? No." — which is the
reply shape exercise 1 showed the extractor mis-parses.

**FINDING: random selection draws it 6 times in 10.** All ten 3-subsets of a
5-item pool, enumerated rather than sampled:

| subsets | mean prompt chars |
|---|---:|
| the 6 containing exemplar 5 | 2,566 |
| the 4 not containing it | 1,098 |

**2.34×** for the same three slots. The variance of the "random" arm is one
exemplar wide.

**FINDING: hand-picking well retrieves the test question.** Similarity
(Jaccard over shared operation words) of the best exemplar, per test question:

```text
[0.33, 0.00, 0.20, 0.33, 1.00]
             ^^^^              a rate problem with no analogue in the pool
                         ^^^^  its nearest exemplar is itself, answer attached
```

On five examples the quality arm either leaks or abstains.

**ANSWER: quality dominates from the fourth example.**

| k | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| prompt chars | 460 | 699 | 1,132 | **1,462** | **3,293** |

Quantity saturates at 5. `k=4` admits the duplicate of a test question; `k=5`
costs **+125%** for the one broken exemplar. Above `k=3` there is no count at
which more is better — which is a fact about this pool, not about few-shot
prompting.

### 3 — the cost is a straight line, and the vote splits on a full stop

| N | total tokens | per sample |
|---:|---:|---:|
| 1 | 482 | 482 |
| 3 | 1,446 | 482 |
| 5 | 2,410 | 482 |
| 7 | 3,374 | 482 |
| 10 | 4,820 | 482 |

**ANSWER: there is no knee.** `self_consistency_solve` rebuilds one prompt and
re-sends it whole N times — no shared prefix, no discount — so cost is linear in
N through the origin. Any knee in the exercise's plot lives on the accuracy
axis, which needs the model the exercise assumes.

**FINDING: the prompt is 92.7% of every call.** 447 prompt tokens (`cl100k_base`,
3 exemplars) against a 35-token completion. Ten samples of one question spend
4,470 prompt tokens to buy 350 tokens of reasoning. And the lesson ships 5 test
questions, not the 20 the exercise says to run.

**FINDING: self-consistency splits its own vote on a full stop.** Ten samples
that all answer 72, five of them ending the sentence:

```text
votes      {'72.': 5, '72': 5}
answer     '72.'          <- Counter.most_common(1) is insertion-ordered
confidence 0.50           <- unanimity, scored as a coin flip
```

**FINDING: confidence is computed over parsed answers, not samples.** Eight
replies that parse to `None` and two that agree:

```text
votes      {'72': 2}
confidence 1.00
```

`solve_with_escalation` escalates below 0.80. So it skips Tree-of-Thought on the
run where 80% of the samples failed, and fires it on the run where every sample
agreed.

### 4 — the loop already ships, and it checks Answer before Action

`react_solve` *is* the exercise: calculator, `eval`, observation fed back. Four
scripted replies, one per branch:

| reply | calls | observations | returned |
|---|---:|---:|---|
| Thought + Action + Answer | **1** | **0** | `'72'` |
| Thought + Action, then Answer | 2 | 1 (`Observation: 72.0`) | `'72.'` |
| Action that raises | 2 | 1 (`Observation: Error - division by zero`) | `'5'` |
| Thought only | **5** | 0 | **`'3'`** |

**ANSWER: the tool never fires for a model that also states its answer.** The
loop tests `Answer:` before `Action:` and returns on the first hit. A reply
holding both — which is what a capable model writes — ends the run in one call
with zero observations, and the arm labelled "tool-grounded" ran pure CoT.

**FINDING: the tool returns a float.** `eval("48 + 48/2")` is `72.0` under true
division, so the observation reads `72.0`. A model echoing its own tool output
answers `"72.0"` against an expected `"72"`.

**FINDING: "(in a sandbox)" is not one.** Under `eval(expr, {"__builtins__":
{}}, {})` — the lesson's guard verbatim —
`().__class__.__mro__[1].__subclasses__()` evaluates to roughly 600 classes, and
`9**9**9` is a syntactically valid expression the guard has no view of. The
guard removes names, not capability; a calculator needs a restricted grammar.

**FINDING: a turn with neither Action nor Answer is a silent five-step no-op.**
Nothing is appended to the conversation, so the loop re-sends the same messages
and at temperature 0 the model repeats. The fallback then runs `extract_answer`
over the concatenated assistant text and returns `'3'` — the last number in the
words "the 3 parts of this". A run that never answered returns a number.

**ANSWER: the three exits are indistinguishable from the return value.** All
return `(str, str)`, and the tool-grounded one's text is the final assistant
turn, which holds no observation. Whether the calculator ran lives only in the
client's call log, and the client is not returned.

### 5 — the beam discards its best node, and returns a number for a story

```text
extract_answer("For sale: baby shoes, never worn.")   ->  ''
extract_answer("He proposed. She laughed. Then cried.") ->  None
extract_answer("Born 1947. Died 1947. Lived anyway.")  ->  '1947.'
```

**ANSWER: the adapted solver returns a number, or the empty string.**
`tree_of_thought_solve` ends with `extract_answer(best_thought)`. Two of the
three stories come back as `''`, because `[\d,]+` matches the comma in the
prose — and `''` is not `None`, so exercise 3's vote counts it as an answer. The
story survives only in the second return value, which `solve_with_escalation`
throws away.

**FINDING: the search replaces its frontier instead of extending it.**

```python
if next_thoughts:
    scored = sorted(next_thoughts, key=lambda x: x[1], reverse=True)   # children only
```

With an evaluator scoring the three initial stories **0.95** and every
continuation **0.40**, the function returns a 0.40 continuation. The beam
carries no incumbent, so the best node found can be discarded at any depth.

**FINDING: the evaluator's parser reads the first number and clamps up.**

| evaluator reply | intended | measured |
|---|---:|---:|
| `0.9` | 0.9 | 0.9 |
| `Score: 0.9` | 0.9 | 0.9 |
| `0.4` | 0.4 | 0.4 |
| `Approach #2. Score: 0.9` | 0.9 | **1.0** |
| `Score: 8/10` | 0.8 | **1.0** |
| `Excellent, but not six words.` | 0.5 | 0.5 |

Both errors land on the clamp maximum, so a formatting slip outranks every
honest score.

**FINDING: no number scores 0.5, and 0.5 ties.** `float(".")` raises and the
except branch returns 0.5, so under a silent evaluator every thought ties,
`sorted` is stable, and the lineage returned is the first generated. Branching
becomes single-shot with extra steps.

**ANSWER: branching costs 30 calls against 1** — 3 generations, 15 evaluations,
12 extensions at breadth 3 and depth 3. Whether the extra 29 bought a better
story cannot be read off the return value, which is a number extracted from
prose.
