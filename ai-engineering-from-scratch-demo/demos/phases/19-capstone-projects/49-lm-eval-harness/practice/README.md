<!-- generated:start -->
# 19-capstone-projects / 49-lm-eval-harness

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/49-lm-eval-harness/) · upstream spec
`phases/19-capstone-projects/49-lm-eval-harness/docs/en.md`

```bash
uv run demo practice run 49-lm-eval-harness --ex 1
uv run demo explain 49-lm-eval-harness --ex 1
uv run pytest demos/phases/19-capstone-projects/49-lm-eval-harness
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a sixth task with a custom metric you write from scratch (BLEU-like overlap, BLEURT-like… | code | T0 | `ex01_the_toy_adapter_echoes_a_sixth_task_for_0_843_and_the_runner_accepts_a_task_score_of_1_5.py` |
| 2 | Extend `code_exec` to capture stdout and accept a list of expected stdouts as targets. | code | T0 | `ex02_the_shipped_code_exec_scores_a_correct_recursive_function_0_and_a_prediction_can_still_reach_os.py` |
| 3 | Add a leaderboard diff command: given two `leaderboard.json` files, print which tasks moved a… | code | T0 | `ex03_the_committed_leaderboard_names_adapter_arithmetic_and_dropping_a_task_moves_echo_overall_0_328_to_0_160.py` |
| 4 | Cap latency per example. Wrap the adapter call in a timeout; surface a separate `timeouts` co… | code | T0 | `ex04_capping_the_batched_call_turns_2_hung_prompts_into_8_timeouts_and_the_hung_threads_keep_running.py` |
| 5 | Pin task content with a sha256 in the leaderboard so a future reader can verify they scored t… | code | T0 | `ex05_a_comment_line_changes_the_file_hash_but_not_the_scores_and_task_pins_miss_a_swapped_metric.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: five JSONL tasks with five
examples each, five metrics, a batched runner, and a `ToyAdapter` that
pattern-matches the fixture prompts. All tasks are seeded into a temporary
directory, so nothing is written into the lesson. Everything here is stdlib
and T0. BLEU was checked against Papineni et al. 2002
(https://aclanthology.org/P02-1040.pdf), read on 2026-09-29.

### 1 — the toy adapter echoes a sixth task for 0.843, and the runner accepts a task score of 1.5

**The sixth task is `paraphrase`: five `restate:` prompts with two
references each, scored by `bleu2`, a from-scratch BLEU-2.** It uses
clipped 1- and 2-gram precision, add-one smoothing on the bigram
precision, and the brevity penalty against the closest reference length.
The contract holds: identical text scores 1.0, empty text 0.0, "the the the
the" 0.25 (clipping), and a one-word answer 0.007 (brevity penalty), with
every score in [0, 1]. The runner picks up `paraphrase.jsonl` with no
change, and a prefix-stripping adapter scores 1.0 on all six tasks.

**The lesson's `ToyAdapter` echoes any prompt it has no rule for.** On
`paraphrase` that scores 0.843, listed as 4/5 correct although none is
exact, and the toy's overall drops from 1.0 to 0.974. Pure echo scores
0.328 across the five shipped tasks, and a full 1.0 on `generation`, where
every target is a word from the prompt.

**The runner does not check the metric contract.** A metric that returns
1.5 gives an overall of 1.083. `correct` is `round(sum of scores)`, so
echo's 0.639 on `summary` shows up as 3/5 correct.

### 2 — the shipped code_exec scores a correct recursive function 0, and a prediction can still reach os

**`code_exec_stdout` adds a buffered `print` to the stripped namespace and
scores stdout against `targets`.** With `extras["inputs"]`, each `f(x)` is
compared to its own target. Without inputs, the whole program's stdout must
match any target. It still scores `io_pairs` by return value.

| prediction | shipped `code_exec` | `code_exec_stdout` |
|---|---:|---:|
| FizzBuzz, per-call stdout targets | n/a | 1.0 |
| FizzBuzz that prints "Fizz" on 5 | n/a | 0.667 |
| `print('hello')` vs ["hi", "hello"] | n/a | 1.0 |
| the 5 shipped fixtures (toy answers) | 1.0 each | 1.0 each |
| a doubler that also prints | 0.0 | 1.0 |
| recursive factorial | 0.0 | 1.0 |
| `f` calling a helper `g` | 0.0 | 1.0 |
| `__subclasses__` walk to `os.getcwd()` | 1.0 | 1.0 |

Three of those rows are findings about the shipped metric. It has no
`print`. It runs `exec` with separate globals and locals, so top-level
names are invisible inside functions and correct recursion or helper calls
score 0. And stripped builtins are not a sandbox: a prediction that walks
`().__class__.__base__.__subclasses__()` to `os._wrap_close` reaches `os`,
against the lesson's "cannot reach the filesystem".

### 3 — the committed leaderboard names adapter "arithmetic", and dropping a task moves echo's overall from 0.328 to 0.160

**`diff_boards(old, new)` joins on task name. Each task is reported as
moved (with its delta), unchanged, added or removed, alongside the overall
delta and any adapter change.** Run the file with two paths and it prints
the table. Toy against echo:

| task | delta |
|---|---:|
| arithmetic | -1.000 |
| code-exec | -1.000 |
| multiple-choice | -1.000 |
| summary | -0.361 |
| generation | unchanged |
| overall | -0.672 |

The lesson's committed `outputs/leaderboard.json` says `"adapter":
"arithmetic"`, but `main.py` only ever writes `toy.v1`. Against a fresh run
its scores match on 5/5 tasks, while `latency_ms` differs on 5/5, so a diff
has to ignore latency and timestamp, or every re-run counts as a move.
Removing `generation.jsonl` moves none of the four remaining tasks, yet
echo's overall falls from 0.328 to 0.160. The overall is an unweighted mean
over whichever tasks are present.

### 4 — capping the batched call turns 2 hung prompts into 8 timeouts, and the hung threads keep running

**Each `adapter.generate` call runs in a thread joined with a 0.5 s cap.
A call that misses the cap scores as empty and counts in a `timeouts`
field, which is added to each row of the JSON the lesson's
`write_leaderboard` writes.** Two prompts (`arith-01`, `code-02`) block
forever. The toy otherwise answers in microseconds, so the result does not
depend on timing.

| wrap | timeouts (arith, code, gen, mc, sum) | arithmetic | code-exec | overall |
|---|---|---:|---:|---:|
| each example | 1, 1, 0, 0, 0 | 0.8 | 0.8 | 0.92 |
| each batch of 4 (`main.py` default) | 4, 4, 0, 0, 0 | 0.2 | 0.2 | 0.68 |

A timeout only stops the wait, not the call: both hung threads are still
alive afterwards. The adapter cap also does not cover the metric.
`metric_code_exec` runs predictions in-process, and a `while True: pass`
prediction is only stopped by killing a subprocess, here at 5 s after the
module has loaded. A correct prediction in the same subprocess prints 1.0.

### 5 — a comment line changes the file hash but not the scores, and task pins miss a swapped metric

**Each task is pinned by the sha256 of its loaded `Example` records
(sorted-key compact JSON). A `tasks_sha256` over the sorted `name:hash`
lines covers the set.** `check_pins` reloads a directory and lists the
tasks that no longer match. A board pinned on freshly seeded tasks gives 0
of 5 mismatches against them and against the lesson's committed
`outputs/tasks/`, which are byte-identical to `seed_fixture_tasks` output.

| edit | file sha256 | pin mismatches | score |
|---|---|---|---|
| comment + blank line in arithmetic | changed | none | 1.0 |
| arithmetic target "25.0" -> "25" | changed | arithmetic | 1.0 -> 0.8 |
| `summary` scored by exact_match | unchanged | none | 1.0 -> 0.0 |

Hashing the loaded records rather than the raw bytes is what lets
contributors annotate files, as the doc invites. The lesson's committed
`leaderboard.json` has no hash field, so the target edit would read as a
model regression. The last row is the limit of the exercise: pins cover the
task data, not the metric code. A full pin would also hash the harness
source.
