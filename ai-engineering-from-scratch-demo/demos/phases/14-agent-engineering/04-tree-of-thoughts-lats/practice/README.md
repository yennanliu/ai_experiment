<!-- generated:start -->
# 14-agent-engineering / 04-tree-of-thoughts-lats

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/04-tree-of-thoughts-lats/) · upstream spec
`phases/14-agent-engineering/04-tree-of-thoughts-lats/docs/en.md`

```bash
uv run demo practice run 04-tree-of-thoughts-lats --ex 1
uv run demo explain 04-tree-of-thoughts-lats --ex 1
uv run pytest demos/phases/14-agent-engineering/04-tree-of-thoughts-lats
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run the toy LATS with UCT c=0.1 vs c=2.0. What changes in the trace? | code | T0 | `ex01_c_is_not_a_parameter_and_the_answer_is_not_a_search_result.py` |
| 2 | Swap the value function for a noisier scorer (add random jitter). Does MCTS still find the be… | code | T0 | `ex02_a_little_noise_helps_because_the_value_function_is_wrong.py` |
| 3 | Implement beam-search ToT (keep top-k at each level) and compare to BFS. Which is better on a… | code | T0 | `ex03_the_beam_is_already_there_and_38_wide_is_free.py` |
| 4 | Read LATS Section 5.1. Reproduce the HumanEval trajectory count: how many rollouts does it ta… | explain | T0 | prose, below |
| 5 | Read the LATS paper's discussion on "when LATS helps less." Write a one-paragraph decision ru… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Three ship code; exercises 4 and 5 ask for
readings of the LATS paper, so they are answered in prose.

The recurring finding is that **the value function is not noisy, it is
wrong**, and every exercise in the lesson ends up measuring the difference.
`value` scores a partial state by the closest remaining number to 24, so
`6*4=24` scores **0.0** — the best any non-terminal state can score — and is
a dead end, because the leftover `4` and `1` cannot be removed. Both real
solutions start at rank **7** and rank **20** of the root's **24** children.

That one fact sets the shape of all three code answers. Widening the ToT beam
from 4 to 32 changes the cost from **88** to **438** expansions and changes
the answer not at all; only width **38** — the winning prefix's rank at level
2 — flips it, and by then the beam costs exactly what unpruned BFS costs.
Adding Gaussian jitter to the scorer at a tight budget *raises* the win rate
from **19/30** to **25/30**, because blurring a wrong ranking is help. And
the UCT constant, swept from 0.1 to 2.0, changes which nodes exist and never
changes which node is returned, because `mcts` closes with
`max(_all_leaves(root), key=value)`.

What holds: the search machinery itself is correct. Kahn-free MCTS,
UCT, backprop, and a BFS that early-returns on a solved child all behave as
written. The gap is that none of them can be better than the function they
are ranking with.

### 1 — `c` is not a parameter, and the answer is not a search result

**ANSWER: at the demo's budget, almost nothing changes.** `c=0.1` and
`c=2.0` build the same **311** nodes in the same shape, expand **286** times
each, and disagree on the visit count of **15** of them. Both return
`['6*4=24', '24*1=24']` scoring **0.0**.

**FINDING: `c` has no way into the search.** `mcts(root, iterations, rng)`
has no exploration constant; `uct(parent, child, c=1.4)` has it as a keyword
default, resolved through the module globals at call time. Running this
exercise at all means rebinding `ref.uct`.

**FINDING: raise the budget and the trees diverge — the answer does not.** At
**300** iterations the two runs build **355** and **322** nodes, expand
**330** against **297** times, and differ on **168** visit counts. Both
return `['6-1=5', '5*4=20', '20+4=24']` scoring **1.0**. The last line of
`mcts` is `max(_all_leaves(root), key=value)`: visits and Q decide which
nodes get *created* and never decide which one is *returned*. That is worth
holding onto — it means every tuning knob in this implementation acts only
through coverage.

**FINDING: at 80 iterations not one trajectory is finished.** **0** of
**286** leaves are depth-3 states, while the instance has **4** winning paths.
The demo's headline answer is an unfinished path whose score of **0.0** comes
from having 24 among its leftovers. The printed trace looks like a result and
is a frontier.

### 2 — a little noise helps, because the value function is wrong

**ANSWER: yes, down to SNR 4.** The signal is exact — a solved state scores
**1.0**, everything else at most **0.0** — so the gap is **1.0** and
SNR = `1.0 / sigma`. Over **30** seeds at **1000** iterations: **30/30** at
sigma 0, 0.1 and 0.25; **17/30** at 0.5; **3/30** at 1.0. The tolerated
minimum is **SNR 4**. At SNR 2 it is a coin flip and at SNR 1 it is gone.

**FINDING: at a tight budget, jitter improves the search.** At **300**
iterations sigma **0** wins **19/30** and sigma **0.1** wins **25/30**. Noise
is only harmful relative to a ranking worth preserving, and this ranking puts
a dead end first. A real system cannot rely on that — the point is the
diagnostic: if adding noise to your evaluator helps, the evaluator is biased,
not imprecise.

**FINDING: the damage is in the final pick, not in the rollouts.** At sigma
**0.5** and 300 iterations, perturbing only the closing argmax wins
**9/30**, while perturbing only the rollouts wins **21/30** against a clean
**19/30**. The rollouts average over many samples through backprop; the close
is a single unweighted `max` over hundreds of independently perturbed scores,
so its error probability *grows with the size of the tree*. More search makes
that specific step worse.

**FINDING: the failures are confident.** At sigma **1.0** and 1000
iterations, **25** of the **27** losses return a complete depth-3 trajectory
that is simply wrong. This is the lesson's warning made concrete: a noisy
evaluator does not make search give up, it makes search produce a finished,
plausible, incorrect answer — the worst possible failure mode for a pattern
sold on correctness.

### 3 — the beam is already there, and 38 wide is free

**ANSWER: `tot_bfs` *is* beam search.** It sorts each level by `value` and
keeps `scored[:max_expansions_per_level]`, so the exercise's beam is the
shipped code and the arm to implement is unpruned BFS. At the shipped width
**8** the answer scores **-0.03** after **152** expansions. The narrowest
width that finds 24 is **38**, costing **460** expansions — exactly what
unpruned BFS costs, because the `value > 0.99` early return fires at the same
node.

**Which is better on a tight token budget?** Neither, and that is the useful
answer. The beam is cheaper only while it is wrong. At every budget below
**460** expansions it returns a confidently wrong trace; at **460** it is
indistinguishable from BFS. On this instance, beam width buys you the
*option* of being wrong more cheaply.

**FINDING: widths 4 through 32 buy nothing at all.** All four return the
identical trace `['6*4=24', '4-1=3', '24+3=27']` while the cost rises from
**88** to **438** expansions — five times the budget, zero change in output.
Every candidate the wider beam retains is worse under the same ranking that
was already pointing the wrong way.

**FINDING: the threshold is the winning prefix's rank, exactly.** Scoring
every node of each unpruned level, the winning path sits at rank **7** of
**24**, then **38** of **286**, then **1** of **1129**. The maximum over the
pruned levels is **38**; width **37** answers **-0.01** and width **38**
answers **1.0**. There is no slack and no luck in it — beam width is a
statement about how badly you are willing to let the heuristic be wrong.

### 4 — the rollout count is a budget, not a constant

*Cites "LATS (Zhou et al., ICML 2024)".*

**The question has no single number, and the reason is the evaluator.**

LATS on HumanEval uses the unit tests as the value function. That changes the
accounting completely: the search does not run a fixed number of rollouts and
report the best: it runs until a candidate *passes*, and then stops. So the
per-problem rollout count is a distribution, not a constant, and the
distribution's mass sits at **1** — a problem the base policy already solves
is solved by the first trajectory, and search never engages. The lesson's own
headline, `pass@1 92.7% with GPT-4`, is therefore composed of two very
different populations: the problems GPT-4 solved on trajectory 1, which were
already the base model's pass@1, and the residual, where the search pays.
Reporting a mean over that mixture would describe neither.

That is the conceptual reproduction the exercise is asking for, and this
lesson's code reproduces the *shape* of it exactly, at a scale one can count.
Exercise 1: at **80** iterations `mcts` has completed **0** trajectories and
returns an unfinished frontier node; at **300** it has completed **31** and
returns a true solution. Exercise 3: at beam width **37** the answer is wrong
and at **38** it is right, and the cost of being right is **460** expansions
whether you prune or not. In both cases the cost is a *threshold* — the
budget at which the first passing candidate becomes reachable — and every
budget below it produces a confidently wrong answer rather than no answer.

Three things follow, and they are what the paper's Section 5 discussion is
about rather than any particular integer:

1. **The right number to report is expansions-to-first-pass, per problem,
   with its tail.** A mean rollout count hides that most problems cost 1 and
   a few cost two orders of magnitude more. The lesson's "100–1000x the
   tokens of CoT" is a statement about that tail, not about the average
   problem.
2. **A machine-checkable evaluator is what makes the tail affordable.** Unit
   tests give a signal gap like the one exercise 2 measures — pass or fail,
   no ambiguity, SNR effectively infinite — so extra rollouts monotonically
   help. Exercise 2 shows what happens when the gap shrinks: at SNR 2 the
   search is a coin flip *however long you run it*.
3. **The honest limit of this reproduction.** The lesson's toy has no model
   and no HumanEval, so an exact trajectory count from the paper's Section
   5.1 cannot be recomputed here, and none is asserted. What can be asserted,
   and is measured above, is that the count is a budget threshold set by the
   evaluator's fidelity and the branching factor — which is the part that
   transfers to a system you are actually building.

### 5 — search pays when the evaluator is cheaper than the policy

*Cites "The cost reality".*

**The decision rule, in one paragraph.** Reach for search only when three
things are true at once: a single trajectory is demonstrably insufficient (if
the base policy already solves the task most of the time, search only taxes
the cases that were already working); the evaluator is *cheap relative to the
policy*, because search's cost is evaluator calls and its benefit is policy
calls avoided; and — the axis the lesson's phrasing understates — the
evaluator is **unbiased**, not merely precise. Noise and bias fail
differently and are fixed differently. A noisy-but-unbiased evaluator degrades
gracefully and is repaired by more search: exercise 2 measures **30/30**
success down to **SNR 4**, and the failure at SNR 2 is recovered by widening
the sample. A *biased* evaluator is not repaired by more search at all:
exercise 3 measures a ranking whose top choice is a dead end, and no beam
width between 4 and 32 changes the answer while the cost rises fivefold; the
only width that works is the one at which pruning has stopped happening.
So the rule is: if your evaluator is noisy, buy more search; if your
evaluator is wrong, buy a better evaluator, because search will spend your
budget confirming its prejudice and hand you a complete, well-formed, wrong
answer — **25** of **27** times, in exercise 2's worst case.

| Task shape | Strategy | Why |
|---|---|---|
| One trajectory usually succeeds; failures are tool or format errors | ReAct + tool-grounded verification (Lesson 05) | Search taxes the 90% that already worked; verification costs one extra call on the 10% |
| Machine-checkable success, expensive policy, heavy tail (code, math) | LATS / MCTS with the checker as the value function | The evaluator is free relative to the model, the gap is exact, and the cost concentrates on the residual |
| Combinatorial state, cheap symbolic scorer that is *correlated* with success | ToT beam, width set from the winning prefix's rank | Exercise 3: the threshold is a property of the heuristic, not a tuning preference |
| Cheap scorer that is *anti*-correlated near the root | Nothing, until the scorer is fixed | Widths 4–32 cost 5x and return the identical wrong answer |
| No ground truth, self-eval only | ReAct, with reflection across trials (Lesson 03) | SNR below ~2 makes the closing argmax a coin flip; verbal feedback across episodes is cheaper than search within one |
| Machine-checkable fitness, very long horizon, offline | Evolutionary search (AlphaEvolve, Lesson 11) | Same evaluator argument, taken to its limit: the checker runs millions of times and the policy runs rarely |

**The one-line version:** search converts evaluator quality into answer
quality, at a token cost set by the branching factor. If the conversion rate
is negative — and exercise 3 shows it can be — the only correct amount of
search is none.
