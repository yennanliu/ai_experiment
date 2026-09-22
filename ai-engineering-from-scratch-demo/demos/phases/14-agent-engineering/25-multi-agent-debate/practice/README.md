<!-- generated:start -->
# 14-agent-engineering / 25-multi-agent-debate

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/25-multi-agent-debate/) · upstream spec
`phases/14-agent-engineering/25-multi-agent-debate/docs/en.md`

```bash
uv run demo practice run 25-multi-agent-debate --ex 1
uv run demo explain 25-multi-agent-debate --ex 1
uv run pytest demos/phases/14-agent-engineering/25-multi-agent-debate
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement a "forced disagreement" rule: in round 1, every debater must produce a distinct pro… | code | T0 | `ex01_the_shipped_debaters_agree_before_the_first_round.py` |
| 2 | Add a confidence-weighted aggregation: debaters return (answer, confidence); aggregator weigh… | code | T0 | `ex02_the_aggregator_already_breaks_ties_by_list_position.py` |
| 3 | Swap one "agent" for a different scripted LLM with different opinions. Does heterogeneity imp… | code | T0 | `ex03_a_lone_correct_debater_is_outvoted_before_it_can_argue.py` |
| 4 | Measure token cost for full mesh vs sparse on your 3 questions. Plot cost vs accuracy. | code | T0 | `ex04_the_lesson_counts_the_hub_reads_and_the_code_pays_for_them.py` |
| 5 | Read the Society of Minds paper. Port your toy to N=5, R=3. What breaks? What gets better? | code | T0 | `ex05_the_hub_is_debaters_zero_and_the_tie_break_is_the_list.py` |
<!-- generated:end -->

## Answers

### 1 — the shipped debaters agree before the first round

`_make_debater` reads `corrections` before `bias`, and beta and gamma carry a
correction for every question in the demo. So all three produce the same answer
at round 0 on all three questions, and across three questions and two topologies
the number of answers that ever change is **zero**. The 18 full-mesh and 12 star
critique ops buy exactly the output that 0 ops would. Measuring the effect of
forced disagreement first requires a baseline that debates, and this is not one.

Forcing distinct round-1 proposals does not speed convergence up — it stops it.
`capital_of_portugal` converges at round 3; the other two never converge within
three rounds. With three debaters holding three distinct answers, every debater's
two peers disagree, `Counter.most_common` breaks the tie by insertion order, and
the answers rotate instead of settling. The majority answer is still correct 3/3,
so on this panel forced disagreement cost convergence and bought nothing. That is
the honest answer to "measure effect on convergence speed", and it is the
opposite of the intuition the exercise is set up to test.

The one question that *does* converge converges by accident. `drift` adopts the
peer majority only `if common != current and common != bias`, so beta — holding
`"Lisbon"` by correction and `"Madrid"` by bias — stays on Lisbon against
unanimous Madrid, and moves to Porto against unanimous Porto. It accepts any
consensus except its own bias. That clause is what breaks the rotation on
`capital_of_portugal`, where the three biases happen to be the three answers.

And convergence does not stop the spending: `run_debate` loops its full round
count whatever happens, so 12 of the baseline's 18 ops are spent after consensus.
`converged_round` is recorded and never acted on, which is the shape of a system
that reports a cost saving it does not take.

### 2 — the shipped aggregator is already weighted, by seat number

`run_debate` ends with `Counter(prior.values()).most_common(1)[0][0]`. On a
1-1-1 split that returns the answer inserted first, so over twelve three-way
disagreements the ties go `{alpha: 12, beta: 0, gamma: 0}` — and reversing the
debater list changes the final answer 12 of 12 times with nobody changing their
mind. Before any confidence is added, the aggregator is already weighted; the
weight is list position.

Does confidence weighting help? With the truth rotating through the panel so no
seat is favoured, plurality is right 4 times of 12 and calibrated confidence
weighting 10. That is a real improvement, and the condition on it is the whole
point: weighting by one debater's *fixed overconfidence* scores 4 — and returns
the same answer as plurality on 12 of 12. A miscalibrated weight does not
degrade gracefully into a vote; it reproduces exactly the bias it was brought in
to replace. Confidence weighting is a transfer of authority to a number, and the
number has to be earned before the transfer is an improvement.

Two mechanical notes. `Debater` has two fields and `drift` returns a bare `str`,
so the widened `(answer, confidence)` signature is a new type rather than an
extra argument, and three of the module's five functions read the answer as a
string. And on the lesson's own questions the panel agrees, so plurality and both
weighting schemes return one distinct answer between them: an aggregator can only
matter where there is disagreement, which the shipped demo never produces.

### 3 — a lone correct debater is outvoted before it can argue

Swapping one agent for a genuinely different source — right on eight questions
where the two incumbents agree on a wrong answer — does not improve accuracy. The
expert adopts the majority in round 1 on 8 of 8, and the debate scores 0, exactly
what a homogeneous wrong panel scores. The swap changes who is right and nothing
else.

The reason is that `drift` is majority-following with no notion of evidence: it
compares the peer majority against its own answer and adopts it whenever they
differ. No confidence, no justification, no tie-break but list order. Two peers
outrank one debater regardless of which is correct. The lesson's claim that
cross-model combinations beat single-model debates depends on an update rule that
can weigh an argument, and this one cannot.

The expert is also memoryless, which is worth seeing directly. Its answers over
the debate run `['Herbert', 'Asimov', 'Asimov', 'Asimov']` — but asked again with
no peers it still says `'Herbert'`. `drift` recomputes `current` from
`corrections` on every call and never reads its own last answer. The debate
changed what delta *said* and nothing about what delta *holds*, which means there
is no state for an argument to move.

The sharpest result is the sweep. Running 1 to 4 experts against 2 incumbents
gives, as (debate, plain vote), `{1: (0, 0), 2: (8, 0), 3: (0, 8), 4: (8, 8)}`.
The plain vote is monotone and flips exactly when the experts outnumber the
incumbents — correct behaviour. The debate goes **0, 8, 0, 8**: right on a 2-2
tie and *wrong on a 3-2 expert majority*. Adding a third correct agent made the
system worse. Accuracy that is not monotone in the number of correct agents is
not an aggregation rule, it is a lottery whose seed is the list order.

### 4 — the star's published cost is half what the code charges

On the shipped three questions, full mesh costs 6 ops per round and the star 4,
so the table is `(18, 3/3)` against `(12, 3/3)` — 33.3% cheaper at identical
accuracy. That reproduces the lesson's claim, on questions where nobody
disagrees, so it is a statement about arithmetic rather than about debate.

Checking the arithmetic against the lesson's own worked example is where it gets
interesting. For N=5, R=3 the prose gives 60 mesh ops and "spokes read only the
hub = 12 critique ops" for the star. The mesh number matches `full_mesh_round`
exactly. The star number does not: `sparse_star_round` charges the hub's 4 reads
*as well as* the 4 spoke reads, 8 per round, 24 over three. So the star's
advantage is **2.5x in the code and 5.0x in the prose**. Both are defensible
models — a hub that reads all four spokes is doing real work, and a hub that only
broadcasts is not — but they are different systems, and the shipped one is the
expensive one.

Ops are also not tokens. Weighting each read by the length of what is read — the
mesh's reads are all full-length peer answers, the star's include short hub
summaries — gives a character ratio of 3.1x against the op ratio's 2.5x. Here the
op count *understates* the star's advantage. Neither number is wrong; they answer
different questions, and only one of them is what a provider bills. Which
direction the correction runs depends entirely on the relative answer lengths, so
the proxy has to be re-checked per workload rather than assumed.

Finally, the plot the exercise asks for is a vertical line: across six
configurations every run returns the correct answer, giving one distinct accuracy
value. A cost-accuracy curve needs a question the panel can get wrong, and this
panel converges on the truth before round 1.

### 5 — N=5, R=3 is four lines, and it changes what decides the answer

Porting is a parameter change: 60 mesh ops and 24 star ops against 12 for the
paper's N=3, R=2. Both panels answer 8 of 8, so the extra 5.0x buys nothing here.
That is not a refutation of Society of Minds — its gains come from model
instances that genuinely propose differently on hard problems, and these scripted
debaters produce one distinct proposal between them (exercise 1). It is a
statement about what this toy can demonstrate.

**What breaks.** At N=5 every debater reads four peers, so a 2-2 split among them
is resolved by `Counter.most_common` taking the earliest debater in the list.
Reversing the panel changes the final answer on 8 of 8 questions with nobody
changing their mind; at N=3 the same reversal changes 0, because two peers can
only tie by agreeing. Going from N=3 to N=5 turns list order from an irrelevance
into the decider — which is exactly why the non-monotonicity in exercise 3
appeared at three experts and not at two.

**What also breaks.** `run_debate` computes `hub = debaters[0]` and
`spokes = debaters[1:]`, so the star's hub is whoever was listed first. Rotating
the hub through all five positions produces two distinct answers on the same
question and the same panel. The lesson's own mitigation for hub failure —
"rotate or use multiple hubs" — has no parameter to express it, and expressing it
means rotating the input list, which is a coincidence rather than an interface.

**And the cost of a bad hub scales the wrong way.** With four correct spokes and
one wrong hub, the star scores 0 of 8, because every spoke reads only the hub.
The same panel in full mesh scores 8 of 8. Sparsity concentrates the blast radius
in exactly the topology the lesson recommends for cost, and at N=5 a single bad
hub corrupts four debaters in one round against two at N=3.

**What gets better.** Honestly: on this toy, nothing measurable. The paper's
result holds where independent proposals differ, and the useful thing the port
exposes is that the shipped update rule and hub selection were load-bearing
decisions disguised as defaults, and only become visible once N is large enough
for ties to happen.
