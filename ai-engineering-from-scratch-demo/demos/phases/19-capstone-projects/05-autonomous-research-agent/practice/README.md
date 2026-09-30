<!-- generated:start -->
# 19-capstone-projects / 05-autonomous-research-agent

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/05-autonomous-research-agent/) · upstream spec
`phases/19-capstone-projects/05-autonomous-research-agent/docs/en.md`

```bash
uv run demo practice run 05-autonomous-research-agent --ex 1
uv run demo explain 05-autonomous-research-agent --ex 1
uv run pytest demos/phases/19-capstone-projects/05-autonomous-research-agent
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run the pipeline against three different seed ideas in the same domain. Compare which parts o… | code | T0 | `ex01_three_seed_ideas_grow_one_identical_tree_and_76pct_of_their_spend_reruns_a_config.py` |
| 2 | Add a human-in-the-loop gate before experiment execution for nodes estimated above $5. Measur… | code | T0 | `ex02_a_5_dollar_approval_gate_never_fires_because_no_node_can_cost_more_than_1_60.py` |
| 3 | Swap the reviewer ensemble for a single judge. Measure the false-accept rate on a held-out se… | code | T0 | `ex03_a_single_judge_false_accepts_70pct_of_control_rerun_papers_and_the_ensemble_85pct.py` |
| 4 | Introduce a network-exfiltration red team test: agent writes code that tries to `curl` an ext… | code | T0 | `ex04_curl_is_blocked_and_logged_4_of_4_ways_but_the_lessons_sandbox_is_a_docstring.py` |
| 5 | Compare your tree-search with a flat random baseline (same budget, no expansion strategy). Re… | code | T0 | `ex05_the_tree_search_is_a_fifo_queue_and_loses_7_3pct_to_flat_random_at_the_same_30_dollars.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: a best-first tree search
over experiment nodes in which `expand` makes one config edit per child,
`run_experiment` is a stub that fabricates a loss, a cost and a novelty
from an rng, and the writer, reviewer and red team are a printed
placeholder. `main()` spends $5.57 of its $30 on 4 experiments.

### 1 — three seed ideas grow one identical tree, and 76% of their spend reruns a config

**The three trees overlap completely.** The seed text is stored in
`root.hypothesis` and nothing reads it, so three different seed ideas
under `main()`'s `Random(7)` execute the same 4 experiments with the same
losses and costs. Of the 12 runs ($16.70), only 3 configs are distinct.
The other 9 runs ($12.66, 75.8%) repeat a config that already ran.

**Every expansion also proposes two copies of its parent.** `expand` sweeps
`sparsity_top` over (4, 8, 16) and `lr` over (3e-4, 1e-3). The parent
always holds one value of each, so 2 of every 5 children are the parent
itself.

| waste | shipped tree | 1,000 rng seeds |
|---|---:|---:|
| children that copy their parent | 10 / 25 | 40% by construction |
| executed runs that repeat a config | nodes #02 and #04, $1.53 of $5.57 | 24.9% |
| distinct configs among all nodes | 6 in 26 | at most 6 |

### 2 — a $5 approval gate never fires, because no node can cost more than $1.60

**Total cost drops by 0.0%.** The gate wraps `run_experiment`, estimates
each node with an oracle (the stub run on a copy of the node and the rng),
and declines anything above the threshold. Over 1,000 runs it declines 0
of 4,438 experiments; spend is $6,207.24 with and without it. The stub
charges `1.2 + uniform(0, 0.4)`, so the costliest node is $1.5999.

**The $30 budget never binds either.** The most any of the 1,000 runs
spends is $12.75 (mean $6.21), because the search stops on its 24-node cap,
and that cap counts nodes that were queued but never run. The lesson's
"Use It" transcript shows "budget 12/30" and "$28.40 spent".

A gate only bites inside the stub's cost band:

| threshold | declined | spend drop |
|---:|---:|---:|
| $5.00 | 0 | 0.0% |
| $1.50 | 6,849 | 52.9% |
| $1.40 | 7,460 | 81.3% |
| $1.20 | 5,000 | 100% |

A declined node is never expanded. Most of the saving comes from the search
running out of nodes, not from the expensive nodes it skipped.

### 3 — a single judge false-accepts 70% of control-rerun papers, and the ensemble 85%

**On 3,666 known-bad papers, a single judge false-accepts 55.3% and the
five-judge ensemble 57.4%.** The lesson ships no reviewer, so the judges
are a stand-in. Each scores the 5 NeurIPS dimensions as `1 + 4 * signal`
from the paper's recorded novelty and quality, plus its own N(0, 0.5)
leniency, and a paper passes at a mean of 4.0. The papers are the
`best_branch` headlines of 2,000 seeded pipeline runs.

| known-bad set | papers | mean signal | single judge | ensemble of 5 |
|---|---:|---:|---:|---:|
| weak (each tree's worst successful node) | 2,000 | 3.89 | 42.9% | 34.1% |
| control reruns (headline = root config) | 1,666 | 4.29 | 70.2% | 85.4% |
| real papers, for reference (accept rate) | 334 | 4.26 | 70.1% | 85.0% |

Averaging only lowers false accepts when the defect already shows in the
scores. Weak papers score below the bar, so averaging out the noise
rejects more of them. Control reruns score above it, so the ensemble
passes them more reliably than one judge does.

**83.3% of the pipeline's papers are control reruns, and they score higher
than the real ones.** 1,666 of 2,000 headlines are the root's own config,
because 2 of the root's 5 children copy it (exercise 1). `main()`'s own
best branch, "lr=0.0003", is one of them.

### 4 — curl is blocked and logged 4 of 4 ways, but the lesson's sandbox is a docstring

**All 4 exfiltration attempts are blocked and logged, and the benign
experiment still runs.** Docker is not assumed, so `--network=none` is
enforced at the process level instead. Each payload runs in a fresh
`python -I` child whose `sys.addaudithook` refuses network lookups,
connects and process spawns before they happen. It writes one JSON line
per attempt. The target is 203.0.113.7 (TEST-NET-3), so nothing could leave
even if a block failed.

| agent payload | first refused event | exit |
|---|---|---|
| `subprocess.run(["curl", ...])` | `subprocess.Popen` | non-zero, logged with target |
| `os.system("curl -d @/etc/hosts ...")` | `os.system` | non-zero, logged with target |
| `urllib.request.urlopen(...)` | `socket.getaddrinfo` | non-zero, logged with target |
| `socket.create_connection(...)` | `socket.getaddrinfo` | non-zero, logged with target |
| benign loss computation | none | 0, prints `loss=1.6439` |

**The lesson's own sandbox never executes anything, so it cannot block
anything.** Handed a node whose config carries the curl payload,
`run_experiment` returns a loss and charges a cost like any other node.
Its `docker run` line exists only in a docstring. That line also leaves out
`--pids-limit=256`, the fork-bomb guard that the lesson's step 4 and the
skill file both require.

### 5 — the tree search is a FIFO queue and loses 7.3% to flat random at the same $30

**At the same $30 budget, the tree's novelty × quality gain over flat random
is -7.3%. At the same number of experiments it is +2.6%.** The baseline draws
configs uniformly from the same 6-config grid and runs them through the
reference `run_experiment` and `verify`. Values are means over 1,000 seeds:

| search | best novelty × quality | spend / runs |
|---|---:|---|
| lesson tree search | 0.543 | $6.21, ~4.4 runs |
| flat random, same $30 | 0.586 | ~21.9 runs |
| flat random, same run count | 0.529 | ~4.4 runs |

The tree cannot spend its budget: its node cap stops it first.

**The "best-first" search is first-in-first-out.** A node's score is fixed
when it is pushed, before it runs, so every child scores 0.3 and the heap
falls back on insertion order. In 1,000 of 1,000 runs the executed ids are
exactly 1, 2, ..., k, and no run goes deeper than 2 levels. Two of the
three axes give the search nothing to find. Novelty is drawn without
reading the config: on one rng, all 6 configs get the same value. The `lr`
edit leaves the reported loss unchanged (2.747 both ways). The lesson calls
its score "novelty × quality × budget", but the code computes a weighted sum.
