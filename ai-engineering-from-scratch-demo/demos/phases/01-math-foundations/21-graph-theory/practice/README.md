<!-- generated:start -->
# 01-math-foundations / 21-graph-theory

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/21-graph-theory/) · upstream spec
`phases/01-math-foundations/21-graph-theory/docs/en.md`

```bash
uv run demo practice run 21-graph-theory --ex 1
uv run demo explain 21-graph-theory --ex 1
uv run pytest demos/phases/01-math-foundations/21-graph-theory
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement PageRank from scratch. Start with uniform scores. At each step: score(v) = (1-d)/n… | code | T0 | `ex01_pagerank_from_scratch.py` |
| 2 | Find communities using spectral clustering. Create a graph with two clearly separated cluster… | code | T0 | `ex02_spectral_clustering.py` |
| 3 | Implement Dijkstra's algorithm for shortest paths in weighted graphs. Compare results to BFS… | code | T0 | `ex03_dijkstra.py` |
| 4 | Build a 2-layer message passing network. Apply message passing twice with different weight ma… | code | T0 | `ex04_two_layer_message_passing.py` |
| 5 | Analyze a real-world graph. Use the Karate Club graph (34 nodes, 78 edges). Compute degree di… | code | T0 | `ex05_karate_club.py` |
<!-- generated:end -->

## Answers

**1 — the update rule in the exercise is incomplete.** It says nothing about
**dangling nodes**, and implemented literally the scores sum to **0.83**, not 1:
a node with no out-links has its share of the mass vanish each iteration. With
that mass redistributed the implementation matches the lesson's `pagerank` to
3.1e-07; without, they differ by 0.065. Adding a single out-edge makes the literal
rule sum to 1 again, which is why the omission is easy to miss.

**2 — the split survives far more cross-edges than "clearly separated" suggests.**

| cross edges | agreement | Fiedler value |
|---:|---:|---:|
| 1 | 100% | 0.258 |
| 2 | 100% | 0.469 |
| 4 | 100% | 0.783 |
| 8 | 100% | 1.497 |
| 16 | 92% | 2.829 |
| 36 (complete) | 50% | 12.000 |

The two measures behave differently and that is the point. Label agreement is a
**cliff** — perfect until it isn't. The Fiedler value, the second-smallest
Laplacian eigenvalue, is the cost of the cheapest cut and rises smoothly by 46×
across the sweep. If you want to know how separable a graph is *before* trusting a
clustering, the eigenvalue is the thing to look at.

**3 — with uniform weights BFS and Dijkstra agree, and that check proves little.**
Unit weights make hop count *equal to* total weight, so agreement is definitional.
The informative case is where they diverge: with a direct 0→5 edge weighing 20
against a 5-hop path weighing 5, BFS reports 1 hop and Dijkstra reports cost 5.
Fewest hops and cheapest route are different questions and each algorithm answers
its own correctly.

The negative-edge failure took some care to construct. This lazy-heap Dijkstra
re-pushes improved distances, so the *directly* affected node comes out right —
node 2 is correctly improved to 0. The wrong answer appears one hop
**downstream**: node 3 was relaxed from node 2's stale value of 1 and, being in
the settled set, is never re-relaxed. It reports 2 where the truth is 1, with no
error raised. That is what Bellman-Ford is for.

**4 — tested by intervention, not inspection.** Perturbing node 0's input feature
and watching which outputs move:

| | node 0 | node 1 | node 2 | node 3 | node 4 |
|---|---:|---:|---:|---:|---:|
| after 1 round | 0 | 13.7 | 0 | 0 | 0 |
| after 2 rounds | 16.4 | 0 | 8.18 | **0** | **0** |

Nodes 3 and 4 hops away are *exactly* zero, not merely small — without that the
claim would be "information spreads", not "information spreads two hops".

The zeros in the wrong places are a finding: node 1 moves after one round and is
exactly unchanged after two. `message_passing` normalises A **without adding I**,
so a node never aggregates its own features and influence alternates by parity —
after k rounds only nodes at distance k, k−2, … are reached. Real GNNs use A + I
precisely to avoid this.

**5 — spectral clustering gets 33 of 34, not 34.** The graph is verified first
(78 edges, degree sum 156, one component) before anything is concluded from it.
Degrees are hub-dominated — 17 at node 33 and 16 at node 0, the club president and
the instructor, against a mean of 4.6 — and the Laplacian has exactly one zero
eigenvalue with a Fiedler value of 0.4685.

The single miss is node 2, degree 10, with edges into both factions (0, 1, 3, 7,
13 on one side; 8, 9, 27, 28, 32 on the other). The "known" partition is who each
member actually followed after the dispute — a social fact the adjacency only
mostly encodes. 33/34 is the honest answer rather than a tuning failure.
