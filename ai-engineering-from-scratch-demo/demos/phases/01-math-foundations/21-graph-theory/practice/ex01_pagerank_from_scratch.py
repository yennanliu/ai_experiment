"""Exercise 1 — PageRank from scratch, against the lesson's implementation.

    **Implement PageRank from scratch.** Start with uniform scores. At each step:
    score(v) = (1-d)/n + d * sum(score(u)/out_degree(u)) for all u pointing to v.
    Use d=0.85. Run until convergence (change < 1e-6). Test on a small web graph.

Reading of the exercise: the update rule as written is incomplete — it says
nothing about **dangling nodes** (pages with no outgoing links), whose score has
nowhere to go. Implemented literally, the scores stop summing to 1, and check 3
measures the leak: on a graph with one dangling node the total settles at 0.83.
The lesson's own `pagerank` redistributes that mass, which is why the two
implementations disagree until the dangling node is handled the same way.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "21-graph-theory"
DAMPING, TOL, MAX_ITER = 0.85, 1e-6, 200
# a small web graph: 0 -> 1,2 ; 1 -> 2 ; 2 -> 0 ; 3 -> 2 ; 4 dangling
EDGES = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2), (4, 2)]
DANGLING_EDGES = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2)]     # node 4 has no out-links
N = 5


def _out_links(edges, n):
    out = {u: [] for u in range(n)}
    for u, v in edges:
        out[u].append(v)
    return out


def _iterate(scores, out, n, handle_dangling):
    """One application of the exercise's update rule."""
    fresh = [(1 - DAMPING) / n] * n
    for u, targets in out.items():
        for v in targets:
            fresh[v] += DAMPING * scores[u] / len(targets)
    if handle_dangling:
        dangling = sum(scores[u] for u in range(n) if not out[u])
        for v in range(n):
            fresh[v] += DAMPING * dangling / n
    return fresh


def pagerank_naive(edges, n, handle_dangling):
    """The exercise's rule; dangling mass is redistributed only when asked."""
    out = _out_links(edges, n)
    scores = [1.0 / n] * n
    for _ in range(MAX_ITER):
        fresh = _iterate(scores, out, n, handle_dangling)
        if max(abs(a - b) for a, b in zip(fresh, scores)) < TOL:
            return fresh, sum(fresh)
        scores = fresh
    return scores, sum(scores)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "graph_theory")
    graph = ref.Graph(N, directed=True)
    for u, v in DANGLING_EDGES:
        graph.add_edge(u, v)
    theirs = [float(v) for v in ref.pagerank(graph, damping=DAMPING, tol=TOL)]
    literal, literal_total = pagerank_naive(DANGLING_EDGES, N, handle_dangling=False)
    fixed, fixed_total = pagerank_naive(DANGLING_EDGES, N, handle_dangling=True)
    no_dangling, nd_total = pagerank_naive(EDGES, N, handle_dangling=False)
    return {"theirs": theirs, "literal": literal, "fixed": fixed,
            "literal_total": literal_total, "fixed_total": fixed_total,
            "nd_total": nd_total,
            "gap_fixed": max(abs(a - b) for a, b in zip(fixed, theirs)),
            "gap_literal": max(abs(a - b) for a, b in zip(literal, theirs)),
            "ranking": sorted(range(N), key=lambda i: theirs[i], reverse=True)}


def verify(result):
    return [
        practice.Check("with dangling mass redistributed, it matches the lesson exactly",
                       result["gap_fixed"] < 1e-5,
                       f"worst |Δ| = {result['gap_fixed']:.3g}; scores "
                       f"{[round(v, 5) for v in result['fixed']]}"),
        practice.Check("…and the scores sum to 1, as a probability distribution must",
                       abs(result["fixed_total"] - 1.0) < 1e-6,
                       f"Σ = {result['fixed_total']:.9f}"),
        practice.Check("FINDING: the exercise's rule as written leaks probability",
                       result["literal_total"] < 0.95,
                       f"implemented literally, the scores sum to "
                       f"{result['literal_total']:.4f}, not 1 — node 4 has no out-links, so "
                       f"its share of the mass vanishes each iteration. The rule says "
                       f"nothing about this case"),
        practice.Check("…and it therefore disagrees with the lesson's implementation",
                       result["gap_literal"] > 100 * result["gap_fixed"],
                       f"worst |Δ| = {result['gap_literal']:.4f} against "
                       f"{result['gap_fixed']:.3g} once dangling mass is handled — a "
                       f"{result['gap_literal'] / max(result['gap_fixed'], 1e-12):.0g}x "
                       f"difference from one unstated case"),
        practice.Check("with no dangling node the leak disappears on its own",
                       abs(result["nd_total"] - 1.0) < 1e-6,
                       f"adding a single 4 -> 2 edge makes the literal rule sum to "
                       f"{result['nd_total']:.9f}. Ranking on the original graph: "
                       f"{result['ranking']} — node 2 first, as the only node two others "
                       f"point to"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
