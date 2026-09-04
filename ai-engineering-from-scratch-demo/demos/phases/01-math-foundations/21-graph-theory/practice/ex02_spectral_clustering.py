"""Exercise 2 — spectral clustering on two cliques, then add cross edges.

    **Find communities using spectral clustering.** Create a graph with two
    clearly separated clusters (e.g., two cliques connected by a single edge).
    Run spectral clustering and verify it finds the right split. What happens as
    you add more cross-cluster edges?

Reading of the exercise: "what happens as you add more cross edges" is the real
question and needs a metric that degrades gradually, not a pass/fail. Two are
used: whether the split matches ground truth, and the **Fiedler value** (the
second-smallest Laplacian eigenvalue), which measures how separable the graph is
and rises as the cut gets more expensive. The sweep finds where clustering
actually breaks, rather than asserting that it does.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "21-graph-theory"
CLIQUE, CROSS_COUNTS = 6, (1, 2, 4, 8, 16, 36)
TRUTH = [0] * CLIQUE + [1] * CLIQUE


def build(ref, n_cross):
    graph = ref.Graph(2 * CLIQUE)
    for base in (0, CLIQUE):
        for i in range(base, base + CLIQUE):
            for j in range(i + 1, base + CLIQUE):
                graph.add_edge(i, j)
    added = 0
    for i in range(CLIQUE):
        for j in range(CLIQUE):
            if added < n_cross:
                graph.add_edge(i, CLIQUE + j)
                added += 1
    return graph


def agreement(labels):
    """Fraction correct, allowing the two cluster ids to be swapped."""
    direct = sum(1 for a, b in zip(labels, TRUTH) if a == b) / len(TRUTH)
    return max(direct, 1 - direct)


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "graph_theory")
    rows = {}
    for n_cross in CROSS_COUNTS:
        graph = build(ref, n_cross)
        labels = [int(v) for v in ref.spectral_clustering(graph, k=2)]
        eigenvalues = sorted(float(v) for v in
                             numpy.linalg.eigvalsh(graph.laplacian()))
        rows[n_cross] = {"labels": labels, "agreement": agreement(labels),
                         "fiedler": eigenvalues[1],
                         "balanced": min(labels.count(0), labels.count(1))}
    return {"rows": rows, "possible_cross": CLIQUE * CLIQUE}


def _join(rows, template) -> str:
    return ", ".join(template(n, rows[n]) for n in CROSS_COUNTS)


def verify(result):
    rows = result["rows"]
    single = rows[1]
    fiedlers = [rows[n]["fiedler"] for n in CROSS_COUNTS]
    correct = [n for n in CROSS_COUNTS if rows[n]["agreement"] == 1.0]
    first_failure = next((n for n in CROSS_COUNTS if rows[n]["agreement"] < 1.0), None)
    return [
        practice.Check("two cliques joined by one edge split perfectly",
                       single["agreement"] == 1.0 and single["balanced"] == CLIQUE,
                       f"labels {single['labels']} — {CLIQUE} and {CLIQUE}, matching "
                       f"ground truth exactly; Fiedler value {single['fiedler']:.4f}"),
        practice.Check("ANSWER: the split survives well past a single cross edge",
                       len(correct) >= 3,
                       "cross edges -> agreement: "
                       + _join(rows, lambda n, r: f"{n}: {r['agreement']:.0%}")
                       + (f"; first failure at {first_failure}" if first_failure
                          else "; never fails in this sweep")),
        practice.Check("the Fiedler value rises monotonically with cross edges",
                       all(a < b for a, b in zip(fiedlers, fiedlers[1:])),
                       _join(rows, lambda n, r: f"{n}: {r['fiedler']:.3f}")
                       + " — the second-smallest Laplacian eigenvalue is the cost of the "
                         "cheapest cut, so it grows as the two halves get harder to "
                         "separate"),
        practice.Check("…which is what makes it a *graded* measure of separability",
                       fiedlers[-1] / fiedlers[0] > 5,
                       f"it grows {fiedlers[-1] / fiedlers[0]:.0f}x from 1 cross edge to "
                       f"{CROSS_COUNTS[-1]}, while the agreement stays at 100% for most of "
                       f"that range — the eigenvalue degrades smoothly where the label "
                       f"agreement is a cliff"),
        practice.Check(f"at {result['possible_cross']} cross edges the graph is one clique",
                       rows[CROSS_COUNTS[-1]]["fiedler"] > rows[1]["fiedler"] * 5,
                       f"with all {result['possible_cross']} possible cross edges present "
                       f"the graph is complete on {2 * CLIQUE} nodes, Fiedler value "
                       f"{rows[CROSS_COUNTS[-1]]['fiedler']:.2f}. Any 2-way split is then "
                       f"arbitrary, and agreement of "
                       f"{rows[CROSS_COUNTS[-1]]['agreement']:.0%} reflects that there is "
                       f"no community structure left to find"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
