"""Exercise 5 — the Karate Club graph: degrees, Laplacian spectrum, clustering.

    **Analyze a real-world graph.** Use the Karate Club graph (34 nodes, 78
    edges). Compute degree distribution, Laplacian eigenvalues, and spectral
    clustering. Compare the spectral clustering result to the known ground truth
    split.

Reading of the exercise: the ground-truth comparison is the interesting part, and
the honest result is **33 of 34** — not all 34. The single miss is worth naming
rather than averaging away: node 2, degree 10, with edges into both factions.
This is a real graph and the "known" partition is who each member actually
followed after the dispute, a social fact the adjacency structure only mostly
encodes. Check 1 verifies the graph is the right one before anything is concluded
from it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "21-graph-theory"
N = 34
EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11),
    (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31), (1, 2), (1, 3), (1, 7),
    (1, 13), (1, 17), (1, 19), (1, 21), (1, 30), (2, 3), (2, 7), (2, 8), (2, 9),
    (2, 13), (2, 27), (2, 28), (2, 32), (3, 7), (3, 12), (3, 13), (4, 6), (4, 10),
    (5, 6), (5, 10), (5, 16), (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33),
    (14, 32), (14, 33), (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32),
    (20, 33), (22, 32), (22, 33), (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
    (24, 25), (24, 27), (24, 31), (25, 31), (26, 29), (26, 33), (27, 33), (28, 31),
    (28, 33), (29, 32), (29, 33), (30, 32), (30, 33), (31, 32), (31, 33), (32, 33),
]
# Zachary's observed split: which faction each member joined
TRUTH = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def _align(labels):
    """Cluster ids are arbitrary, so score both labellings and keep the better."""
    direct = sum(1 for a, b in zip(labels, TRUTH) if a == b)
    if direct >= N - direct:
        return direct, labels
    return N - direct, [1 - v for v in labels]


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "graph_theory")
    graph = ref.Graph(N)
    for u, v in EDGES:
        graph.add_edge(u, v)
    degrees = [graph.degree(i) for i in range(N)]
    eigenvalues = sorted(float(v) for v in numpy.linalg.eigvalsh(graph.laplacian()))
    labels = [int(v) for v in ref.spectral_clustering(graph, k=2)]
    best, aligned = _align(labels)
    wrong = [i for i, (a, b) in enumerate(zip(aligned, TRUTH)) if a != b]
    return {"degrees": degrees, "eigenvalues": eigenvalues[:4], "labels": aligned,
            "correct": best, "wrong": wrong,
            "n_edges": len(EDGES), "degree_sum": sum(degrees),
            "max_degree": max(degrees), "hubs": [i for i, d in enumerate(degrees)
                                                 if d == max(degrees)],
            "components": len(ref.connected_components(graph)),
            "sizes": (aligned.count(0), aligned.count(1))}


def verify(result):
    eigen = result["eigenvalues"]
    return [
        practice.Check(f"the graph really is {N} nodes and {result['n_edges']} edges",
                       result["n_edges"] == 78 and result["degree_sum"] == 156,
                       f"Σdegree = {result['degree_sum']} = 2 × {result['n_edges']}, and "
                       f"{result['components']} connected component"),
        practice.Check("the degree distribution is hub-dominated, as social graphs are",
                       result["max_degree"] == 17 and result["degrees"][0] == 16,
                       f"node 33 has degree {result['max_degree']} and node 0 has "
                       f"{result['degrees'][0]} — the club president and the instructor — "
                       f"against a mean of {result['degree_sum'] / N:.1f}. Two nodes carry "
                       f"{(result['max_degree'] + result['degrees'][0]) / result['degree_sum']:.0%} "
                       f"of all edge endpoints"),
        practice.Check("the Laplacian has exactly one zero eigenvalue",
                       abs(eigen[0]) < 1e-9 and eigen[1] > 0.1,
                       f"λ = {[round(v, 4) for v in eigen]} — one zero per connected "
                       f"component, so exactly one confirms the graph is connected; the "
                       f"Fiedler value {eigen[1]:.4f} is small, which is what makes the "
                       f"graph splittable at all"),
        practice.Check(f"ANSWER: spectral clustering gets {result['correct']} of {N} right",
                       result["correct"] == 33,
                       f"{result['correct']}/{N} = {result['correct'] / N:.1%}, split "
                       f"{result['sizes'][0]}/{result['sizes'][1]}"),
        practice.Check(f"…and misses only node {result['wrong'][0]}, which straddles the split",
                       result["wrong"] == [2],
                       f"node 2 has degree {result['degrees'][2]} with edges into both "
                       f"factions (0, 1, 3, 7, 13 on one side; 8, 9, 27, 28, 32 on the "
                       f"other). The 'known' partition is who each member actually followed "
                       f"after the dispute — a social fact the adjacency only mostly "
                       f"encodes — so 33/34 is the honest answer rather than a tuning "
                       f"failure"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
