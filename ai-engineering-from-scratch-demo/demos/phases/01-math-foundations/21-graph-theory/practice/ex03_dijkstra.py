"""Exercise 3 — Dijkstra for weighted shortest paths, against BFS.

    **Implement Dijkstra's algorithm** for shortest paths in weighted graphs.
    Compare results to BFS on the same graph with uniform weights.

Reading of the exercise: with uniform weights the two agree, which is the check
the exercise asks for — but it is the weak half. The useful comparison is where
they *disagree*: BFS minimises hop count and Dijkstra minimises total weight, so
a graph with one long-but-few-hops route separates them (check 4). Check 5 covers the negative-edge case, and needed care to
construct: this lazy-heap variant re-pushes improved distances, so it gets the
*directly* affected node right. The wrong answer appears one hop **downstream**,
at a node relaxed from the stale value and never revisited.
"""

from __future__ import annotations

import heapq

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "21-graph-theory"
N = 6


def dijkstra(graph, start):
    distances = {node: float("inf") for node in range(graph.n)}
    distances[start] = 0.0
    seen, queue = set(), [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if node in seen:
            continue
        seen.add(node)
        for neighbour, weight in graph.adj[node].items():
            candidate = distance + weight
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return distances


def uniform_graph(ref):
    graph = ref.Graph(N)
    for u, v in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)):
        graph.add_edge(u, v, 1.0)
    return graph


def weighted_graph(ref):
    """0->5 direct is 1 hop but costs 20; the long way is 5 hops costing 5."""
    graph = ref.Graph(N)
    for u, v in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)):
        graph.add_edge(u, v, 1.0)
    graph.add_edge(0, 5, 20.0)
    return graph


def solve():
    ref = parity.load_reference(PHASE, LESSON, "graph_theory")
    uniform = uniform_graph(ref)
    # bfs returns (visit order, hop distances)
    _, bfs_uniform = ref.bfs(uniform, 0)
    dij_uniform = dijkstra(uniform, 0)
    weighted = weighted_graph(ref)
    _, bfs_weighted = ref.bfs(weighted, 0)
    dij_weighted = dijkstra(weighted, 0)
    # 0->2 is cheap so node 2 settles early; 1->2 later improves it to 0, but
    # node 3 downstream was already relaxed from the stale value and is never
    # revisited. True 0->3 is 4 - 4 + 1 = 1.
    negative = ref.Graph(4, directed=True)
    for u, v, w in ((0, 1, 4.0), (0, 2, 1.0), (1, 2, -4.0), (2, 3, 1.0)):
        negative.add_edge(u, v, w)
    return {"bfs_uniform": dict(bfs_uniform), "dij_uniform": dict(dij_uniform),
            "bfs_weighted": dict(bfs_weighted), "dij_weighted": dict(dij_weighted),
            "negative": dict(dijkstra(negative, 0)), "true_negative": 1.0}


def verify(result):
    hops_u, dij_u = result["bfs_uniform"], result["dij_uniform"]
    hops_w, dij_w = result["bfs_weighted"], result["dij_weighted"]
    return [
        practice.Check("on uniform weights BFS and Dijkstra agree at every node",
                       all(abs(hops_u[k] - dij_u[k]) < 1e-12 for k in hops_u),
                       f"BFS {hops_u}, Dijkstra "
                       f"{ {k: round(v, 1) for k, v in dij_u.items()} }"),
        practice.Check("…which is expected: unit weights make hop count the total weight",
                       all(abs(hops_u[k] - dij_u[k]) < 1e-12 for k in hops_u),
                       "so this check confirms the implementation and nothing about the "
                       "algorithms' difference — checks 3-5 go after that"),
        practice.Check("with weights they DISAGREE, and each is right about its own question",
                       hops_w[5] < dij_w[5] or hops_w[5] != dij_w[5],
                       f"node 5: BFS says {hops_w[5]} hops (the direct 0->5 edge), "
                       f"Dijkstra says cost {dij_w[5]:.0f} (the 5-hop path). The direct edge "
                       f"weighs 20 and the long way weighs 5 — fewest hops and cheapest "
                       f"route are different objectives"),
        practice.Check("Dijkstra takes the long way precisely because it is cheaper",
                       abs(dij_w[5] - 5.0) < 1e-12 and hops_w[5] == 1,
                       f"cost {dij_w[5]:.0f} via 0-1-2-3-4-5 against 20 direct; BFS reports "
                       f"1 hop and would route through the expensive edge"),
        practice.Check("FINDING: a negative edge makes it silently wrong, one hop downstream",
                       abs(result["negative"][3] - result["true_negative"]) > 0.5
                       and abs(result["negative"][2] - 0.0) < 1e-9,
                       f"0->3 costs {result['negative'][3]:.0f} against a true "
                       f"{result['true_negative']:.0f}. Node 2 settles early at 1 and "
                       f"relaxes node 3 to 2; a later negative edge improves node 2 to "
                       f"{result['negative'][2]:.0f}, correctly, but node 3 is in the "
                       f"settled set and never re-relaxed. So node 2's answer is right and "
                       f"node 3's is not, with no error raised — which is why Bellman-Ford "
                       f"exists"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
