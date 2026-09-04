"""Exercise 4 — two rounds of message passing reach the 2-hop neighbourhood.

    **Build a 2-layer message passing network.** Apply message passing twice with
    different weight matrices. Show that after 2 rounds, each node has
    information from its 2-hop neighborhood.

Reading of the exercise: "has information from its 2-hop neighborhood" needs an
experiment, not an inspection of the output. The test used is **intervention**:
perturb one node's input feature and see which nodes' outputs change. After one
round only 1-hop neighbours move; after two, 2-hop neighbours do; and 3-hop nodes
must stay exactly unchanged, or the claim is not about 2 hops at all (check 4).
That last part is the one an eyeball check would skip.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "21-graph-theory"
SEED, D_IN, D_HID, D_OUT = 42, 4, 5, 3
# a path 0-1-2-3-4, so hop distance from node 0 is just the index
PATH = [(0, 1), (1, 2), (2, 3), (3, 4)]
N = 5


def build(ref):
    graph = ref.Graph(N)
    for u, v in PATH:
        graph.add_edge(u, v)
    return graph


def two_layers(ref, graph, features, w1, w2):
    hidden = ref.message_passing(graph, features, w1)
    return ref.message_passing(graph, hidden, w2)


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "graph_theory")
    rng = numpy.random.default_rng(SEED)
    graph = build(ref)
    features = rng.normal(size=(N, D_IN))
    w1 = rng.normal(size=(D_IN, D_HID))
    w2 = rng.normal(size=(D_HID, D_OUT))

    perturbed = features.copy()
    perturbed[0] += 10.0                     # intervene on node 0 only

    one_base = ref.message_passing(graph, features, w1)
    one_pert = ref.message_passing(graph, perturbed, w1)
    two_base = two_layers(ref, graph, features, w1, w2)
    two_pert = two_layers(ref, graph, perturbed, w1, w2)

    def moved(base, pert):
        return [float(numpy.abs(pert[i] - base[i]).max()) for i in range(N)]

    return {"after_one": moved(one_base, one_pert),
            "after_two": moved(two_base, two_pert),
            "shapes": {"features": tuple(features.shape),
                       "hidden": tuple(one_base.shape),
                       "output": tuple(two_base.shape)},
            "different_weights": bool(w1.shape != w2.shape)}


def _by_node(values) -> str:
    return "|Δ| by node: " + ", ".join(f"{i}(hop {i}): {v:.3g}"
                                       for i, v in enumerate(values))


def _only_hop(values, hop) -> bool:
    return values[hop] > 1e-6 and all(v < 1e-12 for v in values[hop + 1:])


def verify(result):
    one, two = result["after_one"], result["after_two"]
    return [
        practice.Check(f"two layers reshape {D_IN} -> {D_HID} -> {D_OUT}",
                       result["shapes"]["output"] == (N, D_OUT)
                       and result["different_weights"],
                       f"features {result['shapes']['features']} -> hidden "
                       f"{result['shapes']['hidden']} -> output "
                       f"{result['shapes']['output']}, with genuinely different weight "
                       f"matrices as the exercise requires"),
        practice.Check("after ONE round, only the 1-hop neighbour has moved",
                       _only_hop(one, 1), _by_node(one)),
        practice.Check("after TWO rounds, the 2-hop neighbour has moved too",
                       two[2] > 1e-6, _by_node(two)),
        practice.Check("…and nodes 3 and 4 hops away are EXACTLY unchanged",
                       two[3] == 0.0 and two[4] == 0.0,
                       f"node 3 moved {two[3]}, node 4 moved {two[4]} — exactly zero, not "
                       f"merely small. Without this the claim would be 'information "
                       f"spreads', not 'information spreads two hops'"),
        practice.Check("the receptive field is the layer count, which is why depth is needed",
                       one[2] == 0.0 and two[2] > 0.0,
                       f"node 2 is invisible after one round ({one[2]}) and visible after "
                       f"two ({two[2]:.3g}). k layers see exactly k hops, so reaching a "
                       f"distant node needs depth — and depth is what causes "
                       f"over-smoothing"),
        practice.Check("FINDING: influence alternates by parity — there are no self-loops",
                       one[1] > 1e-6 and two[1] == 0.0 and one[0] == 0.0 and two[0] > 1e-6,
                       f"node 1 moves {one[1]:.3g} after one round and exactly "
                       f"{two[1]} after two; node 0 moves {one[0]} then "
                       f"{two[0]:.3g}. message_passing normalises A without adding I, so a "
                       f"node never aggregates its own features — after k rounds only "
                       f"nodes at distance k, k−2, … are reached. Real GNNs use A + I "
                       f"precisely to avoid this"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
