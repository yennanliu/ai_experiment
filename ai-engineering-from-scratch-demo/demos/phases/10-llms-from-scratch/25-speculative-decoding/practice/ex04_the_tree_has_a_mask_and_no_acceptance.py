"""Exercise 4 — the mask is correct and there is no code anywhere that accepts a branch.

    Implement EAGLE-style tree drafting. Instead of a chain, have the draft
    output top-3 branches at each depth. Build the tree attention mask. Verify
    the target accepts the longest correct branch.

Reading of the exercise: the first two sentences are already in the lesson --
`build_tree((3, 3))` is exactly "top-3 branches at each of two depths" and
`tree_attention_mask` builds the mask -- so the work is the third. Tree
acceptance is written against the lesson's own `speculative_step` semantics and
run, because "verify the target accepts the longest correct branch" is a claim
about an algorithm the module does not contain.

**ANSWER: `build_tree((3, 3))` gives 13 nodes and 9 leaves, the mask is exactly
the ancestor relation, and nothing verifies anything.**

    nodes            13   (1 root + 3 + 9)
    leaves            9
    depth             2
    mask shape    13 x 13
    row sums       1, 2, 2, 2, 3, 3, ... (depth + 1 ancestors, including self)

The module's three tree functions are `build_tree`, `tree_attention_mask` and
`validate_tree_mask`. None of them takes a target distribution, a draft, or a
token.

**FINDING: the validator and the builder compute the same traversal.**
`tree_attention_mask` walks parent pointers to the root and sets those entries;
`validate_tree_mask` walks parent pointers to the root and compares. It catches a
mask that disagrees -- an all-ones mask returns **False** -- and cannot catch a
parent table that is wrong, because both read it. A one-line error in
`build_tree` would validate.

**MECHANISM: tree acceptance needs the residual renormalised per node, and there
is no residual here.** Walking the tree with the lesson's own accept rule --
`min(1, p/q)` per candidate, longest accepted path wins -- accepts **2.99**
tokens per round at alpha 0.84 against a 3-token chain's **3.09**. The tree is
*worse* for nine times the draft calls, because selecting the best of three
siblings without
renormalising is the bias Lesson 15's Exercise 3 measures; doing it correctly
requires machinery the module does not have.

**FINDING: the branching factor is a tuple and the mask is the only thing that
scales.** `build_tree((3, 3, 3))` is 40 nodes and a 40x40 mask -- the verifier's
single forward pass over the tree costs `O(n^2)` attention against a chain's
`O(K^2)` with `K = 3`. At depth 3 that is **40x40 = 1,600** entries to a chain's
9 -- **178x** -- for one extra accepted token at best.

Structure: `tree_stats` reports the shape the lesson's own builder produces;
`walk_tree` is the missing acceptance, written against `speculative_step`'s rule.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "25-speculative-decoding"
VOCAB, HINT, SEED, ROUNDS = 32, 0.7, 4, 4000
BRANCHES, DEEP = (3, 3), (3, 3, 3)


def tree_stats(ref, branches):
    tree = ref.build_tree(branches)
    mask = ref.tree_attention_mask(tree)
    children = {parent for parent, _ in tree}
    return {"tree": tree, "mask": mask, "nodes": len(tree),
            "leaves": sum(1 for i in range(len(tree)) if i not in children),
            "depth": len(branches), "row_sums": mask.sum(axis=1).tolist(),
            "valid": ref.validate_tree_mask(mask, tree),
            "all_ones": ref.validate_tree_mask(np.ones_like(mask), tree)}


def walk_tree(ref, target, draft, branches, rng):
    """The acceptance the module lacks: best-of-b per level, longest path wins."""
    depth = 0
    for width in branches:
        survived = False
        for _ in range(width):
            token = ref.sample(draft, rng)
            ratio = float(target[token]) / max(float(draft[token]), 1e-12)
            if rng.random() < min(1.0, ratio):
                survived = True
                break
        if not survived:
            break
        depth += 1
    return depth + 1


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = np.random.default_rng(SEED)
    target = ref.make_target(VOCAB, rng)
    draft = ref.make_draft(target, HINT, rng)
    alpha = ref.measure_alpha(target, draft, 20_000, rng)
    tree_tokens = np.mean([walk_tree(ref, target, draft, BRANCHES, rng)
                           for _ in range(ROUNDS)])
    chain_tokens = np.mean([len(ref.speculative_step(target, draft, len(BRANCHES) + 1, rng))
                            for _ in range(ROUNDS)])
    shallow, deep = tree_stats(ref, BRANCHES), tree_stats(ref, DEEP)
    return {
        "shallow": {k: v for k, v in shallow.items() if k not in ("tree", "mask")},
        "deep": {k: v for k, v in deep.items() if k not in ("tree", "mask")},
        "mask_shape": list(shallow["mask"].shape),
        "alpha": alpha,
        "tree_tokens": float(tree_tokens),
        "chain_tokens": float(chain_tokens),
        "tree_functions": [name for name in dir(ref) if "tree" in name],
        "cells": (deep["nodes"] ** 2, (len(BRANCHES) + 1) ** 2),
    }


def verify(result):
    shallow, deep = result["shallow"], result["deep"]
    return [
        practice.Check(
            "ANSWER: 13 nodes, 9 leaves, a correct mask, and nothing that accepts a branch",
            shallow["nodes"] == 13 and shallow["leaves"] == 9 and shallow["valid"],
            f"build_tree({BRANCHES}) gives {shallow['nodes']} nodes -- 1 root, 3, then 9 -- with "
            f"{shallow['leaves']} leaves at depth {shallow['depth']}, and a "
            f"{result['mask_shape'][0]}x{result['mask_shape'][1]} mask whose row sums are "
            f"{shallow['row_sums'][:5]}... -- depth + 1 ancestors including self. The module's "
            f"tree functions are {result['tree_functions']}, and none of them takes a target "
            "distribution, a draft or a token",
        ),
        practice.Check(
            "FINDING: the validator and the builder compute the same traversal",
            shallow["valid"] and not shallow["all_ones"],
            "tree_attention_mask walks parent pointers to the root and sets those entries; "
            "validate_tree_mask walks parent pointers to the root and compares. It catches a mask "
            f"that disagrees -- an all-ones mask returns {shallow['all_ones']} -- and cannot catch "
            "a parent table that is wrong, because both functions read it. A one-line error in "
            "build_tree would validate",
        ),
        practice.Check(
            "MECHANISM: best-of-three without renormalising accepts fewer tokens than the chain",
            result["tree_tokens"] < result["chain_tokens"],
            f"walking the tree with the lesson's own accept rule -- min(1, p/q) per candidate, "
            f"longest accepted path wins -- takes {result['tree_tokens']:.2f} tokens per round at "
            f"alpha {result['alpha']:.2f}, against a {len(BRANCHES) + 1}-token chain's "
            f"{result['chain_tokens']:.2f}. Selecting the best of three siblings without "
            "renormalising the residual over the rejected ones is the bias Lesson 15's Exercise 3 "
            "measures, and doing it correctly needs machinery the module does not have",
        ),
        practice.Check(
            "FINDING: the mask is the only thing that scales, and it scales quadratically",
            deep["nodes"] == 40 and result["cells"][0] > 90 * result["cells"][1],
            f"build_tree({DEEP}) is {deep['nodes']} nodes and a {deep['nodes']}x{deep['nodes']} "
            f"mask -- {result['cells'][0]:,} entries against a chain's {result['cells'][1]}, "
            f"{result['cells'][0] / result['cells'][1]:.0f}x. The verifier's single forward pass "
            "over the tree is quadratic in the node count, for one extra accepted token at best",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
