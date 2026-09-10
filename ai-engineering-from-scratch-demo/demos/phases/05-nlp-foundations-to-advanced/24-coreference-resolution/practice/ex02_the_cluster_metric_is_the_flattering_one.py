"""Exercise 2 — the cluster metric is the flattering one.

    **Medium.** Use a pretrained neural coref model on a news article. Compare
    clusters against your own manual annotation. Where did it fail?

Reading of the exercise: no neural coref is installed, so the system under
comparison is the lesson's own `clusters`, and the manual annotation is exercise
1's gold links extended by string identity -- two mentions of the same name
corefer. Over 34 mentions the resolver scores **B-cubed F1 0.8525** against a
mention-link accuracy of **0.6667**. The cluster metric is the more flattering of
the two, on the same output.

It is flattering because most mentions are singletons. 11 of the 21 gold clusters
are one mention that corefers with nothing, and a singleton is a cluster no
system can get wrong. Restricted to mentions that gold puts in a cluster of size
two or more, B-cubed F1 falls to **0.8106** and recall to **0.7536**.

Where it fails is precise: every wrong link merges two entities, and the merge is
a cluster error for every mention on both sides. Four predicted clusters span
more than one gold entity -- `Apple + It`, `Analysts + his`, `Engineers + it`,
`Yusuf Demir + She + him` -- one for each of exercise 1's four bad links. A link
error is never local.

It also fails in the opposite direction, for a reason no link accuracy can see.
`resolve` iterates only pronouns, and `clusters` unions only what `resolve`
returned, so no two non-pronoun mentions are ever joined -- string identity
included. "Nokia announced the phone. Nokia also cut prices. It blamed the
market." comes back as `[Nokia] [the phone] [Nokia, It]` against a gold of
`[Nokia, Nokia, It] [the phone]`. Every entity named twice is two clusters.

The two failures partly cancel in the count: 22 predicted clusters against 21
gold, from a system that mis-links a third of its pronouns.

`main()` prints only clusters of size above one, so the singletons never appear
on screen -- four tidy clusters, and the twelve the system could not place stay
out of the output.

Structure: exercise 1's paragraphs and gold links are loaded rather than copied;
`gold_clusters` extends the gold links with string identity, `predicted` runs the
lesson's own union-find, `b_cubed` scores one paragraph, and `merged` names the
predicted clusters that span more than one gold entity.
"""

from __future__ import annotations

import importlib.util
import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "24-coreference-resolution"

EX1 = practice.load_module(
    pathlib.Path(__file__).resolve().parent / "ex01_recency_resolves_to_the_leftmost_mention.py")
ROWS = EX1.ROWS
UNAVAILABLE = ("spacy", "coreferee", "fastcoref", "allennlp", "transformers")


def merge(size, pairs):
    """Union-find over mention indices: the groups `pairs` implies, as index lists."""
    parent = list(range(size))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    out = {}
    for i in range(size):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def gold_clusters(mentions, gold):
    """The gold links, plus the rule that two mentions of one name corefer."""
    pronouns = [i for i, m in enumerate(mentions) if m["type"] == "pronoun"]
    pairs = [(i, [j for j in range(i) if mentions[j]["text"] == want][-1:])
             for want, i in zip(gold, pronouns)]
    pairs += [(i, [j for j in range(i) if mentions[j]["type"] == "ne" == m["type"]
                   and mentions[j]["text"] == m["text"]][-1:]) for i, m in enumerate(mentions)]
    return merge(len(mentions), [(i, j[0]) for i, j in pairs if j])


def predicted(ref, mentions):
    """The lesson's own clustering, as index lists rather than texts."""
    where = {id(m): i for i, m in enumerate(mentions)}
    return merge(len(mentions), [(where[id(m)], where[id(a)])
                                 for m, a in ref.resolve(mentions) if a is not None])


def b_cubed(gold, pred, keep):
    """Per-mention precision and recall sums over the mentions `keep` selects."""
    gmap = {i: set(c) for c in gold for i in c}
    pmap = {i: set(c) for c in pred for i in c}
    chosen = [i for i in gmap if keep(gmap[i])]
    precision = sum(len(gmap[i] & pmap[i]) / len(pmap[i]) for i in chosen)
    recall = sum(len(gmap[i] & pmap[i]) / len(gmap[i]) for i in chosen)
    return precision, recall, len(chosen)


def sweep(ref, keep):
    """Micro-averaged B-cubed over every paragraph, and the cluster counts."""
    totals, gold_n, pred_n, singles, merged = [0.0, 0.0, 0], 0, 0, 0, []
    for text, gold in ROWS:
        mentions = ref.extract_mentions(text)
        truth, guess = gold_clusters(mentions, gold), predicted(ref, mentions)
        precision, recall, n = b_cubed(truth, guess, keep)
        totals = [totals[0] + precision, totals[1] + recall, totals[2] + n]
        gold_n, pred_n = gold_n + len(truth), pred_n + len(guess)
        singles += sum(1 for c in truth if len(c) == 1)
        merged += [[mentions[i]["text"] for i in c] for c in guess
                   if len({frozenset(g) for g in truth for i in c if i in g}) > 1]
    p, r = totals[0] / totals[2], totals[1] / totals[2]
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4),
            "mentions": totals[2], "gold": gold_n, "pred": pred_n, "singletons": singles,
            "merged": merged}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    mentions = ref.extract_mentions(ROWS[6][0])
    return {
        "absent": [n for n in UNAVAILABLE if importlib.util.find_spec(n) is None],
        "all": sweep(ref, lambda c: True),
        "linked": sweep(ref, lambda c: len(c) > 1),
        "link_accuracy": round(EX1.score(ref)[0] / sum(len(g) for _, g in ROWS), 4),
        "nokia_pred": [[mentions[i]["text"] for i in c] for c in predicted(ref, mentions)],
        "nokia_gold": [[mentions[i]["text"] for i in c] for c in gold_clusters(mentions, ROWS[6][1])],
    }


def verify(result):
    every, linked = result["all"], result["linked"]
    return [
        practice.Check(
            "ANSWER: B-cubed F1 0.8525 against a link accuracy of 0.6667",
            every["f1"] > result["link_accuracy"],
            f"{result['absent']} are all absent, so the system under comparison is the lesson's "
            f"own `clusters`, annotated by exercise 1's gold links extended with string identity. "
            f"Over {every['mentions']} mentions: precision {every['precision']}, recall "
            f"{every['recall']}, F1 {every['f1']} -- against {result['link_accuracy']} of its "
            "pronouns linked correctly",
        ),
        practice.Check(
            "MECHANISM: because most mentions are singletons, which cannot be got wrong",
            every["singletons"] > every["gold"] / 2 and linked["f1"] < every["f1"],
            f"{every['singletons']} of {every['gold']} gold clusters are a single mention that "
            f"corefers with nothing. Restricted to mentions gold puts in a cluster of two or "
            f"more, F1 falls to {linked['f1']} and recall to {linked['recall']} over "
            f"{linked['mentions']} mentions",
        ),
        practice.Check(
            "FINDING: where it fails is one merged cluster per bad link",
            len(every["merged"]) == 4,
            f"{len(every['merged'])} predicted clusters span more than one gold entity: "
            f"{every['merged']}. Each is one of exercise 1's four wrong links, and the merge is a "
            "cluster error for every mention on both sides. A link error is never local",
        ),
        practice.Check(
            "FINDING: and it fails the other way for a reason no link accuracy can see",
            len(result["nokia_pred"]) > len(result["nokia_gold"]),
            f"`resolve` iterates only pronouns and `clusters` unions only what it returned, so no "
            f"two non-pronoun mentions are ever joined -- string identity included. The Nokia "
            f"paragraph comes back {result['nokia_pred']} against {result['nokia_gold']}: every "
            "entity named twice is two clusters",
        ),
        practice.Check(
            "MECHANISM: the two failures partly cancel in the cluster count",
            abs(every["pred"] - every["gold"]) <= 2,
            f"{every['pred']} predicted clusters against {every['gold']} gold, from a system that "
            "mis-links a third of its pronouns: over-splitting repeated names and wrongly merging "
            "distinct entities move the count in opposite directions",
        ),
        practice.Check(
            "CONTROL: `main()` prints only the clusters above size one",
            every["singletons"] > 0,
            f"its loop is `if len(cluster) > 1`, so the {every['singletons']} singleton clusters "
            "never reach the screen. What is printed is four tidy groups; what is hidden is every "
            "mention the system could not place",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
