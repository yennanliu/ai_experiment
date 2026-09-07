"""Exercise 3 — recall index not model.

    **(Hard)** Take 1,000 ImageNet validation images, embed with DINOv2 via
    HuggingFace, build a FAISS flat index, and report recall@{1, 5, 10} against
    the same images as queries (should be 1.0) and against a held-out split with
    ImageNet labels as ground truth.

Reading of the exercise: none of the named stack is installable here --
`transformers`, `faiss` and `datasets` are in no dependency group -- so their
absence is measured rather than worked around, and the retrieval is rebuilt on
the lesson's own `Encoder` and `recall_at_k` over an exact inner-product search,
which is what a FAISS *flat* index computes. That leaves the parenthesis to
examine: "(should be 1.0)" is presented as a check on the pipeline, and it is
1.0 for a randomly initialised encoder too, so it tests the index rather than
the model. The held-out half needs the baseline the exercise omits: with six
classes, recall@10 is mostly free, and the closed form says how much.

Structure: `world` returns the sampler over the lesson's six prototypes; `learn`
trains its Encoder with its own loss and mining; `sweep` reports recall at 1, 5
and 10 for a query/gallery pair; `chance` is the closed-form recall@k for
embeddings carrying no information, summed over the gallery's class frequencies.
"""

from __future__ import annotations

import pathlib

import importlib

from harness import parity, practice

TOY = pathlib.Path(__file__).resolve().parent / "ex01_pca_before_after_margin.py"

PHASE, LESSON = "04-computer-vision", "20-image-retrieval-metric"

CLASSES, DIM, EMB, NOISE = 6, 128, 64, 0.15
GALLERY, QUERIES, STEPS, BATCH, LR = 1000, 200, 200, 48, 3e-3
KS, STACK = (1, 5, 10), ("transformers", "faiss", "datasets")
REAL_RUN = ("the exercise as written wants facebook/dinov2-base over the ImageNet val split: an "
            "estimated ~350 MB of weights plus the split itself, seconds of GPU time for 1,200 "
            "embeddings -- a download no hermetic test can make, not a compute problem")

listing = lambda got: ", ".join(f"{name} {got[name]}" for name in STACK)            # noqa: E731
row = lambda scores: " / ".join(f"{v:.4f}" for v in scores)                         # noqa: E731
drift = lambda a, b: max(abs(x - y) for x, y in zip(a, b))                          # noqa: E731


def learn(torch, ref, sample):
    torch.manual_seed(0)
    encoder = ref.Encoder(in_dim=DIM, emb_dim=EMB)
    optimiser = torch.optim.Adam(encoder.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(0)
    for _ in range(STEPS):
        batch, labels = sample(BATCH, gen)
        emb = encoder(batch)
        pos, neg = ref.semi_hard_negatives(emb, labels)
        loss = ref.triplet_loss(emb, emb[pos], emb[neg])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return encoder.eval()


def sweep(ref, query_emb, gallery_emb, query_labels, gallery_labels) -> list:
    return [ref.recall_at_k(query_emb, gallery_emb, query_labels, gallery_labels, k=k) for k in KS]


def chance(torch, gallery_labels) -> list:
    """Recall@k for embeddings carrying no class information, in closed form."""
    freq = torch.bincount(gallery_labels, minlength=CLASSES).float() / len(gallery_labels)
    return [float((freq * (1 - (1 - freq) ** k)).sum()) for k in KS]


def solve():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    toy = practice.load_module(TOY)          # exercise 1's sampler, not a second copy
    torch.set_num_threads(2)
    missing = {}
    for name in STACK:
        try:
            importlib.import_module(name)
            missing[name] = "present"
        except ImportError as exc:                  # pragma: no cover - none is installed here
            missing[name] = type(exc).__name__
    sample = toy.world(torch, functional)
    gen = torch.Generator().manual_seed(7)
    gallery, gallery_labels = sample(GALLERY, gen)
    queries, query_labels = sample(QUERIES, gen)
    torch.manual_seed(0)
    untrained = ref.Encoder(in_dim=DIM, emb_dim=EMB).eval()
    trained = learn(torch, ref, sample)
    noise = torch.Generator().manual_seed(3)
    random_gallery = functional.normalize(torch.randn(GALLERY, EMB, generator=noise), dim=-1)
    random_query = functional.normalize(torch.randn(QUERIES, EMB, generator=noise), dim=-1)
    collapsed = functional.normalize(torch.ones(GALLERY, EMB), dim=-1)
    with torch.no_grad():
        banks = {name: (net(queries), net(gallery)) for name, net in
                 (("trained", trained), ("untrained", untrained))}
    return {"missing": missing, "chance": chance(torch, gallery_labels),
            "self": {name: sweep(ref, bank[1], bank[1], gallery_labels, gallery_labels)
                     for name, bank in banks.items()},
            "held": {name: sweep(ref, bank[0], bank[1], query_labels, gallery_labels)
                     for name, bank in banks.items()},
            "collapsed": sweep(ref, collapsed, collapsed, gallery_labels, gallery_labels),
            "random": sweep(ref, random_query, random_gallery, query_labels, gallery_labels)}


def verify(result):
    self_, held, null = result["self"], result["held"], result["chance"]
    collapsed, random_ = result["collapsed"], result["random"]
    gap = drift(random_, null)
    return [
        practice.Check(
            "ANSWER: none of DINOv2, FAISS or the ImageNet split is reachable",
            all(v == "ModuleNotFoundError" for v in result["missing"].values()),
            f"importing the exercise's stack gives {listing(result['missing'])}, so the retrieval is "
            f"rebuilt on the lesson's own Encoder and recall_at_k over an exact inner-product "
            f"search -- which is what a FAISS *flat* index computes, the one index type for which "
            f"the exercise's own '(should be 1.0)' is guaranteed. {REAL_RUN}"),
        practice.Check(
            "ANSWER: self-query recall is 1.0 as predicted -- and 1.0 untrained too",
            self_["trained"] == [1.0] * len(KS) == self_["untrained"],
            f"querying the {GALLERY}-vector index with its own contents gives recall@{KS} = "
            f"{row(self_['trained'])} for the trained encoder and {row(self_['untrained'])} for a "
            "randomly initialised one. The exercise offers this as a check on the pipeline; it "
            "passes identically whether or not the model has learned anything"),
        practice.Check(
            "MECHANISM: a normalised vector is its own nearest neighbour, so only collapse can fail it",
            collapsed[0] < 0.5 < self_["trained"][0],
            f"cosine similarity to itself is exactly 1.0, the maximum, so top-1 returns the query and "
            f"its label matches by construction. The one way to break it is ties: a collapsed encoder, "
            f"every vector identical, scores {row(collapsed)} instead of 1.0000 across all three k. "
            "That is what the check detects -- collapse and index bugs, not retrieval quality"),
        practice.Check(
            "FINDING: recall@10 on six classes is 0.84 before any model exists",
            gap < 0.01 and null[-1] > 0.8,
            f"for embeddings carrying no class information, recall@k is sum_c f_c(1-(1-f_c)^k) over "
            f"the gallery's class frequencies: {row(null)} at k = {KS}. Random unit vectors measure "
            f"{row(random_)}, agreeing to {gap:.4f}. So a reported recall@10 of 0.84 is the null "
            "model, and the exercise asks for the number without asking for the baseline"),
        practice.Check(
            "FINDING: only recall@1 separates the trained encoder from the untrained one",
            held["trained"][0] > held["untrained"][0] and held["trained"][-1] == held["untrained"][-1],
            f"on the {QUERIES} held-out queries the trained encoder scores {row(held['trained'])} and "
            f"the untrained one {row(held['untrained'])}: a gap of "
            f"{held['trained'][0] - held['untrained'][0]:.4f} at k=1 that closes to "
            f"{held['trained'][-1] - held['untrained'][-1]:.4f} by k=10. Reporting all three hides "
            "the only one that carries information"),
        practice.Check(
            "CONTROL: a collapsed index scores at the null, not at 1.0",
            drift(collapsed, null) < 0.1 and collapsed[0] < 2 * null[0],
            f"the collapsed encoder's self-query {row(collapsed)} lands on the closed-form null "
            f"{row(null)} to within {drift(collapsed, null):.4f}, not on the 1.0000 the exercise "
            f"expects -- {collapsed[0]:.4f} against {null[0]:.4f} at k=1. Ties resolve by index "
            "order rather than uniformly, which is why it tracks the null without matching it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
