"""Exercise 3 — retrieval recall at k null.

    **(Hard)** Build a zero-shot image retrieval index: embed 1,000 images with CLIP, build a FAISS index, query with a natural language description. Report retrieval recall@5 for 20 held-out queries you write by hand.

Reading of the exercise: `faiss` and `open_clip` are both absent from every
dependency group here, so there is no index library and no checkpoint, and the
queries cannot be "written by hand" because nothing can tokenise a sentence. The
index half loses nothing to that: a FAISS `IndexFlatIP` over L2-normalised
vectors is exhaustive inner-product search, which `torch.topk` on a single
matmul reproduces bitwise, and that identity is asserted rather than assumed.
The gallery is 1,000 pre-extracted 128-d image features built the way the
lesson's own `main` builds them -- 10 class prototypes planted in the first 64
dimensions under Gaussian filler -- and the 20 queries are caption features, one
per randomly chosen indexed image. What the exercise asks for next is the trap.
Recall@5 of the *paired* image comes back at 0.000-0.050 across three seeds --
0 or 1 query of 20 finds its own image, a mean of 0.0333 against the k/N = 0.005
that random embeddings give. The same retrieval puts the right *class* in the top
5 for 20 of 20 queries. The paired image sits at median rank ~60 of 1,000
because 99 gallery images share its class and
therefore its caption, and no contrastive objective ever asked the tower to tell
them apart -- instance recall measures an identity the data does not carry. 20
queries also quantise recall@5 to multiples of 0.05, so the exercise's protocol
cannot resolve anything finer. One lesson claim fails along the way: `docs/en.md`
says the initial loss "should be close to log(N)", which is true only at
temperature 1; at the lesson's own `logit_scale` of e^2.6592 it runs 1.16-1.52x
higher.

Structure: `retrieve` runs the lesson's own `forward` over the
whole gallery and the 20 queries, then reports instance and class-level recall@5,
the paired image's rank, the topk-versus-argsort identity, the embedding norms,
softmax row sums, tower-swap symmetry and top-k invariance to the temperature;
`null_recall` measures recall@k for purely random embeddings over 200 draws; and
`log_n_claim` weighs the docs' log(N) claim at four batch sizes. The gallery and
the training loop are inlined in `solve`, which runs three seeds. At 146 code
lines this sits above D14's 120-line target and 4 clear of the ceiling: five
checks over four probes.
"""

from __future__ import annotations

import importlib
import math

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "18-open-vocab-clip"

IMG_IN, TXT_IN, CLASSES, GALLERY, QUERIES, TOPK = 128, 64, 10, 1000, 20, 5
BATCH, LR, JITTER, STEPS = 64, 3e-3, 0.05, 300
GALLERY_SEED, QUERY_SEED, NULL_SEED, TRIALS = 1234, 31, 500, 200
SEEDS, KS, SCALES, BATCHES = (0, 1, 2), (1, 5, 10, 50), (0.01, 1.0, 14.285, 100.0), (8, 32, 64, 256)
MISSING = ("faiss", "open_clip")

pull = lambda runs, key: [runs[s][key] for s in SEEDS]              # noqa: E731 - a projection
avg = lambda values: sum(values) / len(values)                       # noqa: E731 - a projection
warm = lambda rows: [hot / math.log(n) for n, hot, _ in rows]       # noqa: E731 - a projection
cool = lambda rows: [abs(cold - math.log(n)) for n, _, cold in rows]  # noqa: E731 - a projection
flat = lambda table: max(abs(got - k / GALLERY) for k, got in table)  # noqa: E731 - a projection
null_row = lambda table: "  ".join(f"@{k}:{got:.5f} vs {k / GALLERY:.5f}" for k, got in table)  # noqa: E731
loss_row = lambda rows: "  ".join(f"N={n}: {hot:.3f} / {cold:.3f} vs {math.log(n):.3f}" for n, hot, cold in rows)  # noqa: E731


def retrieve(ref, torch, model, feats, labels, protos) -> dict:
    gen = torch.Generator().manual_seed(QUERY_SEED)
    targets = torch.randperm(GALLERY, generator=gen)[:QUERIES]
    asked = protos[labels[targets]] + JITTER * torch.randn(QUERIES, TXT_IN, generator=gen)
    with torch.no_grad():
        index, queried, scale = model(feats, asked)
        paired, texts, _ = model(feats[targets], asked)
        both = (ref.clip_loss(paired, texts, scale).item(), ref.clip_loss(texts, paired, scale).item())
    sim, norms = queried @ index.T, index.norm(dim=-1)
    top, rows = sim.topk(TOPK, dim=-1).indices, (scale * sim).softmax(dim=-1).sum(dim=-1)
    return {"instance": (top == targets[:, None]).any(-1).float().mean().item(), "scale": scale.item(), "both": both,
            "classwise": (labels[top] == labels[targets][:, None]).any(-1).float().mean().item(), "rows": (rows.min().item(), rows.max().item()),
            "rank": (sim > sim.gather(1, targets[:, None])).sum(-1).float().median().item(), "norm": (norms.min().item(), norms.max().item()),
            "exact": torch.equal(top, sim.argsort(dim=-1, descending=True)[:, :TOPK]), "symmetry": (sim - (index @ queried.T).T).abs().max().item(),
            "invariant": all(torch.equal(top, (s * sim).topk(TOPK, dim=-1).indices) for s in SCALES)}


def null_recall(torch, functional) -> list:
    hits = {k: 0.0 for k in KS}
    for trial in range(TRIALS):
        gen = torch.Generator().manual_seed(NULL_SEED + trial)
        shelf = functional.normalize(torch.randn(GALLERY, TXT_IN, generator=gen), dim=-1)
        sim = functional.normalize(torch.randn(QUERIES, TXT_IN, generator=gen), dim=-1) @ shelf.T
        targets = torch.randperm(GALLERY, generator=gen)[:QUERIES]
        for k in KS:
            hits[k] += (sim.topk(k, dim=-1).indices == targets[:, None]).any(-1).float().sum().item()
    return [(k, hits[k] / (TRIALS * QUERIES)) for k in KS]


def log_n_claim(ref, torch) -> list:
    rows = []
    for size in BATCHES:
        torch.manual_seed(0)
        model = ref.TwoTower()
        img_emb, txt_emb, scale = model(torch.randn(size, IMG_IN), torch.randn(size, TXT_IN))
        rows.append((size, ref.clip_loss(img_emb, txt_emb, scale).item(),
                     ref.clip_loss(img_emb, txt_emb, torch.tensor(1.0)).item()))
    return rows


def solve():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    setup = torch.Generator().manual_seed(GALLERY_SEED)
    protos = functional.normalize(torch.randn(CLASSES, TXT_IN, generator=setup), dim=-1)
    labels, feats = torch.arange(GALLERY) % CLASSES, torch.randn(GALLERY, IMG_IN, generator=setup)
    feats[:, :TXT_IN] = protos[labels] + JITTER * torch.randn(GALLERY, TXT_IN, generator=setup)
    runs, losses, absent = {}, {}, []
    for name in MISSING:
        try:
            importlib.import_module(name)
            absent.append(f"{name} -> present")
        except ModuleNotFoundError as exc:
            absent.append(f"{name} -> {type(exc).__name__}: {exc}")
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = ref.TwoTower(img_in=IMG_IN, txt_in=TXT_IN)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        gen = torch.Generator().manual_seed(100 + seed)
        for _ in range(STEPS):
            pick = torch.randint(0, GALLERY, (BATCH,), generator=gen)
            caption = protos[labels[pick]] + JITTER * torch.randn(BATCH, TXT_IN, generator=gen)
            loss = ref.clip_loss(*model(feats[pick], caption))
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses[seed], runs[seed] = loss.item(), retrieve(ref, torch, model, feats, labels, protos)
    return {"absent": "; ".join(absent), "runs": runs, "null": null_recall(torch, functional), "logn": log_n_claim(ref, torch),
            "loss": [losses[s] for s in SEEDS], "claimed": "log(N) = log(8) = 2.08" in parity.doc_text(PHASE, LESSON)}


def verify(result):
    runs, null, logn = result["runs"], result["null"], result["logn"]
    first, instance, ranks = runs[SEEDS[0]], pull(runs, "instance"), pull(runs, "rank")
    return [
        practice.Check(
            "ANSWER: recall@5 is 0.000-0.050 on a 1,000-image index — and there is no FAISS to build one",
            max(instance) < 2 / QUERIES and first["exact"],
            f"{result['absent']}. A FAISS `IndexFlatIP` over normalised vectors is exhaustive inner-product search, so the index here is one matmul: "
            f"`topk({TOPK})` on it equals the full `argsort` prefix bitwise ({first['exact']}) and every embedding has norm "
            f"{first['norm'][0]:.7f}-{first['norm'][1]:.7f}. Over {len(SEEDS)} towers trained on the lesson's own `clip_loss` (final "
            f"{min(result['loss']):.3f}-{max(result['loss']):.3f}), {QUERIES} queries against {GALLERY:,} images give instance recall@{TOPK} of "
            f"{[round(r, 3) for r in instance]}, mean {avg(instance):.4f}: 0 or 1 query of {QUERIES} finds its own image"),
        practice.Check(
            "FINDING: that number blames the metric, not the tower — class-level recall@5 is 1.000",
            min(pull(runs, "classwise")) == 1.0 and min(ranks) > 20,
            f"the same retrieval puts the query's *class* in the top {TOPK} for {QUERIES} of {QUERIES} queries on all {len(SEEDS)} seeds, while the "
            f"paired image sits at median rank {min(ranks):.0f}-{max(ranks):.0f} of {GALLERY:,}. Every gallery image shares its class -- and so its "
            f"caption -- with {GALLERY // CLASSES - 1} others and `clip_loss` never asked the tower to separate them; {QUERIES} queries also "
            f"quantise recall@{TOPK} to multiples of {1 / QUERIES:.3f}"),
        practice.Check(
            "CONTROL: under random embeddings recall@k is exactly k/N — the null for every number above",
            flat(null) < 0.002,
            f"{TRIALS} independent draws of {GALLERY:,} random unit vectors and {QUERIES} random queries, {TRIALS * QUERIES:,} queries in all: "
            f"{null_row(null)}, worst deviation {flat(null):.5f}. The trained tower's mean {avg(instance):.4f} at k={TOPK} is "
            f"{avg(instance) / (TOPK / GALLERY):.0f}x that floor and still {1 - avg(instance):.1%} misses"),
        practice.Check(
            "MECHANISM: the temperature scales logits exactly and cannot move one retrieval result",
            first["invariant"] and first["symmetry"] == 0.0 and abs(first["rows"][0] - 1.0) < 1e-05,
            f"top-{TOPK} is bit-identical across scales {SCALES}, which bracket this tower's `logit_scale.exp()` of {first['scale']:.3f}: a positive "
            f"scalar cannot reorder a row. The scaled matrix stays a distribution -- softmax rows sum to {first['rows'][0]:.7f}-"
            f"{first['rows'][1]:.7f} -- and swapping the towers transposes it to exactly {first['symmetry']:.1f}, so `clip_loss` agrees both ways: "
            f"{first['both'][0]:.6f} and {first['both'][1]:.6f}"),
        practice.Check(
            "CONTROL: the docs' 'loss close to log(N)' holds only at temperature 1, not at the shipped init",
            result["claimed"] and min(warm(logn)) > 1.15 and max(cool(logn)) < 0.05,
            f"`docs/en.md` says the initial loss should be close to log(N). Untrained `clip_loss` at the lesson's own `logit_scale` / the same "
            f"loss at scale 1.0 / log(N): {loss_row(logn)}. Scale 1.0 matches log(N) to {max(cool(logn)):.3f}, while the shipped e^2.6592 runs "
            f"{min(warm(logn)):.2f}-{max(warm(logn)):.2f}x above it -- sharpening the softmax raises a random model's loss above chance"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
