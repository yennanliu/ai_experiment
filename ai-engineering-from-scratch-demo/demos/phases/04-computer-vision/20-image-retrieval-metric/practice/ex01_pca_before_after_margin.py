"""Exercise 1 — pca before after margin.

    **(Easy)** Run the toy example above. Plot the embeddings with PCA before and
    after training to see the six clusters form.

Reading of the exercise: "to see the six clusters form" assumes they are not
there beforehand, and on this data they are. `main()` builds each sample as
`protos[label] + 0.15 * randn`, six unit prototypes in 128 dimensions, so the
clusters are in the *input* -- retrieval on the raw vectors already scores
recall@1 = 1.000 before any encoder exists. What training forms is margin, not
clusters, so the plot is quantified rather than described: a between-class over
within-class variance ratio, recall@1, and the PCA spectrum the plot is a
projection of. Running the example as written also exposes two properties of the
lesson's own loop that a picture cannot show, so both are measured here.

Structure: `world` returns the six prototypes and a sampler drawing from them;
`spread` is the between-over-within variance ratio on a labelled embedding;
`spectrum` returns per-component explained variance -- what a two-axis plot
keeps and discards; `run` trains the lesson's own Encoder with its own loss and
mining, recording the loss history and how often the mining fell back to the
hardest negative. `plane` in `solve` is that two-component projection itself, so
the plot the exercise asks for can be scored rather than looked at.

At 144 lines of code this file is over D14's 120-line target and under its
150-line ceiling: the overrun is the instrumented copy of the lesson's mining
window -- seven lines, and the only way to count fallbacks without editing the
reference -- plus six checks whose evidence carries 28 measured numbers.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "20-image-retrieval-metric"

CLASSES, DIM, EMB, NOISE = 6, 128, 64, 0.15
STEPS, BATCH, LR, MARGIN, PROBE = 200, 48, 3e-3, 0.2, 300

split = lambda emb, labels, n=50: (emb[:n], emb[n:], labels[:n], labels[n:])   # noqa: E731


def world(torch, functional):
    torch.manual_seed(0)
    protos = functional.normalize(torch.randn(CLASSES, DIM), dim=-1)

    def sample(count, gen):
        labels = torch.randint(0, CLASSES, (count,), generator=gen)
        return protos[labels] + NOISE * torch.randn(count, DIM, generator=gen), labels

    return sample


def spread(torch, emb, labels) -> float:
    means = torch.stack([emb[labels == c].mean(0) for c in range(CLASSES)])
    within = torch.stack([((emb[labels == c] - means[c]) ** 2).sum(1).mean()
                          for c in range(CLASSES)]).mean()
    return float(((means - emb.mean(0)) ** 2).sum(1).mean() / within)


def spectrum(torch, emb) -> list:
    centred = emb - emb.mean(0)
    values = torch.pca_lowrank(centred, q=10)[1] ** 2
    return [float(v) for v in values / values.sum()]


def run(torch, ref, encoder, sample) -> dict:
    optimiser = torch.optim.Adam(encoder.parameters(), lr=LR)
    gen, history, stranded = torch.Generator().manual_seed(0), [], []
    for _ in range(STEPS):
        batch, labels = sample(BATCH, gen)
        emb = encoder(batch)
        pos, neg = ref.semi_hard_negatives(emb, labels, margin=MARGIN)
        distance = torch.cdist(emb, emb)
        window = distance.clone()
        window[labels[:, None] == labels[None, :]] = float("inf")
        anchor = distance[torch.arange(BATCH), pos].unsqueeze(1)
        window[(distance <= anchor) | (distance >= anchor + MARGIN)] = float("inf")
        stranded.append(int((window.min(1).values == float("inf")).sum()))
        loss = ref.triplet_loss(emb, emb[pos], emb[neg], margin=MARGIN)
        history.append(float(loss.detach()))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return {"history": history, "stranded": stranded}


def solve():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    sample = world(torch, functional)
    probe, labels = sample(PROBE, torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    encoder = ref.Encoder(in_dim=DIM, emb_dim=EMB)
    with torch.no_grad():
        before = encoder(probe)
    trained = run(torch, ref, encoder, sample)
    encoder.eval()
    with torch.no_grad():
        after = encoder(probe)
    flat = functional.normalize(probe, dim=-1)
    plane = functional.normalize((after - after.mean(0)) @ torch.pca_lowrank(
        after - after.mean(0), q=10)[2][:, :2], dim=-1)
    return dict(trained, spectrum=spectrum(torch, after),
                spread={"raw": spread(torch, flat, labels), "before": spread(torch, before, labels),
                        "after": spread(torch, after, labels)},
                recall={name: ref.recall_at_k(*split(emb, labels), k=1)
                        for name, emb in (("raw", flat), ("before", before),
                                          ("after", after), ("plane", plane))})


def verify(result):
    spread_, recall, curve = result["spread"], result["recall"], result["history"]
    variance = result["spectrum"]
    zeros = [i for i, v in enumerate(curve) if v == 0.0]
    stranded = result["stranded"]
    return [
        practice.Check(
            "ANSWER: the margin grows 62x -- that is what the two plots differ by",
            spread_["after"] > 50 * spread_["before"] and recall["after"] == 1.0,
            f"between-class over within-class variance goes {spread_['before']:.3f} -> "
            f"{spread_['after']:.3f} ({spread_['after'] / spread_['before']:.0f}x) and recall@1 "
            f"{recall['before']:.3f} -> {recall['after']:.3f}, with the triplet loss ending at "
            f"{curve[-1]:.5f} from {curve[0]:.5f}"),
        practice.Check(
            "FINDING: the six clusters are in the input -- nothing forms them",
            recall["raw"] == 1.0 and spread_["raw"] < spread_["after"],
            f"the raw 128-d samples already retrieve at recall@1 {recall['raw']:.3f}, on a variance "
            f"ratio of only {spread_['raw']:.3f}: `protos[label] + {NOISE} * randn` *is* six "
            f"clusters. Training multiplies the margin by {spread_['after'] / spread_['raw']:.0f}x; "
            "it does not create the structure the exercise asks you to watch appear"),
        practice.Check(
            "CONTROL: 'before training' is worse than no encoder at all",
            recall["before"] < recall["raw"] == 1.0,
            f"the untrained Encoder is a random projection 128 -> {EMB}, and it *loses* separability "
            f"the input had: recall@1 {recall['before']:.3f} against the raw {recall['raw']:.3f}, "
            f"variance ratio {spread_['before']:.3f} against {spread_['raw']:.3f}. The left-hand plot "
            "is not a neutral starting point"),
        practice.Check(
            "MECHANISM: six clusters span exactly five dimensions, so two axes show 46%",
            abs(variance[4] / variance[5]) > 50 and sum(variance[:2]) < 0.5,
            "explained variance per component " + " ".join(f"{v:.4f}" for v in variance[:6])
            + f": five near-equal directions then a {variance[4] / variance[5]:.0f}x collapse at the "
            f"sixth. C centroids span C-1 dimensions, so the plot keeps {sum(variance[:2]):.1%} of a "
            f"five-dimensional object -- though on data this easy the discarded half costs nothing: "
            f"recall@1 read from that two-component projection alone is still {recall['plane']:.3f}"),
        practice.Check(
            "FINDING: two thirds of the run has no gradient",
            len(zeros) > STEPS // 2 and zeros[0] < STEPS // 10,
            f"the triplet loss first reaches exactly 0.0 at step {zeros[0]} and is zero on "
            f"{len(zeros)} of {STEPS} steps -- {len(zeros) / STEPS:.0%} of the run updating nothing, "
            f"because relu clamps the loss once every triplet clears the {MARGIN} margin. The curve "
            "reads " + " ".join(f"{curve[i]:.4f}" for i in (0, 40, 80, 120, 199))),
        practice.Check(
            "CONTROL: `semi_hard_negatives` is hardest-negative mining in practice",
            sum(stranded) / STEPS > 0.8 * BATCH,
            f"the semi-hard window d_ap < d_an < d_ap + {MARGIN} is empty for a mean "
            f"{sum(stranded) / STEPS:.1f} of {BATCH} anchors -- {stranded[0]} at the first step and "
            f"{stranded[-1]} at the last -- so the function's own fallback to the hardest negative "
            f"chooses for {sum(stranded) / STEPS / BATCH:.0%} of them. Its name describes the "
            "intent, not the behaviour on this data"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
