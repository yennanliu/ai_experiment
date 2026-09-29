"""Exercise 1 -- with the paper's -10 bias SigLIP reaches 31% in-batch accuracy where InfoNCE reaches 73%.

    Replace InfoNCE with SigLIP-style sigmoid pair loss and compare convergence on the mock corpus.

Reading of the exercise: the lesson's `info_nce_loss` is swapped, inside
`MultimodalModel.forward`, for Algorithm 1 of the SigLIP paper
(https://arxiv.org/pdf/2303.15343, read 2026-09-29):
`logits = z_img z_txt^T * t + b`, `labels = 2 I - 1`,
`loss = -sum(log_sigmoid(labels * logits)) / n`, with t = exp(t') and
t' = log 10, b = -10 at init. Everything else is the lesson's run: same seed,
model, corpus, Adam at 5e-4, batch 16, LM loss added at weight 1. The two
losses have different scales, so convergence is compared on one neutral
measure: image-to-text top-1 accuracy inside 12 fixed blocks of 16 pairs
(chance 1/16 = 0.0625), at the lesson's 50 steps and at 300.

**ANSWER: SigLIP converges more slowly here.** In-batch accuracy after 50
steps is 0.151 for InfoNCE and 0.094 for SigLIP; after 300 steps it is 0.729
against 0.312. The untrained model scores 0.083 under both.

**FINDING: the paper's b = -10 causes most of the gap, and at batch 16 it
barely moves.** Adam at lr 5e-4 moves the bias by at most about 5e-4 a step: it is
-9.975 after 50 steps and -9.852 after 300. The paper chose -10 for batches of
thousands, where almost every pair is a negative. Start the bias at
-ln(N - 1) = -2.71, the prior for 1 positive in 16, and SigLIP scores 0.146
at 50 steps, level with InfoNCE, and 0.651 at 300 (InfoNCE: 0.729).

**FINDING: the "tau" the lesson prints is the inverse temperature.** It
prints `initial tau: 14.286`. That number is exp(log_tau) = 1/0.07, the
factor the similarities are multiplied by. The doc's "too small (e.g. tau =
0.01)" refers to the temperature itself, which is 1/14.286 = 0.07.

Structure: `train()` replays the lesson's loop step for step (checked against
`ref.train` on its 50-step history) with the loss swapped by a patch that is
always restored; `accuracy()` scores the fixed blocks.
"""

from __future__ import annotations

import contextlib
import io
import math

from harness import parity, practice

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "62-vision-language-pretraining"


def accuracy(ref, model, corpus):
    """Image-to-text top-1 inside 12 fixed blocks of 16 pairs."""
    with torch.no_grad():
        imgs, ids = ref.sample_batch(corpus, list(range(192)))
        img = F.normalize(model.encode_image(imgs)[1], dim=-1)
        sim = img @ F.normalize(model.text_encoder(ids), dim=-1).T
    hits = [(sim[b:b + 16, b:b + 16].argmax(1) == torch.arange(16)).sum().item()
            for b in range(0, 192, 16)]
    return round(sum(hits) / 192, 3)


def siglip(model):
    def loss(image_emb, text_emb, log_tau):
        z = F.normalize(image_emb, dim=-1) @ F.normalize(text_emb, dim=-1).T
        logits = z * log_tau.exp() + model.sig_bias
        labels = 2 * torch.eye(len(z)) - 1
        return -F.logsigmoid(labels * logits).sum() / len(z), logits
    return loss


def train(ref, bias=None, steps=300, marks=(0, 50, 300)):
    """The lesson's loop; bias=None keeps InfoNCE, a float switches to SigLIP."""
    cfg = ref.PretrainConfig()
    torch.manual_seed(cfg.seed)
    model = ref.MultimodalModel(cfg).train()
    saved = ref.info_nce_loss
    if bias is not None:
        model.sig_bias = torch.nn.Parameter(torch.tensor(bias))
        with torch.no_grad():
            model.log_tau.fill_(math.log(10.0))
        ref.info_nce_loss = siglip(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    corpus = ref.make_mock_corpus(cfg.seed + 1, cfg.n_pairs, cfg.text_vocab, cfg.max_text_len)
    rng = np.random.default_rng(cfg.seed + 2)
    out = {"acc": [], "contrast": [], "bias": []}
    try:
        for step in range(steps + 1):
            if step in marks:
                out["acc"].append(accuracy(ref, model, corpus))
                out["bias"].append(round(getattr(model, "sig_bias", torch.tensor(0.0)).item(), 3))
            if step == steps:
                break
            idx = rng.choice(len(corpus), size=cfg.batch_size, replace=False).tolist()
            contrast, lm, _ = model(*ref.sample_batch(corpus, idx))
            opt.zero_grad(set_to_none=True)
            (contrast + cfg.lm_weight * lm).backward()
            opt.step()
            out["contrast"].append(contrast.item())
    finally:
        ref.info_nce_loss = saved
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with contextlib.redirect_stdout(io.StringIO()) as log:
        shipped = ref.train(ref.PretrainConfig())
        ref.main()
    nce = train(ref)
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "parity": nce["contrast"][:50] == shipped["contrast"],
        "nce": nce["acc"], "paper": train(ref, -10.0), "prior": train(ref, -math.log(15)),
        "printed_tau": "initial tau    : 14.286" in log.getvalue(),
        "doc_tau": "Too small (e.g. `tau = 0.01`)" in doc,
        "scale": round(math.exp(ref.PretrainConfig().init_log_tau), 3),
    }


def verify(result):
    r, paper, prior = result, result["paper"], result["prior"]
    return [
        practice.Check(
            "ANSWER: SigLIP converges more slowly here",
            r["parity"] and r["nce"] == [0.083, 0.151, 0.729] and paper["acc"] == [0.083, 0.094, 0.312],
            f"in-batch top-1 at steps 0/50/300: InfoNCE {r['nce']}, SigLIP {paper['acc']} "
            f"(loop matches ref.train: {r['parity']})",
        ),
        practice.Check(
            "FINDING: the paper's b = -10 causes most of the gap, and at batch 16 it barely moves",
            paper["bias"] == [-10.0, -9.975, -9.852] and prior["acc"] == [0.083, 0.146, 0.651],
            f"bias at steps 0/50/300 {paper['bias']}; starting at -ln 15: accuracy {prior['acc']}",
        ),
        practice.Check(
            "FINDING: the 'tau' the lesson prints is the inverse temperature",
            r["printed_tau"] and r["doc_tau"] and r["scale"] == 14.286
            and round(1 / r["scale"], 2) == 0.07,
            f"printed tau {r['scale']} = 1/{1 / r['scale']:.2f}; doc discusses tau = 0.01",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
