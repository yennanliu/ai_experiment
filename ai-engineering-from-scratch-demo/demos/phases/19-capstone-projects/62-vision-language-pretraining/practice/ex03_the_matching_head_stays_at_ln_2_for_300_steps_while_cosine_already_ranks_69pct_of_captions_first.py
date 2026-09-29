"""Exercise 3 -- the matching head stays at ln 2 for 300 steps while cosine already ranks 69% of captions first.

    Add an image-text matching binary head on top of the joint embedding (true/false: do these match?) for a third loss, replicating BLIP's three-head setup.

Reading of the exercise: the head is one `nn.Linear(3D, 1)` over
[z_img, z_txt, z_img * z_txt], the L2-normalised joint embeddings that
InfoNCE already uses (D = 128). As in BLIP, every image in a batch gives one
positive (its own caption) and one hard negative (the in-batch caption most
similar to it), and the head is trained with binary cross-entropy. The total
loss is contrastive + LM + ITM, with everything else as in the lesson.
Seeds 0 and 1 are run for 300 steps with and without the head, and the
seed-0 run with the head continues to 700 steps. The measures come from 12 fixed 16-pair corpus
blocks: ITM accuracy on the 32 pairs of each block, "cosine rank" (the
share of images whose own caption beats their hardest negative, which is
in-batch top-1), and the contrastive loss.

**ANSWER: the three-loss model trains, but the ITM head learns nothing in
300 steps.** Its loss starts and ends at about ln 2 = 0.693 (0.692 -> 0.694
on seed 0, 0.698 -> 0.692 on seed 1), and its accuracy is 0.487 and 0.547:
chance. In the same models the cosine that InfoNCE learned ranks the own
caption first for 0.688 and 0.786 of images.

**FINDING: the head only learns once ranking is already solved.** At 700
steps its accuracy reaches 0.727, but by then the contrastive loss is 0.015
and cosine rank is 1.000. On unit vectors the head's inputs are about
1/sqrt(128) per coordinate, so at lr 5e-4 its logits stay near zero for
hundreds of steps. BLIP's ITM head reads a fused cross-attention encoder,
not the two embeddings being ranked.

**FINDING: the third loss does not consistently help ranking.** Adding ITM
moves the fixed-block contrastive loss after 300 steps from 0.615 to 0.793
on seed 0 (worse) and from 0.672 to 0.621 on seed 1 (better).

Structure: `itm_batch()` builds positives and hard negatives; `train()` is
the lesson's loop with the head's loss added; `measure()` scores the blocks.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "62-vision-language-pretraining"


def itm_batch(image_emb, text_emb):
    """Features and labels: each image with its caption, then with its hardest negative."""
    zi, zt = F.normalize(image_emb, dim=-1), F.normalize(text_emb, dim=-1)
    sim = zi @ zt.T
    neg = sim.detach().clone().fill_diagonal_(-float("inf")).argmax(1)
    x = torch.cat([torch.cat([zi, zt, zi * zt], 1), torch.cat([zi, zt[neg], zi * zt[neg]], 1)])
    y = torch.cat([torch.ones(len(zi)), torch.zeros(len(zi))])
    return x, y, (sim.argmax(1) == torch.arange(len(zi))).float()


def measure(ref, model, head, corpus):
    acc, rank, nce = [], [], []
    with torch.no_grad():
        imgs, ids = ref.sample_batch(corpus, list(range(192)))
        img, txt = model.encode_image(imgs)[1], model.text_encoder(ids)
        for b in range(0, 192, 16):
            x, y, top1 = itm_batch(img[b:b + 16], txt[b:b + 16])
            acc.append(((head(x).squeeze(-1) > 0).float() == y).float().mean().item())
            rank.append(top1.mean().item())
            nce.append(ref.info_nce_loss(img[b:b + 16], txt[b:b + 16], model.log_tau)[0].item())
    return [round(sum(v) / 12, 3) for v in (acc, rank, nce)]


def train(ref, seed, itm, steps, marks=(300,)):
    cfg = ref.PretrainConfig(seed=seed)
    torch.manual_seed(cfg.seed)
    model = ref.MultimodalModel(cfg).train()
    head = torch.nn.Linear(3 * cfg.embed_dim, 1)
    params = list(model.parameters()) + (list(head.parameters()) if itm else [])
    opt = torch.optim.Adam(params, lr=cfg.lr)
    corpus = ref.make_mock_corpus(cfg.seed + 1, cfg.n_pairs, cfg.text_vocab, cfg.max_text_len)
    rng = np.random.default_rng(cfg.seed + 2)
    itm_loss, blocks = [], []
    for step in range(1, steps + 1):
        idx = rng.choice(len(corpus), size=cfg.batch_size, replace=False).tolist()
        imgs, ids = ref.sample_batch(corpus, idx)
        contrast, lm, _ = model(imgs, ids)
        total = contrast + cfg.lm_weight * lm
        if itm:
            x, y, _ = itm_batch(model.encode_image(imgs)[1], model.text_encoder(ids))
            loss = F.binary_cross_entropy_with_logits(head(x).squeeze(-1), y)
            total = total + loss
            itm_loss.append(round(loss.item(), 3))
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if step in marks:
            blocks.append(measure(ref, model, head, corpus))
    return {"itm": [itm_loss[0], itm_loss[299]] if itm else [], "blocks": blocks[0], "late": blocks[-1]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {(s, m): train(ref, s, m, 300) for s in (0, 1) for m in (False, True) if (s, m) != (0, True)}
    runs[(0, True)] = train(ref, 0, True, 700, marks=(300, 700))
    return {"runs": runs, "long": runs[(0, True)]["late"],
            "nce": [[runs[(s, m)]["blocks"][2] for m in (False, True)] for s in (0, 1)]}


def verify(result):
    runs, late = result["runs"], result["long"]
    with_itm = [runs[(s, True)] for s in (0, 1)]
    nce = result["nce"]
    return [
        practice.Check(
            "ANSWER: the three-loss model trains, but the ITM head learns nothing in 300 steps",
            [r["itm"] for r in with_itm] == [[0.692, 0.694], [0.698, 0.692]]
            and [r["blocks"][:2] for r in with_itm] == [[0.487, 0.688], [0.547, 0.786]],
            f"ITM loss first/last per seed {[r['itm'] for r in with_itm]}; "
            f"[ITM accuracy, cosine rank] {[r['blocks'][:2] for r in with_itm]}",
        ),
        practice.Check(
            "FINDING: the head only learns once ranking is already solved",
            late == [0.727, 1.0, 0.015],
            f"700 steps: ITM accuracy {late[0]}, cosine rank {late[1]}, contrastive loss {late[2]}",
        ),
        practice.Check(
            "FINDING: the third loss does not consistently help the ranking objective",
            nce == [[0.615, 0.793], [0.672, 0.621]],
            f"fixed-block contrastive loss [without, with] ITM per seed {nce}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
