"""Exercise 5 -- the reloaded projector reproduces training exactly, but only behind an encoder train() never saves.

    Persist projector weights, then reload and run inference without the vision encoder backward pass to verify that only the projector is needed at deploy time.

Reading of the exercise: the projector from the lesson's own `train()` (seed
0, 200 steps) is saved as a `state_dict` with `torch.save`, loaded with
`weights_only=True` into a fresh `MLPProjector`, and run under
`torch.inference_mode()` on the 32 training pairs through the frozen encoder
`train()` built. "Only the projector is needed" is tested by what the deploy
side must carry: the checkpoint alone, then the checkpoint behind a
different encoder (same config, seed 1) and behind no image at all (zeros).

**ANSWER: verified.** The checkpoint is one 5.25 MB file holding 1,312,256
parameters (4 tensors); the reloaded head reproduces the trained head's
embeddings exactly (max abs difference 0.0) and its mean cosine to the
captions, 0.1867. Inference builds no graph: the output does not require
grad and none of the encoder's 19,501,056 parameters has a `.grad`.

**FINDING: the projector alone is not deployable -- it needs the exact
random encoder, which `train()` never saves.** Behind an encoder of the same
config built at seed 1 the reloaded head scores cosine -0.045 (the untrained
head's step-0 cosine was -0.068), and on all-zeros features 0.040. The head
learned one output direction (exercise 3), but reaches it through fc1 acting
on the seed-0 encoder's near-constant CLS vector, not through its bias. That
encoder exists only as the `torch.manual_seed(0)` call inside `train()`, so a
deploy must re-create it from the seed or ship its 19.5M weights too.

**FINDING: the lesson's encoder is not the 86M-parameter one it describes.**
"The vision encoder has 86M parameters" -- `train()` builds a depth-4,
mlp_ratio-2 ViT with 19,501,056.

Structure: `trained()` runs `train()` and records its encoder; `solve()` does
the save/load round trip in a temporary directory and scores each deploy
variant.
"""

from __future__ import annotations

import pathlib
import tempfile

import torch
import torch.nn.functional as F

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "60-projection-layer-modality-align"


def trained(ref):
    seen, make_enc = {}, ref.VisionEncoder

    def encoder(cfg):
        seen["enc"] = make_enc(cfg)
        return seen["enc"]

    ref.VisionEncoder = encoder
    try:
        with parity.quiet():
            proj, _ = ref.train(ref.AlignConfig())
    finally:
        ref.VisionEncoder = make_enc
    return proj, seen["enc"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = ref.AlignConfig()
    proj, enc = trained(ref)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "projector.pt"
        torch.save(proj.state_dict(), path)
        size, state = path.stat().st_size, torch.load(path, weights_only=True)
    head = ref.MLPProjector(cfg.vision_hidden, cfg.projection_hidden, cfg.text_hidden).eval()
    head.load_state_dict(state)
    torch.manual_seed(1)
    other = ref.VisionEncoder(enc.cfg).eval()
    text = ref.MockTextEmbedding(cfg.vocab_size, cfg.text_hidden, seed=cfg.seed + 1)
    pairs = [ref.make_pair(cfg.seed + 1000 + i, cfg.vocab_size, cfg.max_caption_len)
             for i in range(cfg.pairs)]
    imgs, t = torch.cat([img for img, _ in pairs]), torch.cat([text(ids) for _, ids in pairs])
    with torch.inference_mode():
        x = enc(imgs)[1]
        out = head(x)
        cos = {name: F.cosine_similarity(head(feats), t).mean().item()
               for name, feats in [("real", x), ("seed1", other(imgs)[1]), ("zeros", torch.zeros_like(x))]}
        diff = (out - proj(x)).abs().max().item()
    return {"size_mb": size / 1e6, "tensors": len(state), "params": sum(v.numel() for v in state.values()),
            "diff": diff, "cos": cos, "out_grad": out.requires_grad,
            "enc_grads": sum(p.grad is not None for p in enc.parameters()),
            "enc_params": sum(p.numel() for p in enc.parameters()),
            "doc_86m": "86M parameters" in parity.doc_text(PHASE, LESSON)}


def verify(result):
    r, cos = result, result["cos"]
    return [
        practice.Check(
            "ANSWER: verified",
            all([
                abs(r["size_mb"] - 5.25) < 0.01,
                (r["tensors"], r["params"]) == (4, 1_312_256),
                r["diff"] == 0.0,
                abs(cos["real"] - 0.1867) < 2e-3,
                not r["out_grad"],
                r["enc_grads"] == 0,
            ]),
            f"{r['size_mb']:.2f} MB, {r['tensors']} tensors, {r['params']:,} params; max diff "
            f"{r['diff']}; cos {cos['real']:.4f}; output requires grad {r['out_grad']}; encoder "
            f"params with .grad {r['enc_grads']}",
        ),
        practice.Check(
            "FINDING: the projector alone is not deployable, it needs the unsaved seed-0 encoder",
            all([
                abs(cos["seed1"] + 0.045) < 2e-3,
                abs(cos["zeros"] - 0.040) < 2e-3,
                cos["real"] - max(cos["seed1"], cos["zeros"]) > 0.1,
            ]),
            f"cos behind the real encoder {cos['real']:.4f}, a seed-1 encoder {cos['seed1']:.4f}, "
            f"zeros {cos['zeros']:.4f}",
        ),
        practice.Check(
            "FINDING: the lesson's encoder is not the 86M-parameter one it describes",
            all([
                r["doc_86m"],
                r["enc_params"] == 19_501_056,
            ]),
            f"doc says 86M: {r['doc_86m']}; train() builds {r['enc_params']:,}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
