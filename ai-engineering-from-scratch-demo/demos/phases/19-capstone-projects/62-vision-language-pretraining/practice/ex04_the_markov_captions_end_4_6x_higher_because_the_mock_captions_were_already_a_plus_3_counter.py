"""Exercise 4 -- the Markov captions end 4.6x higher, because the mock captions were already a +3 counter.

    Replace the mock corpus with caption-id sequences drawn from a Markov chain whose transition matrix is conditioned on image hash. The captioning loss should drop further because there is actual learnable signal.

Reading of the exercise: the lesson's 200 images are kept, and so is each
caption's length (6-13 tokens). Each image's SHA-256 (of its float32 bytes)
modulo 4 picks one of 4 seeded transition tables. In each table every token
has 2 equally likely successors, so given the image the best possible
next-token loss is ln 2 = 0.693. The first token is uniform. The lesson's
run (seed 0, batch 16, Adam 5e-4, contrastive + LM) is trained on each
corpus for 300 steps. LM loss is read at step 50 and step 300 on the
training captions, on freshly drawn captions for the same images (held out),
and with each image's caption scored against the neighbouring image.

**ANSWER: no, the captioning loss drops less on the Markov corpus.** At the
lesson's 50 steps the LM loss on the training captions is 4.78 on the mock
corpus and 5.41 on the Markov one. At 300 steps it is 0.14 against 0.65.

**FINDING: the mock captions are not random; each one counts up by 3.**
In all 200 captions every token is the previous one plus 3 (mod 511), so
after the first token the next one is fully determined and the lowest
possible LM loss is 0. The doc calls them "random caption ids". The Markov
chain has a floor of ln 2 by construction, so the exercise's prediction
runs the wrong way.

**FINDING: the Markov model memorises its 200 captions instead of learning
the chain.** Its training loss at 300 steps, 0.652, is below the chain's own
floor of ln 2 = 0.693. On fresh captions for the same images it scores 5.47,
close to uniform over 511 tokens (6.24), although 507 of the 1,244 distinct
transitions in the fresh captions also occur in training.

**FINDING: only the Markov corpus makes the decoder read the image.** At
300 steps, scoring each caption against the neighbouring image moves the
mock loss from 0.142 to 0.152 and the Markov loss from 0.652 to 0.813. The
Markov table depends on the image; the mock's +3 rule does not.

Structure: `markov()` builds the corpus from the lesson's `make_mock_corpus`
images; `train()` is the lesson's loop; `lm()` scores a corpus.
"""

from __future__ import annotations

import hashlib
import math

from harness import parity, practice

try:
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "62-vision-language-pretraining"
CHAINS, BRANCH = 4, 2


def markov(ref, cfg, draw):
    """The lesson's images and lengths; captions from the image's own chain."""
    mock = ref.make_mock_corpus(cfg.seed + 1, cfg.n_pairs, cfg.text_vocab, cfg.max_text_len)
    table = np.random.default_rng(1234).integers(1, cfg.text_vocab, (CHAINS, cfg.text_vocab, BRANCH))
    out = []
    for i, (img, ids) in enumerate(mock):
        chain = int(hashlib.sha256(img.numpy().tobytes()).hexdigest(), 16) % CHAINS
        rng = np.random.default_rng([draw, i])
        toks = [int(rng.integers(1, cfg.text_vocab))]
        for _ in range(int((ids != 0).sum()) - 1):
            toks.append(int(table[chain, toks[-1], rng.integers(BRANCH)]))
        row = torch.zeros_like(ids)
        row[0, :len(toks)] = torch.tensor(toks)
        out.append((img, row))
    return out


def pairs(corpus):
    """(previous token, next token) transitions in a corpus."""
    return {(a, b) for _, ids in corpus
            for a, b in zip(ids[0].tolist(), ids[0, 1:].tolist()) if b}


def lm(ref, model, corpus, shift=0):
    with torch.no_grad():
        imgs, ids = ref.sample_batch(corpus, list(range(len(corpus))))
        return model(imgs.roll(shift, 0), ids)[1].item()


def train(ref, cfg, corpus, held=None):
    torch.manual_seed(cfg.seed)
    model = ref.MultimodalModel(cfg).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed + 2)
    marks = []
    for step in range(1, 301):
        idx = rng.choice(len(corpus), size=cfg.batch_size, replace=False).tolist()
        contrast, loss, _ = model(*ref.sample_batch(corpus, idx))
        opt.zero_grad(set_to_none=True)
        (contrast + cfg.lm_weight * loss).backward()
        opt.step()
        if step in (50, 300):
            marks.append(lm(ref, model, corpus))
    return {"train": marks, "wrong_image": lm(ref, model, corpus, 1),
            "held": lm(ref, model, held) if held else None}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = ref.PretrainConfig()
    mock = ref.make_mock_corpus(cfg.seed + 1, cfg.n_pairs, cfg.text_vocab, cfg.max_text_len)
    steps = set()
    for _, ids in mock:
        row = [t for t in ids[0].tolist() if t]
        steps |= {(b - a) % (cfg.text_vocab - 1) for a, b in zip(row, row[1:])}
    return {
        "mock": train(ref, cfg, mock),
        "markov": train(ref, cfg, markov(ref, cfg, 0), held=markov(ref, cfg, 1)),
        "seen": len(pairs(markov(ref, cfg, 1)) & pairs(markov(ref, cfg, 0))),
        "held_pairs": len(pairs(markov(ref, cfg, 1))),
        "steps": sorted(steps),
        "doc": "random caption ids" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    mock, mk = result["mock"], result["markov"]
    r2 = lambda xs: [round(x, 2) for x in xs]  # noqa: E731
    return [
        practice.Check(
            "ANSWER: no, the captioning loss drops less on the Markov corpus",
            r2(mock["train"]) == [4.78, 0.14] and r2(mk["train"]) == [5.41, 0.65],
            f"LM loss at steps 50/300: mock {r2(mock['train'])}, Markov {r2(mk['train'])}",
        ),
        practice.Check(
            "FINDING: the mock captions are not random; each one counts up by 3",
            result["steps"] == [3] and result["doc"],
            f"token-to-token steps across 200 captions: {result['steps']}",
        ),
        practice.Check(
            "FINDING: the Markov model memorises its 200 captions instead of learning the chain",
            mk["train"][1] < math.log(2) and round(mk["held"], 2) == 5.47
            and (result["seen"], result["held_pairs"]) == (507, 1244),
            f"train {mk['train'][1]:.3f} < ln 2 = {math.log(2):.3f}; fresh captions {mk['held']:.3f}; "
            f"{result['seen']}/{result['held_pairs']} fresh transitions occur in training",
        ),
        practice.Check(
            "FINDING: only the Markov corpus makes the decoder read the image",
            (round(mock["wrong_image"], 2), round(mk["wrong_image"], 2)) == (0.15, 0.81),
            f"right vs wrong image at 300 steps: mock {mock['train'][1]:.3f} vs "
            f"{mock['wrong_image']:.3f}, Markov {mk['train'][1]:.3f} vs {mk['wrong_image']:.3f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
