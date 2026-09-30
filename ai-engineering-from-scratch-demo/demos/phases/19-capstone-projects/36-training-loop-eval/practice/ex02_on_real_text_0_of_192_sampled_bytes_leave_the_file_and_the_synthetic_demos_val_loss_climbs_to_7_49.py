"""Exercise 2 -- on real text every sampled byte is in the file, and the synthetic demo's val loss climbs 5.55 to 7.49.

    Replace synthetic random tokens with bytes from a small text file so the demo trains on something legible. Verify the generated sample uses characters present in the file.

Reading of the exercise: the "small text file" is the Zen of Python (856
bytes, decoded from the stdlib `this` module), written to a temporary
`zen.txt` and read back as bytes, so the run needs no network and no
committed blob. The first 90% trains and the last 10% is held out. The
lesson's own `train()` runs unchanged with the demo's `TrainConfig`
(80 steps), from the prompt `b"Beautiful "`. "Uses characters present in the
file" is checked byte by byte on the demo's own sampler settings
(temperature 0.8, top_k 20) over three seeds, against an untrained model.

**ANSWER: 0 of 192 sampled bytes fall outside the file.** The file has 45
distinct bytes. After 80 steps, three 64-byte samples contain no byte the
file lacks. The same sampler on the untrained model draws 46 of 64 from
outside it. The trained sample is letter-and-space soup, not words: the
held-out loss (2.973 nats) sits only 0.136 below the file's own unigram
entropy (3.109), so the model has learned the character frequencies and
little more.

**FINDING: the check passes because of top_k, not only because of training.**
With top_k = 0 at temperature 1.0 the same trained model puts 47 of 2,000
bytes (2.4%) outside the file. The tail of 211 never-seen bytes still holds
some probability, and top_k = 20 cuts it off.

**FINDING: the lesson's synthetic demo cannot show the held-out loss dropping.**
`_synthetic_byte_tokens` promises "the eval loss should drop visibly", but
train and val use seeds 1 and 2, so they repeat two unrelated 32-token
patterns; their first 32 tokens agree at 1 of 32 positions. Val loss is
5.554 untrained (uniform is ln 256 = 5.545), 6.535 at step 19 and 7.492 at
step 79. Training makes the
held-out number worse by 1.94 nats, while train loss falls to 1.101.

**FINDING: on real text the probe does its job.** Train 400 steps instead
of 80 and train loss falls to 0.960, while val loss rises from 2.981 at step
99 to 3.455 at step 399. The model is memorising 770 bytes, and the
held-out split shows it.

Structure: `fit()` runs the lesson's `train` on a token split; `outside()`
counts sampled bytes the file lacks.
"""

from __future__ import annotations

import codecs
import collections
import contextlib
import io
import math
import pathlib
import tempfile

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "36-training-loop-eval"
PROMPT = b"Beautiful "


def make_fixture(folder):
    """Write the Zen of Python to `zen.txt` and return its path."""
    with contextlib.redirect_stdout(io.StringIO()):
        import this
    path = pathlib.Path(folder) / "zen.txt"
    path.write_text(codecs.decode(this.s, "rot13"), encoding="utf-8")
    return path


def fit(ref, train, val, folder, **overrides):
    cfg = ref.TrainConfig(**overrides)
    torch.manual_seed(0)
    model = ref.GPTModel(ref.ModelConfig(context_length=cfg.context_length, dropout=0.0))
    prompt = torch.tensor([list(PROMPT)])
    with contextlib.redirect_stdout(io.StringIO()):
        records = ref.train(model, train, val, cfg, prompt, log_path=pathlib.Path(folder) / "l.jsonl")
    vals = [round(r["val_loss"], 3) for r in records if "val_loss" in r]
    return model, vals, round(records[-1]["train_loss"], 3)


def outside(ref, model, allowed, n, seed, **sampler):
    with contextlib.redirect_stdout(io.StringIO()):
        seq = ref.generate_and_print_sample(model, torch.tensor([list(PROMPT)]), n, seed=seed, **sampler)
    new = seq[len(PROMPT) :]
    return sum(t not in allowed for t in new), bytes(new).decode("latin-1")


def synthetic(ref, folder):
    train, val = ref._synthetic_byte_tokens(4096, 256, 1), ref._synthetic_byte_tokens(1024, 256, 2)
    torch.manual_seed(0)
    start = ref.evaluate_model(ref.GPTModel(ref.ModelConfig(dropout=0.0)), ref.make_batches(val, 4, 32, 1), 4)
    _, vals, last_train = fit(ref, train, val, folder)
    shared = int((train[:32] == val[:32]).sum())
    return {"start": round(start, 3), "vals": vals, "train": last_train, "shared": shared}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as tmp:
        data = make_fixture(tmp).read_bytes()
        tokens = torch.tensor(list(data), dtype=torch.long)
        cut = int(0.9 * len(tokens))
        model, vals, _ = fit(ref, tokens[:cut], tokens[cut:], tmp)
        _, vals400, train400 = fit(ref, tokens[:cut], tokens[cut:], tmp, num_steps=400, eval_every=100)
        synth = synthetic(ref, tmp)
    allowed, counts, demo = set(data), collections.Counter(data), dict(temperature=0.8, top_k=20)
    torch.manual_seed(0)
    untrained = ref.GPTModel(ref.ModelConfig(dropout=0.0))
    samples = [outside(ref, model, allowed, 64, s, **demo) for s in range(3)]
    return {
        "size": len(data), "distinct": len(allowed), "train_bytes": cut,
        "outside": [c for c, _ in samples], "text": samples[0][1],
        "untrained": outside(ref, untrained, allowed, 64, 0, **demo)[0],
        "no_top_k": outside(ref, model, allowed, 2000, 0, temperature=1.0, top_k=0)[0],
        "val": vals[-1], "vals400": vals400, "train400": train400,
        "unigram": round(-sum(v / len(data) * math.log(v / len(data)) for v in counts.values()), 3),
        "synth": synth, "promise": "eval loss should drop visibly" in parity.lesson_dir(PHASE, LESSON)
        .joinpath("code/main.py").read_text(),
    }


def verify(result):
    r, s = result, result["synth"]
    return [
        practice.Check(
            "ANSWER: 0 of 192 sampled bytes fall outside the file",
            (r["size"], r["distinct"], r["outside"], r["untrained"], r["val"], r["unigram"])
            == (856, 45, [0, 0, 0], 46, 2.973, 3.109),
            f"{r['size']} bytes, {r['distinct']} distinct; outside-file per sample {r['outside']}, "
            f"untrained {r['untrained']}/64; val {r['val']} vs unigram {r['unigram']}; "
            f"sample {r['text'][:40]!r}",
        ),
        practice.Check(
            "FINDING: the check passes because of top_k, not only because of training",
            r["no_top_k"] == 47,
            f"top_k=0, T=1.0: {r['no_top_k']}/2000 bytes outside the file",
        ),
        practice.Check(
            "FINDING: the lesson's synthetic demo cannot show the held-out loss dropping",
            (r["promise"], s["shared"], s["start"], s["vals"], s["train"])
            == (True, 1, 5.554, [6.535, 7.609, 7.465, 7.492], 1.101),
            f"val {s['start']} untrained -> {s['vals']} at the probes; train {s['train']}; "
            f"first 32 tokens agree at {s['shared']}/32",
        ),
        practice.Check(
            "FINDING: on real text the probe does its job",
            (r["train400"], r["vals400"], r["train_bytes"])
            == (0.96, [2.981, 3.258, 3.398, 3.455], 770),
            f"400 steps: train {r['train400']}, val at probes {r['vals400']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
