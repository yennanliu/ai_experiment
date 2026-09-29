"""Exercise 4 -- resuming model and optimizer state is bit-exact only if the batch stream is fast-forwarded too.

    Save a checkpoint every `eval_every` steps in addition to the JSONL log. Add a `resume_from` flag that reloads model state and optimizer state.

Reading of the exercise: the lesson's `train()` cannot be extended in place
(it builds the optimizer internally and deletes the log on entry), so
`train_ckpt()` is the same loop rebuilt from the lesson's own pieces
(`make_batches`, `calc_loss_batch`, `evaluate_model`, `build_param_groups`,
`cosine_with_warmup`). It saves `{model, optimizer, step}` at every probe
step and takes `resume_from=<path>`. It is first checked against the
lesson's `train()`; then a run is "crashed" after step 45 and resumed from
the step-39 checkpoint into a freshly initialised model, on the demo config.

**ANSWER: the resumed run matches the uninterrupted one exactly.** Without
checkpointing, `train_ckpt` reproduces the lesson's `train()` on all 80
records (worst difference 0.0). Resumed from `ckpt_39.pt`, steps 40-79
match the straight run bit for bit: train loss, LR and all probe val losses.
It saves 4 checkpoints (steps 19, 39, 59, 79), each about 3.0x the
parameter bytes (weights plus Adam's two moments). The tied
`lm_head`/`tok_embed` matrix is stored once.

**FINDING: model plus optimizer state is not enough.** "Reloads model state
and optimizer state" is what the exercise asks for, and it does not resume
the run. `make_batches` is a seeded generator, so a restart replays batches
0, 1, 2 ... from step 40 onward unless the stream is fast-forwarded by the
resumed step count. Without that, every resumed step sees a different batch.
Reloading the model but not the optimizer diverges too: Adam restarts with
zero moments and step count 0. Only step 40 matches (its loss is taken
before the first update), and the train loss is off by up to 0.123 over
steps 41-79. The un-fast-forwarded run matches on none of the 40 steps and
is off by up to 0.698.

**FINDING: the JSONL log cannot say where to resume.** The doc says you
"can resume training by reading the last step" of the JSONL. After the crash
the log's last step is 45, but the newest checkpoint is step 39, so the
appended log carries steps 40-45 twice (86 lines for 80 steps). The lesson's
`train()` would do worse: it unlinks `log_path` on entry, so calling it to
resume erases the earlier log (1 line left after a 1-step call).

Structure: `train_ckpt()` is the loop on a freshly seeded model;
`restore()` loads a checkpoint; `lesson_runs()` runs the lesson's `train()`
for parity; `solve()` runs straight, crash-and-resume and the two naive
resumes.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "36-training-loop-eval"


def restore(model, opt, loader, path, mode):
    ck = torch.load(path)  # mode: 'full', 'no_skip' (stream restarted) or 'model_only'
    model.load_state_dict(ck["model"])
    if mode != "model_only":
        opt.load_state_dict(ck["optim"])
    for _ in range(ck["step"] + 1 if mode != "no_skip" else 0):
        next(loader)  # fast-forward the seeded batch stream
    return ck["step"] + 1


def train_ckpt(ref, data, folder, init_seed=0, resume_from=None, stop_after=None, mode="full"):
    """The lesson's loop on a fresh model, plus a checkpoint every probe step and resume_from."""
    torch.manual_seed(init_seed)
    model, cfg = ref.GPTModel(ref.ModelConfig(dropout=0.0)), ref.TrainConfig()
    opt = torch.optim.AdamW(ref.build_param_groups(model, cfg.weight_decay), lr=cfg.max_lr,
                            betas=(0.9, 0.95))
    loader = ref.make_batches(data[0], cfg.batch_size, cfg.context_length, cfg.seed)
    start = restore(model, opt, loader, resume_from, mode) if resume_from else 0
    records, stop = [], cfg.num_steps if stop_after is None else stop_after + 1
    for step in range(start, stop):
        lr = ref.cosine_with_warmup(step, cfg.warmup_steps, cfg.num_steps, cfg.max_lr, cfg.min_lr)
        opt.param_groups[0]["lr"] = opt.param_groups[1]["lr"] = lr
        inputs, targets = next(loader)
        opt.zero_grad(set_to_none=True)
        loss = ref.calc_loss_batch(model, inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        opt.step()
        rec = {"step": step, "train_loss": float(loss.item()), "lr": lr}
        if (step + 1) % cfg.eval_every == 0 or step == cfg.num_steps - 1:
            val = ref.make_batches(data[1], cfg.batch_size, cfg.context_length, cfg.seed + 1)
            rec["val_loss"] = ref.evaluate_model(model, val, cfg.eval_batches)
            torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "step": step},
                       folder / f"ckpt_{step}.pt")
        records.append(rec)
        with (folder / "losses.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    return records, model


def lesson_runs(ref, data, folder, straight):
    torch.manual_seed(0)
    fresh, log = ref.GPTModel(ref.ModelConfig(dropout=0.0)), folder / "lesson.jsonl"
    with contextlib.redirect_stdout(io.StringIO()):
        full = ref.train(fresh, *data, ref.TrainConfig(), torch.tensor([[7, 11, 13, 17]]), log)
        ref.train(fresh, *data, ref.TrainConfig(num_steps=1), torch.tensor([[1]]), log)
    worst = max(abs(x[k] - y[k]) for x, y in zip(straight, full) for k in x)
    return worst, len(log.read_text().splitlines())


def diff(run, straight):
    return (max(abs(x["train_loss"] - y["train_loss"]) for x, y in zip(run, straight[40:])),
            sum(x == y for x, y in zip(run, straight[40:])))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = (ref._synthetic_byte_tokens(4096, 256, 1), ref._synthetic_byte_tokens(1024, 256, 2))
    with tempfile.TemporaryDirectory() as tmp:
        a, b, c = (pathlib.Path(tempfile.mkdtemp(dir=tmp)) for _ in "abc")
        straight, model = train_ckpt(ref, data, a)
        worst, lesson_log = lesson_runs(ref, data, c, straight)
        train_ckpt(ref, data, b, stop_after=45)
        runs = {m: train_ckpt(ref, data, {"full": b}.get(m, c), 7, b / "ckpt_39.pt", mode=m)[0]
                for m in ("full", "no_skip", "model_only")}
        steps = [json.loads(x)["step"] for x in (b / "losses.jsonl").read_text().splitlines()]
        return {
            "parity": worst, "lesson_log": lesson_log, "resumed": runs["full"] == straight[40:],
            "ckpts": sorted(int(p.stem[5:]) for p in a.glob("ckpt_*.pt")),
            "ratio": (a / "ckpt_79.pt").stat().st_size / (4 * sum(p.numel() for p in model.parameters())),
            "no_skip": diff(runs["no_skip"], straight), "no_opt": diff(runs["model_only"], straight),
            "log_lines": len(steps), "dupes": sorted({s for s in steps if steps.count(s) > 1}),
            "doc": "resume training by reading the last step" in parity.doc_text(PHASE, LESSON),
        }


def verify(result):
    r, ((sw, ss), (ow, os_)) = result, (result["no_skip"], result["no_opt"])
    return [
        practice.Check("ANSWER: the resumed run matches the uninterrupted one exactly",
                       (r["parity"], r["resumed"], r["ckpts"]) == (0.0, True, [19, 39, 59, 79])
                       and 2.9 < r["ratio"] < 3.1,
                       f"vs train() {r['parity']}; resumed 40-79 equal {r['resumed']}; "
                       f"checkpoints {r['ckpts']}, {r['ratio']:.2f}x parameter bytes"),
        practice.Check("FINDING: model plus optimizer state is not enough",
                       (ss, os_) == (0, 1) and min(sw, ow) > 1e-2,
                       f"no fast-forward {sw:.3f} ({ss}/40 equal), model-only {ow:.3f} ({os_}/40)"),
        practice.Check("FINDING: the JSONL log cannot say where to resume",
                       (r["doc"], r["log_lines"], r["dupes"], r["lesson_log"])
                       == (True, 86, list(range(40, 46)), 1),
                       f"{r['log_lines']} lines, twice {r['dupes']}; train() re-call leaves "
                       f"{r['lesson_log']} line"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
