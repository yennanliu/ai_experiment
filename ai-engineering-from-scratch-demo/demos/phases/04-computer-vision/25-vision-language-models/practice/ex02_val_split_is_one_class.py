"""Exercise 2 — val split is one class.

    **(Medium)** Fine-tune Qwen2.5-VL-3B or LLaVA-1.6-7B with LoRA (rank 16) on
    500 images of a target domain with captions. Compare zero-shot vs fine-tuned
    MMBench-style accuracy.

Reading of the exercise: neither checkpoint is reachable and no weights may be
downloaded -- `transformers`, `llava`, `peft`, `accelerate`, `bitsandbytes` and
`datasets` all raise ModuleNotFoundError -- so "zero-shot against fine-tuned" is
run on the only trainable thing the lesson ships, `ToyVLM`, at 6,597 parameters.
That substitution is what makes the exercise answerable, and answering it turns
up a defect in the lesson's own recipe. `synthetic_vision_class_data` emits its
samples class by class, and `main()` splits them with the prefix slice
`X[:int(0.85*len(X))]`, so all 30 validation samples are class 4 and the training
set sees only 10 of that class. The `val_acc` the lesson prints is therefore
scored on a single-class set, where a constant "always 4" predictor is perfect --
and the untrained model's 0.000 on it is not a zero-shot score but an artefact of
which single class the random head happens to prefer. Accuracy saturates within a
handful of steps besides, so the arms are separated on steps-to-perfect instead.

Structure: `caught` is exercise 1's import probe, loaded with
`practice.load_module` rather than copied. `splits` returns the lesson's own
prefix split beside a stratified one of identical size. `lora` freezes a ToyVLM
and hangs a rank-16 bypass off each of the projector's two Linears through a
forward hook -- `out + (alpha/r) * up(down(x))`, with `up` zero-initialised so
the adapted model starts exactly where the frozen one did, and the adapters
parked on `model.adapters` so the optimiser can find them. `fit` runs the
lesson's own recipe -- Adam, lr 3e-3, batch 32, 150 steps -- over whatever is
still trainable, recording validation accuracy after every step; `first` is the
step at which a history reaches a level. `fit` is re-used by exercise 3. At 142 lines of code the file is over D14's 120-line target and
under its 150-line ceiling; the overrun is the adapter, which cannot be written
in fewer lines with no `peft` installed.
"""

from __future__ import annotations

import importlib
import pathlib

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "25-vision-language-models"

STACK = ("transformers", "llava", "peft", "accelerate", "bitsandbytes", "datasets")
CLASSES, PER_CLASS, HOLDOUT = 5, 40, 6
RANK, ALPHA, STEPS, BATCH, LR = 16, 32, 150, 32, 3e-3    # STEPS/BATCH/LR are main()'s own

accuracy = lambda model, x, y: float((model(x).argmax(-1) == y).float().mean())         # noqa: E731
first = lambda got, level: next((i for i, v in enumerate(got) if v >= level), -1)       # noqa: E731
listing = lambda got: ", ".join(f"{k} {v}" for k, v in got.items())                     # noqa: E731
absent = lambda got: all(v == "ModuleNotFoundError" for v in got.values())              # noqa: E731
stack = lambda: [(n, lambda n=n: importlib.import_module(n)) for n in STACK]            # noqa: E731
trainable = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)  # noqa: E731
ends = lambda runs: {k: v[-1] for k, v in runs.items()}                                 # noqa: E731
reaches = lambda runs: {k: first(v, 1.0) for k, v in runs.items()}                      # noqa: E731
sibling = practice.load_module(pathlib.Path(__file__).with_name("ex01_cmer_conf_gate_never_fires.py"))


def splits(torch, count) -> dict:
    index = torch.arange(count)
    ends_at = [(c * PER_CLASS, (c + 1) * PER_CLASS - HOLDOUT, (c + 1) * PER_CLASS) for c in range(CLASSES)]
    kept = torch.cat([index[a:b] for a, b, _ in ends_at])
    held = torch.cat([index[b:c] for _, b, c in ends_at])
    cut = int(0.85 * count)                     # main()'s own split, verbatim
    return {"lesson": (index[:cut], index[cut:]), "stratified": (kept, held)}


def lora(torch, model, rank=RANK, alpha=ALPHA):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.adapters = torch.nn.ModuleList()
    for layer in (0, 2):                        # net[1] is the GELU
        base = model.projector.net[layer]
        side = torch.nn.Sequential(torch.nn.Linear(base.in_features, rank, bias=False),
                                   torch.nn.Linear(rank, base.out_features, bias=False))
        torch.nn.init.zeros_(side[1].weight)
        base.register_forward_hook(lambda m, a, out, s=side: out + (alpha / rank) * s(a[0]))
        model.adapters.append(side)
    return model


def fit(torch, functional, model, train, val, count=STEPS):
    optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    gen, history = torch.Generator().manual_seed(0), []
    for _ in range(count):
        picked = torch.randperm(len(train[0]), generator=gen)[:BATCH]
        loss = functional.cross_entropy(model(train[0][picked]), train[1][picked])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        history.append(accuracy(model, *val))
    return history


def solve():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    tokens, labels = ref.synthetic_vision_class_data()
    cut, arms, runs, sizes = splits(torch, len(tokens)), {}, {}, {}
    for name, (train, val) in cut.items():
        pair = ((tokens[train], labels[train]), (tokens[val], labels[val]))
        torch.manual_seed(0)
        full = ref.ToyVLM()
        arms[f"{name}/zero-shot"] = accuracy(full, *pair[1])
        arms[f"{name}/majority"] = float((labels[val] == labels[val].mode().values).float().mean())
        runs[f"{name}/full"] = fit(torch, functional, full, *pair)
        torch.manual_seed(0)
        runs[f"{name}/lora"] = fit(torch, functional, lora(torch, ref.ToyVLM()), *pair)
        sizes[name] = torch.bincount(labels[val], minlength=CLASSES).tolist()
    torch.manual_seed(0)
    pooled = tokens.mean(1)
    centre = torch.stack([pooled[labels == c].mean(0) for c in range(CLASSES)])
    gaps, residual = torch.cdist(centre, centre), pooled - centre[labels]
    ratio = (centre - pooled.mean(0)).pow(2).sum(1).mean() / residual.pow(2).sum(1).mean()
    return {"missing": sibling.caught(stack(), ImportError), "arms": arms, "val_hist": sizes,
            "final": ends(runs), "reach": reaches(runs), "ratio": float(ratio), "gap": float(gaps[gaps > 0].min()),
            "guess": torch.bincount(ref.ToyVLM()(tokens[cut["lesson"][1]]).argmax(-1),
                                    minlength=CLASSES).tolist(),
            "noise": {"patch": float((tokens - pooled[:, None]).std()), "pooled": float(residual.std())},
            "size": {"toyvlm": sum(p.numel() for p in ref.ToyVLM().parameters()),
                     "projector": sum(p.numel() for p in ref.ToyVLM().projector.parameters()),
                     "adapter": trainable(lora(torch, ref.ToyVLM()))}}


def verify(result):
    arms, final, reach, size = result["arms"], result["final"], result["reach"], result["size"]
    hist, noise = result["val_hist"], result["noise"]
    return [
        practice.Check(
            "ANSWER: there is no VLM to fine-tune, so the arms run on the lesson's ToyVLM",
            absent(result["missing"]) and size["toyvlm"] == 6597,
            f"importing the fine-tuning stack gives {listing(result['missing'])}, and no weights may be "
            f"downloaded. What is left is ToyVLM: {size['toyvlm']:,} parameters, {size['projector']:,} of them "
            f"the projector. Qwen2.5-VL-3B is reported at 3e9 -- quoted, not measured -- so the stand-in is ~455,000x smaller"),
        practice.Check(
            "FINDING: the lesson's own validation set is one class, and it is class 4",
            hist["lesson"] == [0, 0, 0, 0, 30] and arms["lesson/majority"] == 1.0,
            f"`synthetic_vision_class_data` emits {CLASSES} classes x {PER_CLASS} in order and main() splits "
            f"at int(0.85*200)=170, so the val label histogram is {hist['lesson']} against {hist['stratified']} "
            f"for a stratified split of the same size. A constant 'always 4' predictor scores "
            f"{arms['lesson/majority']:.3f} there -- which is the lesson's own printed number"),
        practice.Check(
            "ANSWER: zero-shot reads 0.000 or 0.200 depending only on which split is used",
            arms["lesson/zero-shot"] == 0.0 and abs(arms["stratified/zero-shot"] - 0.2) < 0.11,
            f"the same untrained ToyVLM scores {arms['lesson/zero-shot']:.3f} on the lesson's val set and "
            f"{arms['stratified/zero-shot']:.3f} on the stratified one, because its random head answers "
            f"{result['guess']}: all 30 go to class 2, which the lesson's val set does not contain. "
            f"Fine-tuned reaches {final['lesson/full']:.3f} on either"),
        practice.Check(
            "MECHANISM: accuracy saturates in three steps, because the data is separable already",
            max(reach.values()) < STEPS // 4 and min(final.values()) == 1.0 and result["ratio"] > 500,
            f"every arm ends at 1.000 on both splits, so the exercise's comparison is a tie; first step "
            f"reaching 1.000 is {listing(reach)} of {STEPS}, and on the lesson split that only means "
            f"learning to answer 4. The pooled tokens carry a between-over-within variance ratio of "
            f"{result['ratio']:.1f}, closest centres {result['gap']:.3f} apart, because 0.1*randn measured "
            f"at {noise['patch']:.4f} per patch averages to {noise['pooled']:.4f} over 16 patches"),
        practice.Check(
            "CONTROL: LoRA rank 16 is not a parameter reduction at this width",
            size["adapter"] > 0.5 * size["projector"] and final["stratified/lora"] == 1.0,
            f"the rank-{RANK} adapter carries {size['adapter']:,} trainable parameters against the projector's "
            f"{size['projector']:,} -- {size['adapter'] / size['projector']:.0%}, because {RANK} is half of "
            f"min(32, 64); on a 3B model the same rank is reported to land well under 1%. With base and head "
            f"frozen it still reaches {final['stratified/lora']:.3f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
