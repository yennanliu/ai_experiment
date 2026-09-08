"""Exercise 2 — dino centring prevents collapse.

    **(Medium)** Implement a DINO-style centre buffer. Show that without the centring, the student collapses to a constant vector within a few epochs.

Reading of the exercise: the centre buffer is already implemented -- the
lesson's `DinoHead` registers it and updates it with momentum 0.9 -- so the work
is the demonstration, and the demonstration needs a collapse metric that is not
itself saturated. The obvious one, "the max column mean of the teacher output",
turns out to be useless: without centring it reads 0.99 in one seed and 0.51 in
another after 40 epochs, both fully collapsed, because collapse picks an
arbitrary output direction rather than always the same one. What discriminates
is the effective rank of the student-output covariance, `(sum e)^2 / sum e^2`
over its eigenvalues: without centring it lands at 1.00-1.01 in all four seeds
and with centring at 1.96-3.16 in all four, with no overlap. "Within a few
epochs" is the part that does not survive reseeding -- three seeds cross the
collapse line at epoch 1, 3 and 3, but the fourth takes 25, so the file reports
the spread instead of the phrase. Two things about this specific toy are worth
more than the DINO story in general. `DinoHead` has no EMA teacher at all:
student and teacher share the one `nn.Linear`, so the only asymmetry is the
temperature gap plus the centre, and raising the teacher temperature from its
default 0.04 to 0.2 stops the collapse with no centring whatsoever (effective
rank 10.27-10.85, all 16 output modes used). Collapse here is a property of the
temperature ratio, and the centre is what makes a sharp teacher safe to use.
And `update_centre`'s parameter is named `teacher_out`, i.e. probabilities,
while `main.py` calls it with `head.proj(feats)`, i.e. logits; both readings
prevent collapse, because `teacher()` divides by 0.04 and so multiplies any
centre by 25x. The encoder and the two-cluster inputs here are this file's own
experimental scaffolding; the head, its centring and its temperatures are the
lesson's. Nothing is downloaded.

Structure: `effective_rank` reduces a batch of student probabilities to one
scalar via the eigenvalues of their covariance; `probe` reads effective rank,
peak column mean and the number of distinct argmax modes off a trained head;
`train` runs one self-distillation configuration -- centring mode and teacher
temperature -- over EPOCHS and returns the per-epoch effective ranks; `variants`
runs every configuration across SEEDS and reduces each to its final rank range
and the first epoch that crossed the collapse line.

At 146 code lines this sits above D14's 120-line target and 4 clear of the
ceiling: five checks over 5 configurations x 4 seeds x 40 epochs of training,
which runs in about two seconds.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "17-self-supervised-vision"

IN_DIM, OUT_DIM, LATENT = 64, 16, 8
SAMPLES, BATCH, EPOCHS = 512, 128, 40
SEEDS = (0, 1, 2, 3)
COLLAPSED = 1.05
CONFIGS = {"none": (None, 0.04), "logits": ("logits", 0.04), "probs": ("probs", 0.04),
           "warm": (None, 0.1), "flat": (None, 0.2)}

span = lambda values: f"{min(values):.2f}-{max(values):.2f}"           # noqa: E731 - a formatter


def effective_rank(torch, probs) -> float:
    eig = torch.linalg.eigvalsh(torch.cov(probs.T)).clamp(min=0)
    return (eig.sum() ** 2 / (eig ** 2).sum()).item()


def probe(torch, head, feats) -> dict:
    with torch.no_grad():
        probs = head.student(feats).exp()
    return {"rank": effective_rank(torch, probs), "peak": probs.mean(0).max().item(),
            "modes": int((probs.argmax(-1).bincount(minlength=OUT_DIM) > 0).sum()),
            "centre": head.centre.abs().max().item()}


def train(torch, ref, centring, teacher_temp, seed) -> tuple:
    torch.manual_seed(seed)
    encoder = torch.nn.Sequential(torch.nn.Linear(LATENT, IN_DIM), torch.nn.GELU(),
                                  torch.nn.Linear(IN_DIM, IN_DIM))
    head = ref.DinoHead(in_dim=IN_DIM, out_dim=OUT_DIM)
    optimiser = torch.optim.Adam([*encoder.parameters(), *head.parameters()], lr=3e-3)
    torch.manual_seed(100 + seed)
    data = torch.randn(SAMPLES, LATENT)
    ranks = []
    for _ in range(EPOCHS):
        for start in range(0, SAMPLES, BATCH):
            chunk = data[start:start + BATCH]
            first = encoder(chunk + 0.1 * torch.randn_like(chunk))
            second = encoder(chunk + 0.1 * torch.randn_like(chunk))
            target = head.teacher(second, temp=teacher_temp)
            loss = -(target * head.student(first)).sum(-1).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if centring == "logits":                 # what `main.py` actually passes
                head.update_centre(head.proj(second.detach()))
            elif centring == "probs":                # what the parameter name says
                head.update_centre(target)
        with torch.no_grad():
            ranks.append(effective_rank(torch, head.student(encoder(data)).exp()))
    return ranks, probe(torch, head, encoder(data).detach())


def variants(torch, ref) -> dict:
    table = {}
    for name, (centring, teacher_temp) in CONFIGS.items():
        finals, firsts, modes, centres, peaks = [], [], [], [], []
        for seed in SEEDS:
            ranks, final = train(torch, ref, centring, teacher_temp, seed)
            finals.append(ranks[-1])
            firsts.append(next((i + 1 for i, r in enumerate(ranks) if r < COLLAPSED), None))
            modes.append(final["modes"])
            centres.append(final["centre"])
            peaks.append(final["peak"])
        table[name] = {"finals": finals, "firsts": firsts, "modes": modes, "centres": centres,
                       "peaks": peaks,
                       "collapsed": sum(rank < COLLAPSED for rank in finals)}
    return table


def solve():
    try:
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    head = ref.DinoHead(in_dim=IN_DIM, out_dim=OUT_DIM)
    return {"table": variants(torch, ref), "momentum": head.momentum, "fresh": head.centre.abs().max().item(),
            "grad": head.centre.requires_grad, "buffered": "centre" in head.state_dict(),
            "params": sum(p.numel() for p in head.parameters())}


def verify(result):
    table = result["table"]
    none, logits, probs, warm, flat = (table[k] for k in ("none", "logits", "probs", "warm", "flat"))
    return [
        practice.Check(
            "ANSWER: without centring the student collapses in all 4 seeds — effective rank falls to 1.0",
            none["collapsed"] == len(SEEDS) and max(none["finals"]) < COLLAPSED,
            f"self-distilling the lesson's own `DinoHead` for {EPOCHS} epochs with `update_centre` never "
            f"called leaves the effective rank of the {OUT_DIM}-d student output at "
            f"{span(none['finals'])} across seeds {SEEDS}, i.e. a single direction, using only "
            f"{min(none['modes'])}-{max(none['modes'])} of {OUT_DIM} argmax modes. The centre stays exactly "
            f"{result['fresh']:.1f}, so `teacher()` degenerates to plain softmax sharpening"),
        practice.Check(
            "FINDING: 'within a few epochs' does not survive reseeding — the spread is 1 to 25 epochs",
            sorted(none["firsts"])[0] <= 3 and max(none["firsts"]) > 10,
            f"the first epoch whose effective rank drops below {COLLAPSED}, per seed: {none['firsts']} -- "
            f"three seeds collapse by epoch {sorted(none['firsts'])[2]} but the slowest needs "
            f"{max(none['firsts'])}, so \"a few epochs\" is right for the median run and wrong by an order of "
            f"magnitude for the tail. All four are collapsed by epoch {EPOCHS}. The peak column mean, the "
            f"metric one would reach for first, is useless here: across the same four collapsed runs it "
            f"spans {span(none['peaks'])}, so no single threshold on it separates collapse from health"),
        practice.Check(
            "ANSWER: the centre buffer prevents it — rank 1.96-3.16 against 1.00-1.01, no overlap",
            min(logits["finals"]) > max(none["finals"]) + 0.9 and logits["collapsed"] == 0,
            f"calling `update_centre(head.proj(...))` every step, exactly as `main.py` does, holds the final "
            f"effective rank at {span(logits['finals'])} in {len(SEEDS)}/{len(SEEDS)} seeds against "
            f"{span(none['finals'])} without it -- a gap of "
            f"{min(logits['finals']) - max(none['finals']):.2f} with no seed overlap -- and keeps "
            f"{min(logits['modes'])}-{max(logits['modes'])} of {OUT_DIM} modes alive. The buffer grows to "
            f"|centre|max {span(logits['centres'])}; at momentum {result['momentum']} it needs a few dozen "
            "steps to build up, which is why one seed dips toward rank 1 early and recovers"),
        practice.Check(
            "MECHANISM: collapse is a temperature-ratio effect — a teacher at 0.2 needs no centring at all",
            flat["collapsed"] == 0 and max(warm["finals"]) < 3.0 < min(flat["finals"]),
            f"`DinoHead.teacher` defaults to temp=0.04 against the student's 0.1. With centring off and the "
            f"teacher raised to 0.2, the final rank is {span(flat['finals'])} and all "
            f"{min(flat['modes'])}/{OUT_DIM} modes stay in use -- no collapse, no buffer. At teacher temp 0.1, "
            f"equal to the student's, the rank is {span(warm['finals'])} -- degenerate in every seed and "
            f"below {COLLAPSED} in {warm['collapsed']}/{len(SEEDS)} of them, so 0.1 is already sharp "
            f"enough to be unsafe. So sharpening is what drives the collapse the exercise asks about, "
            "and the centre is what makes a sharp teacher usable rather than a cure for distillation itself"),
        practice.Check(
            "CONTROL: `centre` is a buffer, and the naming mismatch in `update_centre` is latent, not fatal",
            not result["grad"] and result["buffered"] and probs["collapsed"] == 0,
            f"`register_buffer` puts `centre` in `state_dict()` ({result['buffered']}) with requires_grad "
            f"{result['grad']}, so all {result['params']} trainable parameters in the head are `proj`'s and "
            f"the centre moves only through its momentum-{result['momentum']} EMA. `update_centre`'s parameter "
            f"is named `teacher_out` (probabilities) but `main.py` passes logits; feeding it probabilities "
            f"instead also prevents collapse ({span(probs['finals'])}, {probs['collapsed']}/{len(SEEDS)} "
            f"collapsed) because dividing by 0.04 scales any centre by 25x -- the smaller buffer "
            f"({span(probs['centres'])} against {span(logits['centres'])}) is still large enough"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
