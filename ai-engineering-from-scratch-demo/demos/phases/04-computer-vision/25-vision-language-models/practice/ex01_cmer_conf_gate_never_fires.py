"""Exercise 1 — cmer conf gate never fires.

    **(Easy)** Run three prompts ("what is this?", "count the objects", "describe
    the scene") through any open VLM on five images. Score each answer as correct
    / partially correct / hallucinated by hand. Compute a first-pass CMER-like
    rate.

Reading of the exercise: no open VLM is reachable -- `transformers`, `llava`,
`qwen_vl_utils`, `peft`, `timm` and `openai` all raise ModuleNotFoundError -- and
the lesson ships nothing that would consume a prompt if one were: handing the
exercise's own three prompt strings to each of the lesson's three entry points
raises, so the prompts have no way in. What the lesson does ship is
`cross_modal_error_rate`, so the exercise is read as a question about the
instrument rather than about a model: the 3x5 grid of answers is built with every
item's image-text cosine set by construction, hand-graded by that construction,
and pushed through the lesson's own CMER. Two things then fall out. CMER cannot
express "partially correct" -- it is a step function of the cosine with a single
break at `sim_threshold`, so a half-right answer scores 0 or 1 depending on which
side of 0.25 it lands. And the confidence half of the metric never fires, on this
grid or on the lesson's own demo fixture, because every confidence in both is
above `conf_threshold`; CMER there is the similarity term alone.

Structure: `caught` runs a list of labelled calls and records the exception each
one raises -- once over `stack` for the six missing packages, once over `prompts`
for the exercise's own three prompt strings handed to the lesson's own functions.
`aim` returns unit rows at an exact cosine to a batch of image embeddings, which
is what makes the hand grade a construction rather than an opinion; `grid`
assembles the five images, fifteen stray directions and the confidences; `graded`
fills in five correct, five partial and five hallucinated answers at a chosen
partial cosine; and `leak` measures the detector's miss rate on pairs that are
all hallucination, at three embedding widths. The module-level lambdas hold every
comprehension and every compound condition, which keeps `solve` and `verify`
inside D14's complexity cap. At 144 lines of code the file is over D14's 120-line
target and under its 150-line ceiling; the overrun is six checks carrying 41
measured numbers.
"""

from __future__ import annotations

import importlib
import math

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "25-vision-language-models"

DIM, IMAGES, PROMPTS = 32, 5, 3
ITEMS = IMAGES * PROMPTS
STACK = ("transformers", "llava", "qwen_vl_utils", "peft", "timm", "openai")
ASKED = ("what is this?", "count the objects", "describe the scene")
ENTRIES = ("cross_modal_error_rate", "deepstack_features", "ToyVLM.forward")
CONFIDENCES = (0.95, 0.90, 0.88, 0.85, 0.92)      # the lesson demo's own range, all > 0.8
GOOD, PARTIAL = 0.95, 0.45                        # cosines the graded answers are built at
SWEEP = (0.05, 0.15, 0.24, 0.26, 0.45, 0.95)
WIDTHS = (32, 128, 512)
SIM_GATE, CONF_GATE = 0.25, 0.8                   # cross_modal_error_rate's own defaults

listing = lambda got: ", ".join(f"{k} {v}" for k, v in got.items())                     # noqa: E731
row = lambda vals: " ".join(f"{v:.4f}" for v in vals)                                   # noqa: E731
sigmas = lambda: " ".join(f"{SIM_GATE * math.sqrt(d):.2f} sigma at d={d}" for d in WIDTHS)  # noqa: E731
sweeps = lambda got: " ".join(f"{c}->{v:.4f}" for c, v in zip(SWEEP, got))              # noqa: E731
leaks = lambda got: " ".join(f"d={d}: {v:.4f}" for d, v in zip(WIDTHS, got))            # noqa: E731
worst = lambda vals, target: max(abs(v - target) for v in vals)                         # noqa: E731
absent = lambda got: all(v == "ModuleNotFoundError" for v in got.values())              # noqa: E731
stepped = lambda got: got[0] == got[1] == got[2] > got[3] == got[4] == got[5]           # noqa: E731
inert = lambda got, cmer: got[0] == got[1] == got[2] == 0.0 and got[3] == got[4] == cmer  # noqa: E731
exact = lambda cos: max(worst(cos["good"], GOOD), worst(cos["partial"], PARTIAL))       # noqa: E731
stack = lambda: [(n, lambda n=n: importlib.import_module(n)) for n in STACK]             # noqa: E731
prompts = lambda ref, vlm: list(zip(ENTRIES, (                                           # noqa: E731
    lambda: ref.cross_modal_error_rate(ASKED[0], ASKED[0], ASKED[0]),
    lambda: ref.deepstack_features([ASKED[1]]), lambda: vlm(ASKED[2]))))


def caught(labelled_calls, expected) -> dict:
    got = {}
    for label, call in labelled_calls:
        try:
            call()
            got[label] = "accepted"                 # pragma: no cover - every one of these raises
        except expected as exc:
            got[label] = type(exc).__name__
    return got


def aim(functional, base, other, cosine):
    base = functional.normalize(base, dim=-1)
    perp = functional.normalize(other - (other * base).sum(-1, keepdim=True) * base, dim=-1)
    return cosine * base + math.sqrt(1.0 - cosine ** 2) * perp


def grid(torch, functional):
    gen = torch.Generator().manual_seed(0)
    unit = lambda count: functional.normalize(torch.randn(count, DIM, generator=gen), dim=-1)
    return unit(IMAGES).repeat_interleave(PROMPTS, 0), unit(ITEMS), torch.tensor(CONFIDENCES).repeat_interleave(PROMPTS)


def graded(functional, paired, stray, partial_cosine):
    text = stray.clone()
    text[0::PROMPTS] = aim(functional, paired[0::PROMPTS], stray[0::PROMPTS], GOOD)
    text[1::PROMPTS] = aim(functional, paired[1::PROMPTS], stray[1::PROMPTS], partial_cosine)
    return text


def leak(torch, functional, ref, width, count=4000):
    gen = torch.Generator().manual_seed(7)
    draw = lambda: functional.normalize(torch.randn(count, width, generator=gen), dim=-1)
    return ref.cross_modal_error_rate(draw(), draw(), torch.full((count,), 0.9))


def solve():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    paired, stray, confidence = grid(torch, functional)
    text = graded(functional, paired, stray, PARTIAL)
    cosine = (paired * text).sum(-1)
    torch.manual_seed(0)
    return {"missing": caught(stack(), ImportError),
            "refused": caught(prompts(ref, ref.ToyVLM()), (TypeError, AttributeError)),
            "cmer": ref.cross_modal_error_rate(paired, text, confidence),
            "cosine": {name: sorted(float(c) for c in cosine[i::PROMPTS])
                       for i, name in enumerate(("good", "partial", "wild"))},
            "sweep": [ref.cross_modal_error_rate(paired, graded(functional, paired, stray, c),
                                                 confidence) for c in SWEEP],
            "gate": [ref.cross_modal_error_rate(paired, text, torch.full((ITEMS,), c))
                     for c in (0.0, 0.5, CONF_GATE, CONF_GATE + 0.01, 1.0)],
            "leak": [leak(torch, functional, ref, w) for w in WIDTHS]}


def verify(result):
    missing, cosine, sweep, gate = (result[k] for k in ("missing", "cosine", "sweep", "gate"))
    cmer, refused, miss = result["cmer"], result["refused"], [1.0 - c for c in result["leak"]]
    return [
        practice.Check(
            "ANSWER: there is no open VLM to run, and nothing here would take a prompt",
            absent(missing) and "accepted" not in refused.values(),
            f"importing the stack gives {listing(missing)}. Handing the exercise's own three prompts to the "
            f"lesson's three entry points gives {listing(refused)}, so the {ITEMS} answers are graded by "
            f"construction instead -- which is what makes the hand score reproducible"),
        practice.Check(
            f"ANSWER: CMER reads {cmer:.4f} where the hand score says {1 / PROMPTS:.4f}",
            abs(cmer - 4 / ITEMS) < 1e-6 and max(cosine["wild"]) > SIM_GATE,
            f"{ITEMS} answers, {IMAGES} hallucinated by construction: hand rate {1 / PROMPTS:.4f} ({IMAGES}"
            f"/{ITEMS}) against CMER {cmer:.4f} ({round(cmer * ITEMS)}/{ITEMS}). The gap is one random caption "
            f"at cosine {max(cosine['wild']):.4f}, over the {SIM_GATE} gate; the rest sit at {row(cosine['wild'][:4])}"),
        practice.Check(
            "FINDING: 'partially correct' has no representation -- it is a step at 0.25",
            stepped(sweep) and abs(sweep[0] - 9 / ITEMS) < 1e-6 and abs(sweep[5] - cmer) < 1e-6,
            f"sweeping the five partial answers' cosine {sweeps(sweep)}: CMER is {sweep[0]:.4f} (9/{ITEMS}) "
            f"below the gate and {sweep[5]:.4f} (4/{ITEMS}) above it, nothing in between. A three-way hand "
            f"grade cannot survive a two-way metric; {SIM_GATE} exactly is a float knife-edge, so it is skipped"),
        practice.Check(
            "MECHANISM: the confidence half of CMER never fires on realistic confidences",
            inert(gate, cmer),
            f"forcing every confidence to 0.0/0.5/{CONF_GATE}/{CONF_GATE + 0.01}/1.0 gives CMER {row(gate)} -- "
            f"one step at `text_confidence > {CONF_GATE}` and nothing else. The demo's own confidences run "
            f"{min(CONFIDENCES)}-{max(CONFIDENCES)}, reused here, so CMER is the similarity term alone"),
        practice.Check(
            "FINDING: the 0.25 gate is a width-dependent constant, and 32 dimensions is too few",
            miss[0] > 20 * miss[1] and miss[2] < 1e-3,
            f"on pairs that are all hallucination -- 4,000 random unit pairs at confidence 0.9 -- CMER should "
            f"be 1.0000 and reads {leaks(result['leak'])}, missing {row(miss)}. Random unit cosines have "
            f"standard deviation 1/sqrt(d), so {SIM_GATE} is {sigmas()} -- at {DIM}-d it lets {miss[0]:.1%} through"),
        practice.Check(
            "CONTROL: the graded cosines are exact, so the hand score is not an opinion",
            exact(cosine) < 1e-5,
            f"the {IMAGES} correct answers sit at cosine {GOOD} and the {IMAGES} partial ones at {PARTIAL}, worst "
            f"deviation {exact(cosine):.2e} -- a structural identity, `cos*u + sqrt(1-cos^2)*v` with u and v "
            f"orthonormal, so 1e-5 is round-off, not a tolerance on a fitted result. Only the wild five are random"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
