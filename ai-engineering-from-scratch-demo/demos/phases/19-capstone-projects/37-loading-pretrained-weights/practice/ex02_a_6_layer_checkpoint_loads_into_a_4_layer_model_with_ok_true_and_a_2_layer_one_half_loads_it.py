"""Exercise 2 -- a 6-layer checkpoint loads into a 4-layer model with ok() True, and a 2-layer one half-loads it.

    Add an `expected_layers` argument that refuses to load a checkpoint whose `h.N` indices do not match the model's `num_layers`.

Reading of the exercise: `load_checked(ref, model, path, expected_layers)`
reads only the file's names, collects the `N` of every `h.N.` prefix, and
raises `ValueError` before the lesson's `load_safetensors` runs unless
`expected_layers == model.cfg.num_layers` and the indices are exactly
0..expected_layers-1. It is run against five checkpoints built with the
lesson's own `make_stub_safetensors` (vocab 256, d_model 192) into a 4-layer
model: 2, 4 and 6 layers, a 5-layer file with `h.2` removed (a gap), and the
right file with a wrong `expected_layers`. No real weights are downloaded.

**ANSWER: it refuses 4 of the 5 and leaves the model untouched.** Only the
4-layer file with `expected_layers=4` loads (`loaded=52`, report ok). The
other four raise before any assignment, and every parameter of the model is
bit-identical afterwards.

**FINDING: without the check, a deeper checkpoint "succeeds".** The 6-layer
file into the 4-layer model reports `ok() == True`: `h.4` and `h.5` (24
tensors) go to `unexpected`, which `ok()` ignores, so two trained layers are
dropped silently. The 5-layer file with a gap also lands 40 tensors before
anyone reads the report.

**FINDING: without the check, a shallower checkpoint half-loads the model.**
The loader refuses to assign only on a shape mismatch, not on missing names,
so the 2-layer file writes 28 tensors, lists the 24 of blocks 2-3 as missing,
and still changes 18 of the model's 52 parameters. The lesson's "Always
validate the file before any assignment" is not what `load_safetensors` does.

Structure: `layer_indices()` reads names only; `load_checked()` is the guard;
`attempt()` runs one case and records whether the model changed.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from harness import parity, practice

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch and safetensors: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "37-loading-pretrained-weights"


def layer_indices(path):
    with safe_open(str(path), framework="pt") as reader:
        return sorted({int(m.group(1)) for k in reader.keys() if (m := re.match(r"h\.(\d+)\.", k))})


def load_checked(ref, model, path, expected_layers):
    """The lesson's loader, refusing a checkpoint whose h.N indices do not match."""
    found, want = layer_indices(path), list(range(expected_layers))
    if expected_layers != model.cfg.num_layers or found != want:
        raise ValueError(
            f"checkpoint h.N indices {found} vs expected_layers={expected_layers}, "
            f"model num_layers={model.cfg.num_layers}"
        )
    return ref.load_safetensors(model, Path(path), verbose=False)


def attempt(ref, cfg, path, loader):
    torch.manual_seed(0)
    model = ref.GPTModel(cfg)
    before = {k: v.clone() for k, v in model.named_parameters()}
    try:
        report = loader(model, path)
        outcome = [report.summary(), report.ok()]
    except ValueError:
        outcome = ["refused", False]
    changed = sum(not torch.equal(before[k], v) for k, v in model.named_parameters())
    return outcome + [changed]


def fixtures(ref, tmp):
    def stub(n):
        path = Path(tmp) / f"h{n}.safetensors"
        ref.make_stub_safetensors(path, ref.ModelConfig(256, 64, 192, 6, n), seed=1)
        return path

    with safe_open(str(stub(5)), framework="pt") as reader:
        gap = {k: reader.get_tensor(k) for k in reader.keys() if not k.startswith("h.2.")}
    save_file(gap, str(Path(tmp) / "gap.safetensors"))
    return {"2": stub(2), "4": stub(4), "6": stub(6), "gap": Path(tmp) / "gap.safetensors"}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = ref.ModelConfig(vocab_size=256, context_length=64, d_model=192, num_heads=6, num_layers=4)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        files = fixtures(ref, tmp)
        for name, path in files.items():
            out[name] = {
                "guarded": attempt(ref, cfg, path, lambda m, p: load_checked(ref, m, p, 4)),
                "plain": attempt(ref, cfg, path, lambda m, p: ref.load_safetensors(m, p, verbose=False)),
            }
        out["4_wrong_arg"] = {"guarded": attempt(ref, cfg, files["4"], lambda m, p: load_checked(ref, m, p, 6))}
        out["gap_indices"] = layer_indices(files["gap"])
    return out


def verify(result):
    r = result
    refused = ["refused", False, 0]
    cases = ("2", "6", "gap", "4_wrong_arg")
    return [
        practice.Check(
            "ANSWER: it refuses 4 of the 5 and leaves the model untouched",
            r["4"]["guarded"][:2] == ["loaded=52 missing=0 unexpected=0 shape_mismatch=0", True]
            and all(r[c]["guarded"] == refused for c in cases),
            f"4 layers: {r['4']['guarded']}; " + "; ".join(f"{c}: {r[c]['guarded']}" for c in cases),
        ),
        practice.Check(
            "FINDING: without the check, a deeper checkpoint 'succeeds'",
            r["6"]["plain"][:2] == ["loaded=52 missing=0 unexpected=24 shape_mismatch=0", True]
            and r["gap"]["plain"][:2] == ["loaded=40 missing=12 unexpected=12 shape_mismatch=0", False]
            and r["gap_indices"] == [0, 1, 3, 4],
            f"6-layer file: {r['6']['plain']}; gap file {r['gap_indices']}: {r['gap']['plain']}",
        ),
        practice.Check(
            "FINDING: without the check, a shallower checkpoint half-loads the model",
            r["2"]["plain"] == ["loaded=28 missing=24 unexpected=0 shape_mismatch=0", False, 18],
            f"2-layer file: {r['2']['plain'][0]}, ok={r['2']['plain'][1]}, "
            f"{r['2']['plain'][2]} of 52 parameters changed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
