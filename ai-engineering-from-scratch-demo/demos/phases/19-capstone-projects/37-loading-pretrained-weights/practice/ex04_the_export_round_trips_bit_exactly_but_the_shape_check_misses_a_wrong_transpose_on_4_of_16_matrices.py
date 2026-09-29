"""Exercise 4 -- the export round trips bit-exactly, but the shape check misses a wrong transpose on 4 of 16 matrices.

    Add an export path: write the current model state into a fresh safetensors file using the pretrained naming convention. Round trip the loader and confirm the report has zero shape mismatches.

Reading of the exercise: `export(ref, model, path)` inverts the lesson's
`make_pretrained_to_local`, transposes back every name the lesson's
`_needs_transpose` flags, skips the tied `lm_head.weight` (the lesson says the
head "is not in the file"), and writes with `safetensors.torch.save_file`.
"Current model state" is a random-init replica model (seed 0, the demo config:
vocab 256, d_model 192, 4 layers). It is exported, loaded into a second model
(seed 1), and exported again from a model loaded with the lesson's stub. All
files go to a temp directory. No real weights are downloaded.

**ANSWER: zero shape mismatches, and the round trip is exact.** The export
writes 52 tensors. Loading them reports `loaded=52 missing=0 unexpected=0
shape_mismatch=0`, all 52 parameters of the second model are bit-identical to
the first, and its greedy sample matches. Exporting a stub-loaded model gives a
file byte-identical to the stub (7,369,512 bytes).

**FINDING: the two shortcuts both fail in safetensors.** `save_file(model.state_dict())`
raises, because the tied `lm_head.weight` and `tok_embed.weight` share memory.
Transposing with `.t()` and no `.contiguous()` also raises (non-contiguous
tensor).

**FINDING: the shape check catches a missing transpose on 12 of 16 matrices,
not all 16.** An export that forgets to transpose is refused with 12 shape
mismatches (`c_attn`, `c_fc`, `mlp.c_proj`, one per layer). But
`attn.c_proj.weight` is square (192 x 192), so a file with only those 4 stored
the wrong way round loads with `shape_mismatch=0`, and the model changes: the
greedy sample differs. When the report shows zero mismatches, that does not
mean the layout is right.

**FINDING: the Problem section's example name would load nothing.** The lesson
cites `transformer.h.0.attn.c_attn.weight` of shape `(2304, 768)`. The published
file has neither the prefix nor that shape. Its header (read 2026-09-29 from
https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors)
lists 160 F32 tensors, including `h.0.attn.c_attn.weight` [768, 2304] and one
`h.N.attn.bias` [1, 1, 1024, 1024] mask per layer. With the prefix added, the
stub loads `loaded=0 unexpected=52`. The loader raises nothing and only
`ok()` is False.

Structure: `export()` is the answer; `reload()` loads a file into a
fresh model and returns its report and greedy sample.
"""

from __future__ import annotations

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
PROMPT = [[7, 11, 13, 17]]


def export(ref, model, path, transpose=True):
    """Write the model under the pretrained names (tied lm_head left out)."""
    inverse = {local: src for src, local in ref.make_pretrained_to_local(model.cfg.num_layers).items()}
    tensors = {}
    for local, value in model.state_dict().items():
        if local in inverse:
            flip = transpose and ref._needs_transpose(inverse[local])
            tensors[inverse[local]] = (value.t() if flip else value).detach().contiguous()
    save_file(tensors, str(path))
    return tensors


def reload(ref, cfg, path, seed=1):
    torch.manual_seed(seed)
    model = ref.GPTModel(cfg)
    return model, ref.load_safetensors(model, Path(path), verbose=False), ref.quick_generate(model, torch.tensor(PROMPT), n=8)


def refused(call):
    try:
        call()
    except (RuntimeError, ValueError) as exc:
        return type(exc).__name__
    return "accepted"


def variants(path, square_path, prefix_path):
    """The stub with the 4 square attn.c_proj flipped, and with a 'transformer.' prefix."""
    with safe_open(str(path), framework="pt") as reader:
        stub = {k: reader.get_tensor(k) for k in reader.keys()}
    square = [k for k, v in stub.items() if k.endswith("c_proj.weight") and v.shape[0] == v.shape[1]]
    save_file({**stub, **{k: stub[k].t().contiguous() for k in square}}, str(square_path))
    save_file({f"transformer.{k}": v for k, v in stub.items()}, str(prefix_path))
    return len(square)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = ref.ModelConfig(vocab_size=256, context_length=64, d_model=192, num_heads=6, num_layers=4)
    with tempfile.TemporaryDirectory() as tmp:
        f = {k: Path(tmp) / f"{k}.safetensors" for k in ("stub", "a", "b", "c", "d", "e", "f")}
        torch.manual_seed(0)
        model = ref.GPTModel(cfg)
        written = export(ref, model, f["a"])
        copy, report, sample = reload(ref, cfg, f["a"])
        same = sum(torch.equal(p, q) for p, q in zip(model.parameters(), copy.parameters()))
        ref.make_stub_safetensors(f["stub"], cfg, seed=42)
        stub_model, _, stub_sample = reload(ref, cfg, f["stub"])
        export(ref, stub_model, f["b"])
        export(ref, stub_model, f["c"], transpose=False)
        n_square = variants(f["stub"], f["d"], f["e"])
        _, sq_report, sq_sample = reload(ref, cfg, f["d"])
        prefix = reload(ref, cfg, f["e"])[1]
        return {
            "written": len(written), "report": report.summary(), "same": same,
            "sample": sample == ref.quick_generate(model, torch.tensor(PROMPT), n=8),
            "bytes": (f["b"].read_bytes() == f["stub"].read_bytes(), f["stub"].stat().st_size),
            "naive": refused(lambda: save_file(model.state_dict(), str(f["f"]))),
            "noncontig": refused(lambda: save_file({"w": model.blocks[0].attn.qkv.weight.detach().t()}, str(f["f"]))),
            "no_transpose": reload(ref, cfg, f["c"])[1].summary(),
            "square": (sq_report.summary(), sq_sample != stub_sample, n_square),
            "prefix": (prefix.summary(), prefix.ok()),
            "doc": "`transformer.h.0.attn.c_attn.weight` of shape `(2304, 768)`" in parity.doc_text(PHASE, LESSON),
        }


def verify(result):
    r = result
    clean = "loaded=52 missing=0 unexpected=0 shape_mismatch=0"
    return [
        practice.Check(
            "ANSWER: zero shape mismatches, and the round trip is exact",
            all([r["written"] == 52, r["report"] == clean, r["same"] == 52,
                 r["sample"], r["bytes"] == (True, 7_369_512)]),
            f"{r['written']} tensors written; {r['report']}; {r['same']}/52 parameters "
            f"bit-identical; stub re-export byte-identical {r['bytes']}",
        ),
        practice.Check(
            "FINDING: the two shortcuts both fail in safetensors",
            (r["naive"], r["noncontig"]) == ("RuntimeError", "ValueError"),
            f"save_file(state_dict): {r['naive']} (shared memory); .t() without .contiguous(): {r['noncontig']}",
        ),
        practice.Check(
            "FINDING: the shape check catches a missing transpose on 12 of 16 matrices, not all 16",
            r["no_transpose"] == "loaded=0 missing=52 unexpected=0 shape_mismatch=12"
            and r["square"] == (clean, True, 4),
            f"export without transpose: {r['no_transpose']}; 4 square attn.c_proj stored flipped: "
            f"{r['square'][0]}, greedy sample changed {r['square'][1]}",
        ),
        practice.Check(
            "FINDING: the Problem section's example name would load nothing",
            r["doc"] and r["prefix"] == ("loaded=0 missing=52 unexpected=52 shape_mismatch=0", False),
            f"doc cites the prefixed (2304, 768) name: {r['doc']}; stub with 'transformer.' prefix: {r['prefix'][0]}, ok={r['prefix'][1]}, no exception",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
