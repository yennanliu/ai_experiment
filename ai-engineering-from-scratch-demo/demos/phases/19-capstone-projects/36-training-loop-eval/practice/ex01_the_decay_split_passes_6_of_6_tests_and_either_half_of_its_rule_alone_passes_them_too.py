"""Exercise 1 -- the split passes 6 of 6 tests, and either half of its rule alone passes them too.

    Add `weight_decay_groups()` unit tests that confirm scale and bias parameters land in the no decay group and linear and embedding weights land in the decay group.

Reading of the exercise: `main.py` has no `weight_decay_groups()`; the function
that builds the two AdamW groups is `build_param_groups(model, weight_decay)`,
so that is what the tests exercise. Six tests are written against it (every
LayerNorm scale undecayed, every shift and bias undecayed, every Linear weight
decayed, every Embedding weight decayed, the two groups partition
`model.parameters()`, and the groups carry `weight_decay` and 0.0). They run on
three models (the demo's tied model, an untied one, and one with no biases),
and then on four deliberately broken splits, so the tests show they can fail.

**ANSWER: all 6 tests pass on all 3 models.** Of the demo model's 28
parameter tensors, 10 matrices (116,736 values) are decayed and 18 vectors
(1,792 values) are not. The broken splits "decay everything", "decay nothing"
and "swap the groups" fail 2, 2 and 4 of the 6 tests on every model.

**FINDING: the rule's two halves are redundant.** `build_param_groups` sends
a tensor to no-decay when `dim() < 2` OR its name ends in `.bias`, `.shift`
or `.scale`. Keep only the dim test, or only the name test, and both produce
the same groups on all 3 models and pass all 6 tests. The lesson's own
`code/tests/test_training.py` never checks an embedding, so it could not
tell an embedding-excluding split from this one. That is a real
alternative: minGPT's `configure_optimizers` blacklists `torch.nn.Embedding`
from decay (github.com/karpathy/minGPT, mingpt/model.py, read 2026-09-29 at
https://raw.githubusercontent.com/karpathy/minGPT/master/mingpt/model.py).

**FINDING: the demo labels the tied count "untied".** `demo()` prints
`sum(p.numel() for p in model.parameters())` as "(untied count)", but
`parameters()` yields the shared `tok_embed`/`lm_head` tensor once, so it
prints 118,528. That is the tied count; untying gives 134,912, which is
256 x 64 = 16,384 more. In the tied model the shared matrix is decayed once,
under the name `tok_embed.weight`.

Structure: `kinds()` labels every parameter by the module that owns it;
`run_tests()` applies the six tests to one split function; `solve()` runs
them over models x splits.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON, WD = "19-capstone-projects", "36-training-loop-eval", 0.1


def kinds(ref, model):
    out = {}
    for module in model.modules():
        for pname, p in module.named_parameters(recurse=False):
            if isinstance(module, torch.nn.Linear) and pname == "weight":
                out.setdefault(id(p), "linear")
            elif isinstance(module, torch.nn.Embedding):
                out[id(p)] = "embedding"
            else:
                out.setdefault(id(p), pname)  # scale, shift, bias
    return out


def run_tests(ref, model, split):
    """The six unit tests against one split function; returns how many fail."""
    groups = split(model, WD)
    decay, no_decay = ({id(p) for p in g["params"]} for g in groups)
    kind = kinds(ref, model)
    of = lambda *ks: {i for i, k in kind.items() if k in ks}  # noqa: E731
    tests = {
        "scale_undecayed": of("scale") <= no_decay - decay,
        "shift_and_bias_undecayed": of("shift", "bias") <= no_decay - decay,
        "linear_decayed": of("linear") <= decay - no_decay,
        "embedding_decayed": of("embedding") <= decay - no_decay,
        "partition": not decay & no_decay and decay | no_decay == set(kind),
        "wd_values": (groups[0]["weight_decay"], groups[1]["weight_decay"]) == (WD, 0.0),
    }
    return sum(not ok for ok in tests.values())


def splits(ref):
    def two(model, wd, rule):
        named = list(model.named_parameters())
        return [{"params": [p for n, p in named if rule(n, p) == keep], "weight_decay": w}
                for keep, w in ((True, wd), (False, 0.0))]

    by_name = lambda n, p: not n.endswith((".bias", ".shift", ".scale"))  # noqa: E731
    return {
        "lesson": ref.build_param_groups,
        "dim_only": lambda m, wd: two(m, wd, lambda n, p: p.dim() >= 2),
        "name_only": lambda m, wd: two(m, wd, by_name),
        "decay_all": lambda m, wd: two(m, wd, lambda n, p: True),
        "decay_none": lambda m, wd: two(m, wd, lambda n, p: False),
        "swapped": lambda m, wd: two(m, wd, lambda n, p: p.dim() < 2),
    }


def same_groups(ref, models, split):
    ids = lambda gs: [sorted(id(p) for p in g["params"]) for g in gs]  # noqa: E731
    return all([ids(split(m, WD)) == ids(ref.build_param_groups(m, WD)) for m in models])


def sizes(ref, models):
    groups = ref.build_param_groups(models["tied"], WD)
    return {
        "tensors": [len(g["params"]) for g in groups],
        "values": [sum(p.numel() for p in g["params"]) for g in groups],
        "counts": [sum(p.numel() for p in models[k].parameters()) for k in ("tied", "untied")],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.manual_seed(0)
    cfgs = {"tied": {}, "untied": {"weight_tying": False}, "no_bias": {"use_bias": False}}
    models = {k: ref.GPTModel(ref.ModelConfig(dropout=0.0, **kw)) for k, kw in cfgs.items()}
    fns, code = splits(ref), parity.lesson_dir(PHASE, LESSON) / "code"
    return {
        **sizes(ref, models),
        "fails": {s: [run_tests(ref, m, f) for m in models.values()] for s, f in fns.items()},
        "same": [same_groups(ref, models.values(), fns[s]) for s in ("dim_only", "name_only")],
        "has_fn": hasattr(ref, "weight_decay_groups"),
        "tests_embed": "embed" in (code / "tests/test_training.py").read_text(),
        "label": "(untied count)" in (code / "main.py").read_text(),
    }


def verify(result):
    r, f, (tied, untied) = result, result["fails"], result["counts"]
    return [
        practice.Check(
            "ANSWER: all 6 tests pass on all 3 models",
            (f["lesson"], r["tensors"], r["values"]) == ([0] * 3, [10, 18], [116736, 1792])
            and (f["decay_all"], f["decay_none"], f["swapped"]) == ([2] * 3, [2] * 3, [4] * 3),
            f"failures per model {f['lesson']}; tensors/values decay vs not {r['tensors']} "
            f"{r['values']}; broken splits fail {f['decay_all']}, {f['decay_none']}, {f['swapped']}",
        ),
        practice.Check(
            "FINDING: the rule's two halves are redundant",
            (f["dim_only"], f["name_only"], r["same"], r["has_fn"], r["tests_embed"])
            == ([0] * 3, [0] * 3, [True, True], False, False),
            f"dim-only / name-only fail {f['dim_only']} / {f['name_only']}, same groups {r['same']};"
            f" weight_decay_groups exists {r['has_fn']}; tests mention embeddings {r['tests_embed']}",
        ),
        practice.Check(
            "FINDING: the demo labels the tied count 'untied'",
            (r["label"], tied, untied, untied - tied) == (True, 118528, 134912, 256 * 64),
            f"parameters() sums {tied:,} (tied), untied model {untied:,}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
