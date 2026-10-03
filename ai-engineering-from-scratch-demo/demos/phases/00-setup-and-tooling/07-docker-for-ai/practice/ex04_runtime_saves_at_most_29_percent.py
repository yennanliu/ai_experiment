"""Exercise 4 — devel to runtime is a free swap, but saves at most 29%, not the base's 62%.

    Measure the image size with `docker images`. Try switching the base image from
    `devel` to `runtime` and compare sizes

Reading of the exercise: `docker images` cannot run in CI and its numbers would
depend on the host's pulls, so the sizes come from the lesson's own table in
"Step 3: Understand base images" (devel ~4 GB, runtime ~1.5 GB, the PyTorch
runtime image ~6 GB). The comparison asked for is the finished image, not the
bare base, so the base sizes are combined with the layers this Dockerfile adds
on top. The swap itself is applied to `code/Dockerfile` in memory and checked
for anything that needs `devel`.

**ANSWER: the swap is safe and saves the base's 2.5 GB.** Nothing in the
Dockerfile uses what `devel` adds: no `nvcc`, no flash-attn, no bitsandbytes,
no CUDA source build -- the doc's own reasons for `devel`. The torch wheels are
prebuilt.

**FINDING: on the finished image that is at most 29%, not 62%.** The doc's
table puts CUDA runtime plus PyTorch at ~6 GB, so the torch stack alone is
~4.5 GB. Both images carry it, so the saving is at most
2.5 / (4 + 4.5) = 29% -- and less again once the 9 libraries and Jupyter are
added. The bare-base ratio, 2.5 / 4 = 62%, is what `docker images` will not
show.

**FINDING: the one-token swap rebuilds every layer.** `FROM` is instruction 0,
so all 13 instructions re-run, torch download included: the comparison costs a
full second build.

**CONTROL: the runtime tag is the same release.** Swapping changes exactly one
instruction, and the new tag keeps CUDA 12.4.1 and ubuntu22.04, matching the
`cu124` wheels; it is listed in the doc's own table.

Structure: `sizes` reads the doc's table; the checks compare and bound.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "07-docker-for-ai"
NEEDS_DEVEL = ("nvcc", "flash-attn", "flash_attn", "bitsandbytes", "CUDA_HOME", "TORCH_CUDA_ARCH")


def sizes(doc):
    """Image name -> GB, from the doc's 'Size: ~N GB/MB' table."""
    section = doc.split("### Step 3", 1)[1].split("### Step 4", 1)[0]
    found = re.findall(r"(?m)^(\S+:\S+)\n(?:  .*\n)*?  Size: ~([\d.]+) (GB|MB)", section)
    return {name: float(n) / (1000 if unit == "MB" else 1) for name, n, unit in found}


def instructions(text):
    lines = (" ".join(ln.split()) for ln in re.sub(r"\\\n", " ", text).splitlines())
    return [ln for ln in lines if ln and not ln.startswith("#")]


def solve():
    root = parity.lesson_dir(PHASE, LESSON)
    text = (root / "code" / "Dockerfile").read_text(encoding="utf-8")
    steps = instructions(text)
    swapped = [steps[0].replace("-devel-", "-runtime-")] + steps[1:]
    table = sizes(parity.doc_text(PHASE, LESSON))
    devel, runtime = steps[0].split()[-1], swapped[0].split()[-1]
    torch_image = next(v for k, v in table.items() if k.startswith("pytorch/"))
    return {
        "devel": (devel, table[devel]),
        "runtime": (runtime, table.get(runtime)),
        "torch_stack": torch_image - table[runtime],
        "needs_devel": [w for w in NEEDS_DEVEL if w in text],
        "changed": sum(a != b for a, b in zip(steps, swapped)),
        "first_changed": next(i for i, (a, b) in enumerate(zip(steps, swapped)) if a != b),
        "steps": len(steps),
    }


def verify(result):
    (devel, dev_gb), (runtime, run_gb) = result["devel"], result["runtime"]
    stack = result["torch_stack"]
    saved = dev_gb - run_gb
    base_ratio, image_ratio = saved / dev_gb, saved / (dev_gb + stack)
    return [
        practice.Check(
            "ANSWER: the swap is safe and saves the base's 2.5 GB",
            not result["needs_devel"] and saved == 2.5,
            f"{devel} ~{dev_gb:g} GB -> {runtime} ~{run_gb:g} GB per the doc's table; the "
            f"Dockerfile mentions none of {', '.join(NEEDS_DEVEL)}",
        ),
        practice.Check(
            "FINDING: on the finished image the saving is at most 29%, not 62%",
            abs(base_ratio - 0.625) < 1e-9 and 0.28 < image_ratio < 0.30,
            f"the doc's PyTorch runtime image minus CUDA runtime leaves a ~{stack:g} GB torch "
            f"stack in both images, so the saving is at most {saved:g} / {dev_gb + stack:g} = "
            f"{image_ratio:.1%}, against {base_ratio:.1%} on the bare base",
        ),
        practice.Check(
            "FINDING: the one-token swap re-runs all 13 layers",
            result["first_changed"] == 0 and result["steps"] == 13,
            f"FROM is instruction {result['first_changed']}, so all {result['steps']} "
            "instructions re-run, torch download included",
        ),
        practice.Check(
            "CONTROL: the runtime tag is the same CUDA 12.4.1 / ubuntu22.04 release",
            result["changed"] == 1 and runtime == devel.replace("-devel-", "-runtime-")
            and run_gb is not None and "12.4.1" in runtime,
            f"{result['changed']} instruction changes; {runtime} is in the doc's table",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
