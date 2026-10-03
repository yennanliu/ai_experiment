"""Exercise 1 — the container prints 2.6.0+cu124, but only 3 of 15 packages are pinned.

    Build the Dockerfile and run `python -c "import torch; print(torch.__version__)"`
    inside the container

Reading of the exercise: there is no docker daemon in CI and a real build pulls
a ~4 GB base plus ~2.5 GB of wheels, so the build is replaced by reading the
lesson's own `code/Dockerfile` instruction by instruction and deriving what that
command must print. The question underneath is "which torch does this image
contain, and is that the same on every build?".

**ANSWER: `2.6.0+cu124`.** The image's only torch is
`torch==2.6.0+cu124` from the cu124 index, and `torch.__version__` carries the
`+cu124` local label. `python` is the deadsnakes 3.12 via `update-alternatives`.

**FINDING: 3 of the 15 pip packages are pinned; the other 12 resolve on build
day.** pip, setuptools, wheel and the 9 libraries (numpy, transformers, datasets,
accelerate, ...) carry no version, so two builds a month apart give different
images. The doc's "Your Dockerfile works on both" holds for one built image
shared as an artifact, not for the Dockerfile.

**FINDING: the lesson's prose names a different torch than its Dockerfile.**
`docs/en.md` says the image delivers "PyTorch 2.3" (5 mentions, including all
three boxes of the "Same image everywhere" diagram); the Dockerfile pins 2.6.0.

**CONTROL: the pins are mutually consistent.** The base `cuda:12.4.1`, the wheel
label `cu124` and the index URL `.../whl/cu124` agree; torchvision 0.21.0 and
torchaudio 2.6.0 are the releases paired with torch 2.6.0; get-pip is pinned to
a commit and a 64-hex sha256. The doc's ```dockerfile walk-through equals
`code/Dockerfile` exactly.

Structure: `instructions` splits the Dockerfile; `pip_packages` reads one RUN.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "07-docker-for-ai"
# torch release -> (torchvision, torchaudio) it was released with (pytorch.org/get-started)
PAIRS = {"2.6.0": ("0.21.0", "2.6.0"), "2.5.1": ("0.20.1", "2.5.1"), "2.4.1": ("0.19.1", "2.4.1")}


def instructions(text):
    """(INSTRUCTION, args) per Dockerfile instruction, continuation lines joined."""
    out = []
    for line in re.sub(r"\\\n", " ", text).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            op, _, args = line.partition(" ")
            out.append((op.upper(), " ".join(args.split())))
    return out


def pip_packages(args):
    """Requirement tokens of the first `pip install` in one RUN, flags dropped."""
    if "pip install" not in args:
        return []
    tokens = args.split("pip install", 1)[1].split(" && ")[0].split()
    flag_values = {tokens[i + 1] for i, t in enumerate(tokens[:-1]) if t == "--index-url"}
    return [t for t in tokens if not t.startswith("-") and t not in flag_values]


def solve():
    root = parity.lesson_dir(PHASE, LESSON)
    text = (root / "code" / "Dockerfile").read_text(encoding="utf-8")
    steps = instructions(text)
    runs = [args for op, args in steps if op == "RUN"]
    pkgs = [p for args in runs for p in pip_packages(args)]
    pins = dict(p.split("==") for p in pkgs if "==" in p)
    doc = parity.doc_text(PHASE, LESSON)
    block = re.search(r"```dockerfile\n(.*?)```", doc, re.S).group(1)
    return {
        "base": steps[0][1],
        "pins": pins,
        "unpinned": [p for p in pkgs if "==" not in p],
        "index": re.search(r"--index-url (\S+)", text).group(1),
        "sha": re.search(r'echo "([0-9a-f]+) ', text).group(1),
        "alternatives": "python python /usr/bin/python3.12" in text,
        "doc_23": doc.count("PyTorch 2.3"),
        "doc_equals_file": block.strip() == text.strip(),
    }


def pins_agree(result, torch_v, label, vision, audio):
    """CUDA tag, wheel label, index, companion versions, get-pip hash and doc block agree."""
    return all(
        (
            "cuda:12.4" in result["base"],
            label == "cu124",
            result["index"].endswith("/" + label),
            PAIRS[torch_v] == (vision, audio),
            len(result["sha"]) == 64,
            result["doc_equals_file"],
        )
    )


def verify(result):
    pins, loose = result["pins"], result["unpinned"]
    torch_v, _, label = pins["torch"].partition("+")
    vision, audio = (pins[k].split("+")[0] for k in ("torchvision", "torchaudio"))
    total = len(pins) + len(loose)
    return [
        practice.Check(
            "ANSWER: the container prints 2.6.0+cu124",
            pins["torch"] == "2.6.0+cu124" and result["alternatives"],
            f"the only torch is torch=={pins['torch']}, so torch.__version__ prints "
            f"'{pins['torch']}'; `python` is /usr/bin/python3.12 via update-alternatives",
        ),
        practice.Check(
            "FINDING: 3 of 15 pip packages are pinned; the rest resolve on build day",
            len(pins) == 3 and total == 15,
            f"pinned {sorted(pins)}; unpinned {len(loose)}: {', '.join(loose)}",
        ),
        practice.Check(
            "FINDING: the prose promises PyTorch 2.3, the Dockerfile pins 2.6.0",
            result["doc_23"] >= 4 and torch_v == "2.6.0",
            f"docs/en.md says 'PyTorch 2.3' {result['doc_23']} times, including the "
            f"'Same image everywhere' diagram; the Dockerfile pins {torch_v}",
        ),
        practice.Check(
            "CONTROL: CUDA tags, companion versions and get-pip hash agree",
            pins_agree(result, torch_v, label, vision, audio),
            f"base {result['base'].split()[-1]}, wheel +{label}, index {result['index']}; "
            f"torchvision {vision} / torchaudio {audio} pair with torch {torch_v}; get-pip "
            f"sha256 has {len(result['sha'])} hex chars; doc block == code/Dockerfile: "
            f"{result['doc_equals_file']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
