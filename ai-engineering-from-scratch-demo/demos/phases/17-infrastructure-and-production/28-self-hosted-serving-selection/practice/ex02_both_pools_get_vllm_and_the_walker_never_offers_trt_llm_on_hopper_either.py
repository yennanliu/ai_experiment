"""Exercise 2 — both pools get vLLM, and the walker never offers TRT-LLM on Hopper either.

    Your infra is 12 H100s and 8 MI300X AMD. What engine? Why is TRT-LLM off
    the table?

Reading of the exercise: a mixed fleet cannot be one tensor-parallel group
across vendors, so it is two pools, and the engine is asked for per pool, as
`outputs/skill-engine-picker.md` requires ("require per-cluster engine
decisions"). The walker takes one hardware string, so it is called once per
pool for each of the lesson's five workloads at production scale.

**ANSWER: vLLM on both pools, one engine for the fleet.** H100 -> "NVIDIA
Hopper" and MI300X -> "AMD" give the same engine for all 5 workloads: vLLM for
chat, code and long context, SGLang for agentic and prefix-heavy work. Neither
card name is accepted as given: "H100", "MI300X" and "12 H100s and 8 MI300X"
all return `engine: None`. TRT-LLM is off the table because it is NVIDIA-only:
it could serve the 12 H100s, 60% of the GPUs, and only 960 of 2496 GB of HBM
-- the 8 MI300X hold 62% of the fleet's memory (80 GB per H100 SXM, 192 GB per
MI300X, vendor specs). Picking it means two engines, two configs and two
dashboards for a throughput edge the lesson places on Blackwell.

**FINDING: the walker never returns TRT-LLM on Hopper.** The lesson says
Hopper -> "vLLM or SGLang or TRT-LLM. All three top-tier." Across the 100-cell
grid TRT-LLM is returned on 20 cells, all Blackwell, and on 0 of 20 Hopper
cells. The sentence "TRT-LLM is NVIDIA-only" exists only on the AMD branch.

**FINDING: the Blackwell fallback cites the wrong Blackwell.** On B200/GB200
the walker names vLLM as "close second" because of "Blackwell SM120 (v0.15.1)".
SM120 is RTX Blackwell (compute capability 12.0), which is the lesson's own
wording ("RTX Blackwell SM120"). B200 is SM100.

Structure: `pools()` asks the walker once per pool per workload; the
coverage figures are counts of GPUs and GB, not throughput.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "28-self-hosted-serving-selection"
FLEET = {"NVIDIA Hopper": (12, 80), "AMD": (8, 192)}  # class: (GPUs, GB HBM each)
WORKLOADS = ("general chat", "agentic multi-turn", "RAG with heavy prefix reuse",
             "code generation", "long-context 128K")
HARDWARE = ("CPU", "Apple Silicon", "AMD", "NVIDIA Hopper", "NVIDIA Blackwell")
SCALES = ("single_user", "small_team", "production", "enterprise")
AS_WRITTEN = ("H100", "MI300X", "12 H100s and 8 MI300X")


def pools(ref):
    return {wl: {hw: ref.pick_engine(hw, "production", wl)["engine"] for hw in FLEET}
            for wl in WORKLOADS}


def trt_cells(ref):
    cells = itertools.product(HARDWARE, SCALES, WORKLOADS)
    return [c[0] for c in cells if ref.pick_engine(*c)["engine"] == "TRT-LLM"]


def said_on(ref, phrase):
    """Hardware classes on whose outputs some reason contains `phrase`."""
    cells = itertools.product(HARDWARE, SCALES, WORKLOADS)
    return sorted({c[0] for c in cells if any(phrase in r for r in ref.pick_engine(*c)["reasons"])})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gpus = sum(n for n, _ in FLEET.values())
    hbm = {hw: n * gb for hw, (n, gb) in FLEET.items()}
    blackwell = ref.pick_engine("NVIDIA Blackwell", "production", "general chat")["reasons"]
    return {
        "pools": pools(ref),
        "as_written": {s: ref.pick_engine(s, "production", "general chat")["engine"]
                       for s in AS_WRITTEN},
        "trt": trt_cells(ref), "gpu_share": FLEET["NVIDIA Hopper"][0] / gpus,
        "hbm": hbm, "amd_hbm_share": hbm["AMD"] / sum(hbm.values()),
        "nvidia_only_on": said_on(ref, "NVIDIA-only"),
        "sm120": [r for r in blackwell if "SM120" in r],
        "b200": any("B200" in r for r in blackwell),
        "doc_hopper_trt": "vLLM or SGLang or TRT-LLM" in parity.doc_text(PHASE, LESSON),
        "doc_rtx": "RTX Blackwell SM120" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    pools_ = result["pools"]
    same = [wl for wl, by_pool in pools_.items() if len(set(by_pool.values())) == 1]
    return [
        practice.Check(
            "ANSWER: vLLM on both pools, one engine for the fleet",
            all([same == list(WORKLOADS), pools_["general chat"]["AMD"] == "vLLM",
                 pools_["agentic multi-turn"]["AMD"] == "SGLang",
                 set(result["as_written"].values()) == {None},
                 result["gpu_share"] == 0.6, sum(result["hbm"].values()) == 2496,
                 round(result["amd_hbm_share"], 2) == 0.62]),
            f"per pool {pools_}; both pools agree on {len(same)} of 5 workloads; as written "
            f"{result['as_written']}; TRT-LLM could reach {result['gpu_share']:.0%} of GPUs and "
            f"{result['hbm']['NVIDIA Hopper']} of {sum(result['hbm'].values())} GB",
        ),
        practice.Check(
            "FINDING: the walker never returns TRT-LLM on Hopper",
            result["doc_hopper_trt"] and len(result["trt"]) == 20
            and set(result["trt"]) == {"NVIDIA Blackwell"}
            and result["nvidia_only_on"] == ["AMD"],
            f"TRT-LLM on {len(result['trt'])} of 100 cells, all {set(result['trt'])}; the doc "
            "lists it top-tier on Hopper; 'NVIDIA-only' is said only on "
            f"{result['nvidia_only_on']}",
        ),
        practice.Check(
            "FINDING: the Blackwell fallback cites the wrong Blackwell",
            len(result["sm120"]) == 1 and result["b200"] and result["doc_rtx"],
            f"on B200/GB200 the walker says {result['sm120']}; the lesson calls SM120 "
            "RTX Blackwell, and B200 is SM100",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
