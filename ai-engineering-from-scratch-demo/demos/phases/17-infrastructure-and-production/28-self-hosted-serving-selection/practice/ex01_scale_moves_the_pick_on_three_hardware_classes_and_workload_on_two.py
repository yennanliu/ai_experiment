"""Exercise 1 — scale moves the pick on three hardware classes and workload on two.

    Run `code/main.py` with your hardware / scale / workload. Does the output
    match your intuition?

Reading of the exercise: "your" inputs are one real setup -- a single RTX
4090 coding box -- plus the whole grid the lesson's own vocabulary spans: its
5 hardware classes, 4 scales and the 5 workloads "Workload-third decision"
names, 100 calls to `pick_engine`. Intuition is taken to be the lesson's own
rule, "hardware first, scale second, workload third", so a mismatch is a cell
where the code disagrees with the lesson.

**ANSWER: on the lesson's 7 scenarios it matches; off them it does not.**
The shipped scenarios read as intended. Over the grid, scale changes the pick
on 3 of 5 hardware classes (CPU, Apple Silicon, Hopper) and workload on 2 (AMD,
Hopper). Only 2 of the 5 named workloads can change anything -- "code
generation" and "long-context 128K" never do. My own box, "NVIDIA RTX 4090",
gets `engine: None`, and so do "H100", "cpu" and "MI300X": only exact-cased
class names match, and an unmatched one fails silently with the TGI line as the
whole explanation.

**FINDING: the tree does not run hardware, then scale, then workload.** On
Hopper a single user doing agentic work gets SGLang: workload overrides scale.
On AMD a single user gets vLLM, on Blackwell TRT-LLM -- scale is never read,
though the lesson sends 1 user to Ollama. Blackwell returns TRT-LLM on all 20
cells, agentic included, while the lesson says RadixAttention "dominates"
agentic.

**FINDING: the enterprise line stacks vLLM's production-stack onto engines
that are not vLLM.** It is appended to all 25 enterprise cells; 19 of them
chose llama.cpp, SGLang or TRT-LLM. And TGI's maintenance line is on 100 of
100 outputs while TGI is the engine on 0.

Structure: `grid()` calls `pick_engine` on every combination; the checks
count cells by the column that changes.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "28-self-hosted-serving-selection"
HARDWARE = ("CPU", "Apple Silicon", "AMD", "NVIDIA Hopper", "NVIDIA Blackwell")
SCALES = ("single_user", "small_team", "production", "enterprise")
WORKLOADS = ("general chat", "agentic multi-turn", "RAG with heavy prefix reuse",
             "code generation", "long-context 128K")
MINE = ("NVIDIA RTX 4090", "single_user", "coding assistant")
OFF_MENU = ("H100", "cpu", "MI300X")


def grid(ref):
    return {cell: ref.pick_engine(*cell) for cell in itertools.product(HARDWARE, SCALES, WORKLOADS)}


def moves(cells, axis):
    """Hardware classes on which varying `axis` (1 = scale, 2 = workload) changes the engine."""
    varied = set()
    for (hw, sc, wl), out in cells.items():
        for other in (SCALES if axis == 1 else WORKLOADS):
            key = (hw, other, wl) if axis == 1 else (hw, sc, other)
            if cells[key]["engine"] != out["engine"]:
                varied.add(hw)
    return sorted(varied, key=HARDWARE.index)


def tally(cells):
    """Counts over the grid: live workloads, stacking and the TGI line."""
    enterprise = [c for c in cells if c[1] == "enterprise"]
    return {
        "live_workloads": sorted({wl for (hw, sc, wl), out in cells.items()
                                  if out["engine"] != cells[(hw, sc, "general chat")]["engine"]}),
        "stacked_off_vllm": sum(cells[c]["engine"] != "vLLM" for c in enterprise),
        "enterprise": len(enterprise),
        "tgi_line": sum(any("TGI" in r for r in out["reasons"]) for out in cells.values()),
        "tgi_engine": sum(out["engine"] == "TGI" for out in cells.values()),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cells = grid(ref)
    return {
        "shipped": {s: ref.pick_engine(*s)["engine"] for s in ref.SCENARIOS},
        "scale_moves": moves(cells, 1), "workload_moves": moves(cells, 2),
        "mine": ref.pick_engine(*MINE),
        "off_menu": {hw: ref.pick_engine(hw, "production", "chat")["engine"] for hw in OFF_MENU},
        "cells": {k: v["engine"] for k, v in cells.items()},
        **tally(cells),
    }


def verify(result):
    cells = result["cells"]
    blackwell = {e for (hw, _, _), e in cells.items() if hw == "NVIDIA Blackwell"}
    return [
        practice.Check(
            "ANSWER: on the lesson's 7 scenarios it matches; off them it does not",
            all([len(result["shipped"]) == 7, None not in result["shipped"].values(),
                 result["scale_moves"] == ["CPU", "Apple Silicon", "NVIDIA Hopper"],
                 result["workload_moves"] == ["AMD", "NVIDIA Hopper"],
                 result["live_workloads"] == ["RAG with heavy prefix reuse", "agentic multi-turn"],
                 result["mine"]["engine"] is None, len(result["mine"]["reasons"]) == 1,
                 set(result["off_menu"].values()) == {None}]),
            f"scale moves the pick on {result['scale_moves']}, workload on "
            f"{result['workload_moves']}; only {result['live_workloads']} ever change it; "
            f"{MINE[0]} -> {result['mine']['engine']}, {result['off_menu']}",
        ),
        practice.Check(
            "FINDING: the tree does not run hardware, then scale, then workload",
            all([cells[("NVIDIA Hopper", "single_user", "agentic multi-turn")] == "SGLang",
                 cells[("AMD", "single_user", "general chat")] == "vLLM",
                 blackwell == {"TRT-LLM"}]),
            "Hopper single-user agentic -> SGLang (workload beats scale); AMD single-user -> "
            f"vLLM; Blackwell returns {sorted(blackwell)} on all 20 cells, agentic included",
        ),
        practice.Check(
            "FINDING: the enterprise line stacks vLLM's production-stack onto engines that are not vLLM",
            result["stacked_off_vllm"] == 19 and result["enterprise"] == 25
            and result["tgi_line"] == 100 and result["tgi_engine"] == 0,
            f"{result['stacked_off_vllm']} of {result['enterprise']} enterprise cells get the "
            f"production-stack line without vLLM; the TGI line is on {result['tgi_line']} of 100 "
            f"outputs, TGI the engine on {result['tgi_engine']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
