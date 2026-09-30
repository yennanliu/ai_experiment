"""Exercise 5 — MASTER_ADDR and MASTER_PORT stay, the socket variable changes name, and the lesson's code would still fail on a CUDA box.

    Switch the backend to `nccl` on a CUDA box. Note which environment variables change and which stay the same.

Reading of the exercise: there is no CUDA device here, so this is the
scaled-down run D11 asks for. It measures three things. First, which
variables the lesson's own `init_process_group` sets, by diffing
`os.environ` around a real `rank_main(..., backend="nccl")` call. Second,
what that call does on a build without NCCL. Third, whether the variables
said to stay the same are enough on their own: 2 ranks started the way
torchrun starts them (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT in the
environment, `init_process_group` given nothing but the backend) run one
all-reduce over gloo. Which variables NCCL reads comes from the PyTorch
distributed docs (https://docs.pytorch.org/docs/2.14/distributed.html, read
2026-09-29).

**ANSWER: MASTER_ADDR and MASTER_PORT stay the same; GLOO_SOCKET_IFNAME
becomes NCCL_SOCKET_IFNAME; TP_SOCKET_IFNAME is not read by NCCL; under
torchrun, RANK, WORLD_SIZE and LOCAL_RANK are added, and LOCAL_RANK picks the
GPU.** The lesson sets exactly four variables: GLOO_SOCKET_IFNAME,
MASTER_ADDR, MASTER_PORT, TP_SOCKET_IFNAME. The rendezvous variables are
backend-independent: with only RANK, WORLD_SIZE, MASTER_ADDR and MASTER_PORT
set, 2 env:// ranks all-reduce 1 + 2 = 3. On this machine
`is_nccl_available()` is False and `rank_main` with backend "nccl" returns
"Distributed package doesn't have NCCL built in" straight away, with no hang.

**FINDING: the lesson's code would fail on a CUDA box too.** The doc says
"the only changes are backend="nccl", device tensors, and torchrun", but
there is no device to change: `make_model`, `rank_main`,
`manual_all_reduce_matches_single_process` and `fsdp_round_trip_sketch`
never mention a device, `cuda` or `.to(`, so every tensor is on CPU. The
docs' backend table marks all NCCL collectives as unsupported on CPU, so
`--backend nccl` would fail at the first broadcast. `main()` also only
checks that gloo is available, never nccl.

Structure: `launch()` starts this file once per rank as a subprocess with a
60 s timeout and kills stragglers. Expected output: two PASS checks.
"""

from __future__ import annotations

import inspect
import json
import os
import socket
import subprocess
import sys
import time

from harness import parity, practice

try:
    import torch
    import torch.distributed as dist
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "48-distributed-fsdp-ddp"
LESSON_FUNCS = ("make_model", "rank_main", "manual_all_reduce_matches_single_process", "fsdp_round_trip_sketch")


def launch(ws, *args, timeout=60):
    """Run this file as `ws` gloo ranks in subprocesses; return each rank's JSON, kill stragglers."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        argv = [sys.executable, __file__, "--rank", str(ws), str(s.getsockname()[1]), *map(str, args)]
    pipe = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    procs = [subprocess.Popen(argv, env=os.environ | {"RANK": str(r)}, **pipe) for r in range(ws)]
    try:
        outs = [p.communicate(timeout=timeout) for p in procs]
    finally:
        for p in procs:
            p.kill()
    if any(p.returncode for p in procs):
        raise RuntimeError(f"a rank failed: {[err[-300:] for _, err in outs]}")
    return [json.loads(out.splitlines()[-1]) for out, _ in outs]


def lesson_nccl(port):
    """The lesson's own rank_main with backend="nccl": its env changes and its outcome."""
    ref, queue, before = parity.load_reference(PHASE, LESSON, "main"), [], set(os.environ)
    sink = type("Queue", (), {"put": lambda self, item: queue.append(item)})()
    start = time.perf_counter()
    ref.rank_main(0, 1, "nccl", port, sink, 32, 16, 4, 8, 1, 0.05, 0)
    return {"added": sorted(set(os.environ) - before), "error": queue[0][1].get("error"),
            "seconds": time.perf_counter() - start, "nccl": dist.is_nccl_available()}


def torchrun_style(ws, port):
    """env:// init from RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT alone, then one all-reduce."""
    os.environ.update(WORLD_SIZE=str(ws), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo")
    try:
        total = torch.tensor([float(dist.get_rank() + 1)])
        dist.all_reduce(total)
        return {"rank": dist.get_rank(), "world": dist.get_world_size(), "total": total.item()}
    finally:
        dist.destroy_process_group()


def worker(rank, ws, port, mode):
    return lesson_nccl(int(port)) if mode == "lesson_nccl" else torchrun_style(int(ws), int(port))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    code = " ".join(inspect.getsource(getattr(ref, name)) for name in LESSON_FUNCS)
    return {
        "lesson": launch(1, "lesson_nccl")[0],
        "env_init": launch(2, "torchrun_style"),
        "mentions": [word for word in ("device", "cuda", ".to(") if word in code],
        "main_checks": [b for b in ("is_gloo_available", "is_nccl_available") if b in inspect.getsource(ref.main)],
        "doc": "the only changes are `backend=\"nccl\"`, device tensors, and `torchrun`" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    lesson = result["lesson"]
    return [
        practice.Check(
            "ANSWER: MASTER_ADDR/MASTER_PORT stay; GLOO_SOCKET_IFNAME becomes NCCL_SOCKET_IFNAME",
            lesson["added"] == ["GLOO_SOCKET_IFNAME", "MASTER_ADDR", "MASTER_PORT", "TP_SOCKET_IFNAME"]
            and [(r["rank"], r["world"], r["total"]) for r in result["env_init"]] == [(0, 2, 3.0), (1, 2, 3.0)]
            and not lesson["nccl"] and lesson["error"] == "Distributed package doesn't have NCCL built in"
            and lesson["seconds"] < 30,
            f"lesson sets {lesson['added']}; env:// ranks {result['env_init']}; nccl here: "
            f"{lesson['error']!r} after {lesson['seconds']:.2f} s",
        ),
        practice.Check(
            "FINDING: the lesson's code would fail on a CUDA box too",
            result["mentions"] == [] and result["main_checks"] == ["is_gloo_available"] and result["doc"],
            f"device/cuda/.to( in the lesson's functions: {result['mentions']}; main() checks {result['main_checks']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(os.environ["RANK"], *sys.argv[2:]))))
    raise SystemExit(practice.selfcheck(globals()))
