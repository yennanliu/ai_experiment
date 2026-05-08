"""
Parallel Agent Fan-out — the core harness pattern in app3.

Instead of running generators sequentially, we submit one thread per plan
component to a ThreadPoolExecutor. Each thread receives an isolated copy of
the shared memory so agents can't corrupt each other's state.

After all threads complete (or fail), memory deltas are merged back into the
shared dict using last-write-wins per key.

Usage:
    results, merged_mem = fan_out.run(task, plan, generator.run, shared_mem)
"""

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Callable

from . import config


@dataclass
class ComponentResult:
    component: str
    artifact: str
    text: str
    elapsed: float = 0.0
    error: str | None = None
    mem_delta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


def run(
    task: str,
    plan: dict,
    generate_fn: Callable[[str, dict, str, dict], str],
    shared_mem: dict,
    max_workers: int | None = None,
) -> tuple[list[ComponentResult], dict]:
    """
    Fan out generation across all plan components in parallel.

    Args:
        task:         Original task string (passed through to each generator).
        plan:         Planner output with 'components' and 'artifacts' lists.
        generate_fn:  Callable matching generator.run(task, plan, component, mem).
        shared_mem:   Memory snapshot to clone for each thread.
        max_workers:  Thread pool size; defaults to min(len(components), MAX_WORKERS).

    Returns:
        (results, merged_memory) — results are in original component order;
        failed components have result.ok == False with result.error set.
    """
    components = plan["components"]
    artifacts = plan.get("artifacts", components)
    workers = min(max_workers or len(components), config.MAX_WORKERS)

    results: list[ComponentResult | None] = [None] * len(components)

    def _run_one(idx: int, component: str) -> None:
        artifact = artifacts[idx] if idx < len(artifacts) else component
        mem_copy = dict(shared_mem)  # isolated snapshot — no shared state
        t0 = time.perf_counter()
        try:
            text = generate_fn(task, plan, component, mem_copy, verbose=False)
            results[idx] = ComponentResult(
                component=component,
                artifact=artifact,
                text=text,
                elapsed=time.perf_counter() - t0,
                mem_delta=mem_copy,
            )
        except Exception as exc:
            results[idx] = ComponentResult(
                component=component,
                artifact=artifact,
                text="",
                elapsed=time.perf_counter() - t0,
                error=str(exc),
                mem_delta=mem_copy,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, i, c) for i, c in enumerate(components)]
        concurrent.futures.wait(futures)

    # Re-raise any unhandled exceptions from the futures themselves
    for f in futures:
        f.result()  # no-op if already handled; propagates executor-level errors

    ordered = [r for r in results if r is not None]

    # Merge memory: last-write-wins per key across all thread snapshots
    merged_mem = dict(shared_mem)
    for r in ordered:
        merged_mem.update(r.mem_delta)

    return ordered, merged_mem
