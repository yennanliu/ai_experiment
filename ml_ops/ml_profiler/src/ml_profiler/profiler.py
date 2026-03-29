"""Core profiling logic for ML models."""

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# Try to import torch, but make it optional
try:
    import torch
    from torch.profiler import profile, ProfilerActivity
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class KernelMetric:
    """Metrics for a single kernel/operation."""
    name: str
    cpu_time_ms: float
    cuda_time_ms: float
    calls: int


@dataclass
class ProfileMetrics:
    """Container for profiling metrics."""

    model_name: str
    model_version: str
    hardware_target: str
    total_time_ms: float
    cpu_time_ms: float
    cuda_time_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    total_params: int
    total_flops: float
    input_shape: str
    # Detailed kernel breakdown (from torch.profiler)
    kernel_metrics: list[KernelMetric] = field(default_factory=list)
    top_kernels_json: str = "[]"  # JSON string for DB storage


@runtime_checkable
class Callable(Protocol):
    """Protocol for callable objects."""
    def __call__(self, *args, **kwargs) -> Any: ...


def profile_callable(
    func: Callable,
    *args,
    model_name: str = "unknown",
    model_version: str = "1.0.0",
    warmup_runs: int = 3,
    profile_runs: int = 10,
    **kwargs,
) -> ProfileMetrics:
    """Profile any callable function (torch-free)."""
    # Warmup
    for _ in range(warmup_runs):
        _ = func(*args, **kwargs)

    # Profile
    times = []
    for _ in range(profile_runs):
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = sum(times) / len(times)

    return ProfileMetrics(
        model_name=model_name,
        model_version=model_version,
        hardware_target="cpu",
        total_time_ms=avg_time,
        cpu_time_ms=avg_time,
        cuda_time_ms=0.0,
        memory_allocated_mb=0.0,
        memory_reserved_mb=0.0,
        total_params=0,
        total_flops=0.0,
        input_shape=str(args) if args else "[]",
    )


def profile_torch_model(
    model: Any,
    input_tensor: Any,
    model_name: str = "unknown",
    model_version: str = "1.0.0",
    warmup_runs: int = 3,
    profile_runs: int = 10,
    detailed: bool = True,
) -> ProfileMetrics:
    """Profile a PyTorch model with optional detailed kernel metrics."""
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required. Install with: uv pip install ml-profiler[torch]")

    import json

    device = next(model.parameters()).device
    hardware_target = "cuda" if device.type == "cuda" else "cpu"
    input_tensor = input_tensor.to(device)

    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    if hardware_target == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Detailed profiling with torch.profiler
    kernel_metrics = []
    if detailed:
        activities = [ProfilerActivity.CPU]
        if hardware_target == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, record_shapes=True) as prof:
            with torch.no_grad():
                _ = model(input_tensor)

        # Extract top operations
        key_averages = prof.key_averages()
        for evt in sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:10]:
            kernel_metrics.append(KernelMetric(
                name=evt.key,
                cpu_time_ms=evt.cpu_time_total / 1000,  # us to ms
                cuda_time_ms=(evt.cuda_time_total / 1000) if hasattr(evt, 'cuda_time_total') else 0,
                calls=evt.count,
            ))

    # Standard timing profiling
    cpu_times = []
    cuda_times = []

    with torch.no_grad():
        for _ in range(profile_runs):
            if hardware_target == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            cpu_start = time.perf_counter()
            _ = model(input_tensor)
            cpu_end = time.perf_counter()

            if hardware_target == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                cuda_times.append(start_event.elapsed_time(end_event))

            cpu_times.append((cpu_end - cpu_start) * 1000)

    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    avg_cuda_time = sum(cuda_times) / len(cuda_times) if cuda_times else 0.0
    total_time = avg_cuda_time if hardware_target == "cuda" else avg_cpu_time

    # Memory
    if hardware_target == "cuda":
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        memory_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    else:
        memory_allocated = 0.0
        memory_reserved = 0.0

    # Params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Serialize kernel metrics to JSON
    top_kernels_json = json.dumps([
        {"name": k.name, "cpu_time_ms": k.cpu_time_ms, "cuda_time_ms": k.cuda_time_ms, "calls": k.calls}
        for k in kernel_metrics
    ])

    return ProfileMetrics(
        model_name=model_name,
        model_version=model_version,
        hardware_target=hardware_target,
        total_time_ms=total_time,
        cpu_time_ms=avg_cpu_time,
        cuda_time_ms=avg_cuda_time,
        memory_allocated_mb=memory_allocated,
        memory_reserved_mb=memory_reserved,
        total_params=total_params,
        total_flops=0.0,
        input_shape=str(list(input_tensor.shape)),
        kernel_metrics=kernel_metrics,
        top_kernels_json=top_kernels_json,
    )


# Alias for backward compatibility
def profile_model(*args, **kwargs) -> ProfileMetrics:
    """Profile a model (auto-detect torch or callable)."""
    if TORCH_AVAILABLE and len(args) >= 2:
        return profile_torch_model(*args, **kwargs)
    return profile_callable(*args, **kwargs)
