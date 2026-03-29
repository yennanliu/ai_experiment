"""Core profiling logic for PyTorch models."""

import time
from dataclasses import dataclass
from typing import Any

import torch


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


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
    """Estimate FLOPs for a forward pass (simplified estimation)."""
    total_flops = 0.0

    def hook_fn(module, input, output):
        nonlocal total_flops
        if isinstance(module, torch.nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            batch_size = input[0].shape[0] if input[0].dim() > 1 else 1
            total_flops += 2 * batch_size * in_features * out_features
        elif isinstance(module, torch.nn.Conv2d):
            out_h, out_w = output.shape[2], output.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            total_flops += 2 * output.shape[0] * module.out_channels * out_h * out_w * kernel_ops

    hooks = []
    for layer in model.modules():
        hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return total_flops


def profile_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    model_name: str = "unknown",
    model_version: str = "1.0.0",
    warmup_runs: int = 3,
    profile_runs: int = 10,
) -> ProfileMetrics:
    """Profile a PyTorch model with the given input."""
    device = next(model.parameters()).device
    hardware_target = "cuda" if device.type == "cuda" else "cpu"
    input_tensor = input_tensor.to(device)

    # Warmup runs
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    # Synchronize if using CUDA
    if hardware_target == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Profile runs
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

    # Calculate metrics
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    avg_cuda_time = sum(cuda_times) / len(cuda_times) if cuda_times else 0.0
    total_time = avg_cuda_time if hardware_target == "cuda" else avg_cpu_time

    # Memory metrics
    if hardware_target == "cuda":
        memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    else:
        memory_allocated = 0.0
        memory_reserved = 0.0

    # Model metrics
    total_params = count_parameters(model)
    total_flops = estimate_flops(model, input_tensor)
    input_shape = str(list(input_tensor.shape))

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
        total_flops=total_flops,
        input_shape=input_shape,
    )
