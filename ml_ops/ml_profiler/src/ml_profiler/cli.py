"""CLI interface for ML Profiler."""

import json
import random
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .db import ProfileDB
from .profiler import ProfileMetrics, profile_callable, TORCH_AVAILABLE, KernelMetric

if TORCH_AVAILABLE:
    import torch
    from .profiler import profile_torch_model

console = Console()


def generate_demo_metrics(model_name: str, version: str) -> ProfileMetrics:
    """Generate demo metrics for testing without torch."""
    base_times = {"simple": 0.5, "medium": 2.0, "complex": 8.0}
    base_params = {"simple": 1000, "medium": 50000, "complex": 500000}

    model_type = "simple"
    if "resnet" in model_name.lower():
        model_type = "medium"
    elif "vit" in model_name.lower() or "transformer" in model_name.lower():
        model_type = "complex"

    # Add version-based variation for bisection testing
    version_factor = 1.0
    if version:
        parts = version.split(".")
        if len(parts) >= 2:
            minor = int(parts[1]) if parts[1].isdigit() else 0
            version_factor = 1.0 + (minor * 0.05)  # Each minor version 5% slower

    time_ms = base_times[model_type] * version_factor * (1 + random.uniform(-0.05, 0.05))
    params = base_params[model_type]

    # Generate fake kernel metrics
    kernel_metrics = [
        KernelMetric("aten::conv2d", time_ms * 0.4, 0.0, 10),
        KernelMetric("aten::relu", time_ms * 0.1, 0.0, 20),
        KernelMetric("aten::batch_norm", time_ms * 0.15, 0.0, 10),
        KernelMetric("aten::linear", time_ms * 0.2, 0.0, 5),
        KernelMetric("aten::add", time_ms * 0.05, 0.0, 15),
    ]

    top_kernels_json = json.dumps([
        {"name": k.name, "cpu_time_ms": k.cpu_time_ms, "cuda_time_ms": k.cuda_time_ms, "calls": k.calls}
        for k in kernel_metrics
    ])

    return ProfileMetrics(
        model_name=model_name,
        model_version=version,
        hardware_target="cpu",
        total_time_ms=time_ms,
        cpu_time_ms=time_ms,
        cuda_time_ms=0.0,
        memory_allocated_mb=params * 4 / (1024 * 1024),
        memory_reserved_mb=0.0,
        total_params=params,
        total_flops=params * 2,
        input_shape="[1, 3, 224, 224]",
        kernel_metrics=kernel_metrics,
        top_kernels_json=top_kernels_json,
    )


@click.group()
def main():
    """Mini ML Profiler - Profile ML models and track performance."""
    pass


@main.command()
@click.option("--model", "-m", default="demo_model", help="Model name")
@click.option("--version", "-v", default="1.0.0", help="Model version")
@click.option("--demo", is_flag=True, help="Use demo mode (no torch required)")
@click.option("--save/--no-save", default=True, help="Save results to database")
@click.option("--notes", "-n", default="", help="Notes for this profile run")
def profile(model: str, version: str, demo: bool, save: bool, notes: str):
    """Profile a model."""
    console.print(f"[bold blue]Profiling {model} v{version}...[/bold blue]")

    if demo or not TORCH_AVAILABLE:
        if not demo and not TORCH_AVAILABLE:
            console.print("[yellow]torch not installed, using demo mode[/yellow]")
        metrics = generate_demo_metrics(model, version)
    else:
        # Real torch profiling
        try:
            if model == "resnet18":
                torch_model = torch.hub.load("pytorch/vision", "resnet18", weights=None)
                input_tensor = torch.randn(1, 3, 224, 224)
            elif model == "resnet50":
                torch_model = torch.hub.load("pytorch/vision", "resnet50", weights=None)
                input_tensor = torch.randn(1, 3, 224, 224)
            else:
                # Create simple model for unknown names
                torch_model = torch.nn.Sequential(
                    torch.nn.Linear(100, 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50, 10),
                )
                input_tensor = torch.randn(1, 100)

            metrics = profile_torch_model(
                torch_model, input_tensor,
                model_name=model, model_version=version,
                detailed=True,
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    # Display results
    table = Table(title="Profile Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", metrics.model_name)
    table.add_row("Version", metrics.model_version)
    table.add_row("Hardware", metrics.hardware_target)
    table.add_row("Total Time", f"{metrics.total_time_ms:.2f} ms")
    table.add_row("Parameters", f"{metrics.total_params:,}")
    table.add_row("Input Shape", metrics.input_shape)

    console.print(table)

    # Show top kernels if available
    if metrics.kernel_metrics:
        kernel_table = Table(title="Top Kernels (by CPU time)")
        kernel_table.add_column("Operation", style="cyan")
        kernel_table.add_column("CPU (ms)", style="green")
        kernel_table.add_column("Calls", style="yellow")

        for k in metrics.kernel_metrics[:5]:
            kernel_table.add_row(k.name, f"{k.cpu_time_ms:.3f}", str(k.calls))

        console.print(kernel_table)

    if save:
        try:
            db = ProfileDB()
            result = db.save(metrics, notes=notes)
            console.print(f"[green]Saved to database (id={result.id})[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not save: {e}[/yellow]")


@main.command()
@click.argument("model_name")
@click.option("--baseline", "-b", default=None, help="Baseline version (default: earliest)")
@click.option("--target", "-t", default=None, help="Target version (default: latest)")
@click.option("--threshold", default=10.0, help="Regression threshold percentage")
def bisect(model_name: str, baseline: str, target: str, threshold: float):
    """Detect performance regressions between versions."""
    try:
        db = ProfileDB()
        results = db.compare_versions(model_name)

        if not results:
            console.print(f"[yellow]No results for model: {model_name}[/yellow]")
            return

        # Group by version
        versions: dict = {}
        for r in results:
            if r.model_version not in versions:
                versions[r.model_version] = []
            versions[r.model_version].append(r)

        if len(versions) < 2:
            console.print("[yellow]Need at least 2 versions to compare[/yellow]")
            return

        # Sort versions
        sorted_versions = sorted(versions.keys())

        # Determine baseline and target
        baseline_ver = baseline or sorted_versions[0]
        target_ver = target or sorted_versions[-1]

        if baseline_ver not in versions:
            console.print(f"[red]Baseline version {baseline_ver} not found[/red]")
            return
        if target_ver not in versions:
            console.print(f"[red]Target version {target_ver} not found[/red]")
            return

        # Calculate stats
        baseline_times = [r.total_time_ms for r in versions[baseline_ver]]
        target_times = [r.total_time_ms for r in versions[target_ver]]

        baseline_avg = sum(baseline_times) / len(baseline_times)
        target_avg = sum(target_times) / len(target_times)
        diff_pct = ((target_avg - baseline_avg) / baseline_avg) * 100

        # Display comparison
        console.print(Panel(f"[bold]Bisection: {model_name}[/bold]"))

        table = Table()
        table.add_column("Version", style="cyan")
        table.add_column("Avg Time (ms)", style="green")
        table.add_column("Runs", style="yellow")
        table.add_column("Status", style="bold")

        table.add_row(
            f"{baseline_ver} (baseline)",
            f"{baseline_avg:.2f}",
            str(len(baseline_times)),
            "📊"
        )
        table.add_row(
            f"{target_ver} (target)",
            f"{target_avg:.2f}",
            str(len(target_times)),
            "📊"
        )

        console.print(table)

        # Regression detection
        if diff_pct > threshold:
            console.print(f"\n[bold red]⚠️  REGRESSION DETECTED[/bold red]")
            console.print(f"[red]Performance degraded by {diff_pct:.1f}% (threshold: {threshold}%)[/red]")

            # Show intermediate versions if any
            intermediate = [v for v in sorted_versions if baseline_ver < v < target_ver]
            if intermediate:
                console.print(f"\n[yellow]Intermediate versions to investigate:[/yellow]")
                for v in intermediate:
                    v_times = [r.total_time_ms for r in versions[v]]
                    v_avg = sum(v_times) / len(v_times)
                    v_diff = ((v_avg - baseline_avg) / baseline_avg) * 100
                    marker = "🔴" if v_diff > threshold else "🟢"
                    console.print(f"  {marker} {v}: {v_avg:.2f}ms ({v_diff:+.1f}%)")

            return 1  # Exit code for CI

        elif diff_pct < -threshold:
            console.print(f"\n[bold green]✅ IMPROVEMENT[/bold green]")
            console.print(f"[green]Performance improved by {abs(diff_pct):.1f}%[/green]")
        else:
            console.print(f"\n[bold blue]ℹ️  STABLE[/bold blue]")
            console.print(f"[blue]Performance change: {diff_pct:+.1f}% (within threshold)[/blue]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@main.command()
@click.option("--model", "-m", default=None, help="Filter by model name")
@click.option("--limit", "-l", default=20, help="Number of results")
def history(model: str, limit: int):
    """Show profiling history."""
    try:
        db = ProfileDB()
        results = db.get_history(model, limit=limit) if model else db.get_all(limit=limit)

        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        table = Table(title="Profile History")
        table.add_column("ID", style="dim")
        table.add_column("Model", style="cyan")
        table.add_column("Version", style="blue")
        table.add_column("Time (ms)", style="green")
        table.add_column("Params", style="yellow")
        table.add_column("Date", style="dim")

        for r in results:
            table.add_row(
                str(r.id),
                r.model_name,
                r.model_version,
                f"{r.total_time_ms:.2f}",
                f"{r.total_params:,}" if r.total_params else "-",
                r.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)
    except Exception as e:
        console.print(f"[red]Database error: {e}[/red]")


@main.command()
def models():
    """List all profiled models."""
    try:
        db = ProfileDB()
        model_names = db.get_models()

        if not model_names:
            console.print("[yellow]No models found[/yellow]")
            return

        console.print("[bold]Profiled Models:[/bold]")
        for name in model_names:
            console.print(f"  - {name}")
    except Exception as e:
        console.print(f"[red]Database error: {e}[/red]")


@main.command()
def serve():
    """Start the web dashboard."""
    import uvicorn
    console.print("[bold blue]Starting dashboard at http://localhost:8000[/bold blue]")
    uvicorn.run("ml_profiler.dashboard:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
