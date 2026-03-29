"""CLI interface for ML Profiler."""

import random
import click
from rich.console import Console
from rich.table import Table

from .db import ProfileDB
from .profiler import ProfileMetrics, profile_callable, TORCH_AVAILABLE

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

    time_ms = base_times[model_type] * (1 + random.uniform(-0.1, 0.1))
    params = base_params[model_type]

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

    if save:
        try:
            db = ProfileDB()
            result = db.save(metrics, notes=notes)
            console.print(f"[green]Saved to database (id={result.id})[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not save: {e}[/yellow]")


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
