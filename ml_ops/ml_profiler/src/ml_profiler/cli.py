"""CLI interface for ML Profiler."""

import click
import torch
from rich.console import Console
from rich.table import Table

from .db import ProfileDB
from .profiler import profile_model

console = Console()


@click.group()
def main():
    """Mini ML Profiler - Profile PyTorch models and track performance."""
    pass


@main.command()
@click.option("--model", "-m", default="resnet18", help="Model name (resnet18, resnet50, vit)")
@click.option("--version", "-v", default="1.0.0", help="Model version")
@click.option("--batch-size", "-b", default=1, help="Batch size")
@click.option("--save/--no-save", default=True, help="Save results to database")
@click.option("--notes", "-n", default="", help="Notes for this profile run")
def profile(model: str, version: str, batch_size: int, save: bool, notes: str):
    """Profile a built-in PyTorch model."""
    console.print(f"[bold blue]Profiling {model} v{version}...[/bold blue]")

    # Load model
    try:
        if model == "resnet18":
            torch_model = torch.hub.load("pytorch/vision", "resnet18", weights=None)
            input_tensor = torch.randn(batch_size, 3, 224, 224)
        elif model == "resnet50":
            torch_model = torch.hub.load("pytorch/vision", "resnet50", weights=None)
            input_tensor = torch.randn(batch_size, 3, 224, 224)
        elif model == "vit":
            torch_model = torch.hub.load("pytorch/vision", "vit_b_16", weights=None)
            input_tensor = torch.randn(batch_size, 3, 224, 224)
        else:
            console.print(f"[red]Unknown model: {model}[/red]")
            return
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        return

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_model = torch_model.to(device)
    console.print(f"[dim]Using device: {device}[/dim]")

    # Profile
    metrics = profile_model(
        torch_model,
        input_tensor,
        model_name=model,
        model_version=version,
    )

    # Display results
    table = Table(title="Profile Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", metrics.model_name)
    table.add_row("Version", metrics.model_version)
    table.add_row("Hardware", metrics.hardware_target)
    table.add_row("Total Time", f"{metrics.total_time_ms:.2f} ms")
    table.add_row("CPU Time", f"{metrics.cpu_time_ms:.2f} ms")
    table.add_row("CUDA Time", f"{metrics.cuda_time_ms:.2f} ms")
    table.add_row("Memory Allocated", f"{metrics.memory_allocated_mb:.2f} MB")
    table.add_row("Parameters", f"{metrics.total_params:,}")
    table.add_row("FLOPs", f"{metrics.total_flops:,.0f}")
    table.add_row("Input Shape", metrics.input_shape)

    console.print(table)

    # Save to database
    if save:
        try:
            db = ProfileDB()
            result = db.save(metrics, notes=notes)
            console.print(f"[green]Saved to database (id={result.id})[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not save to database: {e}[/yellow]")


@main.command()
@click.option("--model", "-m", default=None, help="Filter by model name")
@click.option("--limit", "-l", default=20, help="Number of results")
def history(model: str, limit: int):
    """Show profiling history."""
    try:
        db = ProfileDB()
        if model:
            results = db.get_history(model, limit=limit)
        else:
            results = db.get_all(limit=limit)

        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        table = Table(title="Profile History")
        table.add_column("ID", style="dim")
        table.add_column("Model", style="cyan")
        table.add_column("Version", style="blue")
        table.add_column("Hardware", style="magenta")
        table.add_column("Time (ms)", style="green")
        table.add_column("Memory (MB)", style="yellow")
        table.add_column("Date", style="dim")

        for r in results:
            table.add_row(
                str(r.id),
                r.model_name,
                r.model_version,
                r.hardware_target,
                f"{r.total_time_ms:.2f}",
                f"{r.memory_allocated_mb:.2f}",
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
            console.print(f"  • {name}")
    except Exception as e:
        console.print(f"[red]Database error: {e}[/red]")


@main.command()
@click.argument("model_name")
def compare(model_name: str):
    """Compare versions of a model."""
    try:
        db = ProfileDB()
        results = db.compare_versions(model_name)

        if not results:
            console.print(f"[yellow]No results for model: {model_name}[/yellow]")
            return

        table = Table(title=f"Version Comparison: {model_name}")
        table.add_column("Version", style="cyan")
        table.add_column("Time (ms)", style="green")
        table.add_column("Memory (MB)", style="yellow")
        table.add_column("Hardware", style="magenta")
        table.add_column("Runs", style="dim")

        # Group by version
        versions: dict = {}
        for r in results:
            if r.model_version not in versions:
                versions[r.model_version] = []
            versions[r.model_version].append(r)

        for version, runs in versions.items():
            avg_time = sum(r.total_time_ms for r in runs) / len(runs)
            avg_mem = sum(r.memory_allocated_mb for r in runs) / len(runs)
            hardware = runs[0].hardware_target
            table.add_row(
                version,
                f"{avg_time:.2f}",
                f"{avg_mem:.2f}",
                hardware,
                str(len(runs)),
            )

        console.print(table)
    except Exception as e:
        console.print(f"[red]Database error: {e}[/red]")


if __name__ == "__main__":
    main()
