"""
Project 13: Streaming Evaluator Dashboard — Rich TUI.

Shows live project statuses, scores, and streaming LLM tokens as the
orchestrator runs. Uses Rich Live + Layout for a real-time terminal view.
"""

import asyncio
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import config


class Dashboard:
    def __init__(self, goal: str, gen_id: str = "") -> None:
        self.goal = goal
        self.gen_id = gen_id
        self.console = Console()
        self._projects: dict[str, str] = {}    # label → "pending"|"running"|"done"|"error"
        self._scores: dict[str, dict] = {}     # label → scores dict
        self._stream: list[str] = []           # rolling token buffer (last ~400 chars)
        self._log: list[str] = []              # activity log lines
        self._phase = "starting"

    # ── Public update API ────────────────────────────────────────────────────

    def set_phase(self, phase: str) -> None:
        self._phase = phase
        self._log_line(f"Phase: {phase}")

    def set_projects(self, projects: list[dict]) -> None:
        for p in projects:
            if p["label"] not in self._projects:
                self._projects[p["label"]] = "pending"

    def set_status(self, label: str, status: str) -> None:
        self._projects[label] = status
        self._log_line(f"{label} → {status}")

    def set_scores(self, label: str, scores: dict) -> None:
        self._scores[label] = scores

    def push_token(self, token: str) -> None:
        self._stream.extend(list(token))
        if len(self._stream) > 600:
            self._stream = self._stream[-600:]

    # ── Live runner ──────────────────────────────────────────────────────────

    async def run(self, coro) -> Any:
        """Await coro while keeping the dashboard live. Returns coro's result."""
        with Live(self._render(), refresh_per_second=10, console=self.console) as live:
            async def _tick():
                while True:
                    live.update(self._render())
                    await asyncio.sleep(0.1)

            ticker = asyncio.create_task(_tick())
            try:
                result = await coro
            finally:
                ticker.cancel()
                try:
                    await ticker
                except asyncio.CancelledError:
                    pass
        return result

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="stream", size=9),
        )
        layout["body"].split_row(
            Layout(name="projects", ratio=3),
            Layout(name="log", ratio=2),
        )
        layout["header"].update(self._header())
        layout["projects"].update(self._projects_panel())
        layout["log"].update(self._log_panel())
        layout["stream"].update(self._stream_panel())
        return layout

    def _header(self) -> Panel:
        goal_preview = self.goal[:80] + "…" if len(self.goal) > 80 else self.goal
        run_info = f"run=[bold]{self.gen_id}[/]  " if self.gen_id else ""
        return Panel(
            f"[bold cyan]Multi-Project Orchestrator[/]  ·  "
            f"[dim]{config.PROVIDER}/{config.active_model()}[/]  ·  "
            f"{run_info}"
            f"[yellow]{goal_preview}[/]  ·  "
            f"[dim]{self._phase}[/]",
        )

    def _projects_panel(self) -> Panel:
        table = Table(show_header=True, header_style="bold magenta", expand=True, box=None)
        table.add_column("Sub-project", style="cyan", ratio=4)
        table.add_column("Status", justify="center", ratio=2)
        table.add_column("C/Co/Q", justify="right", ratio=2)
        table.add_column("Overall", justify="right", ratio=1)

        _icon = {"pending": "○", "running": "●", "done": "✓", "error": "✗"}
        _style = {"pending": "dim", "running": "yellow bold", "done": "green", "error": "red"}

        for label, status in self._projects.items():
            s = self._scores.get(label, {})
            dims = (
                f"{s.get('completeness','?')}/{s.get('correctness','?')}/{s.get('quality','?')}"
                if s else "—/—/—"
            )
            overall = f"[bold]{s.get('overall','?')}[/]/5" if s else "—"
            st = _style.get(status, "")
            icon = _icon.get(status, "?")
            table.add_row(label, f"[{st}]{icon} {status}[/]", dims, overall)

        if not self._projects:
            table.add_row("[dim]waiting for director…[/]", "", "", "")

        return Panel(table, title="[bold]Projects[/]", border_style="blue")

    def _log_panel(self) -> Panel:
        lines = self._log[-12:]
        body = Text("\n".join(lines) if lines else "Waiting…", overflow="fold")
        return Panel(body, title="[bold]Activity[/]", border_style="dim")

    def _stream_panel(self) -> Panel:
        raw = "".join(self._stream[-500:])
        body = Text(raw or "Waiting for evaluator stream…", style="dim green", overflow="fold")
        return Panel(body, title="[bold green]Live LLM Stream (evaluator)[/]", border_style="green")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _log_line(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[dim]{ts}[/] {msg}")
