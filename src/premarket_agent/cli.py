"""Command-line interface for Premarket Agent."""

from __future__ import annotations

import asyncio

import typer

from .bootstrap import create_simulation, create_system
from .config.schemas import TradeMode
from .config.settings import get_settings
from .gui.dashboard import render_snapshot, run_dashboard

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def run(
    once: bool = typer.Option(False, help="Run a single cycle instead of continuous mode."),
    dashboard: bool = typer.Option(True, help="Display live virtual-trading dashboard."),
) -> None:
    """Start the multi-agent orchestrator."""
    settings = get_settings()
    controller, context = create_system(settings)

    async def _runner() -> None:
        try:
            if once:
                await controller.run_once()
                if dashboard:
                    snapshot = await context.portfolio_service.snapshot(TradeMode.PAPER)
                    from rich.console import Console

                    Console().print(render_snapshot(snapshot))
            else:
                tasks = [controller.run_forever()]
                if dashboard:
                    tasks.append(
                        run_dashboard(
                            portfolio_service=context.portfolio_service,
                            mode=TradeMode.PAPER,
                        )
                    )
                await asyncio.gather(*tasks)
        finally:
            await context.shutdown()

    asyncio.run(_runner())


@app.command()
def simulate(episodes: int = typer.Option(50, help="Number of training episodes.")) -> None:
    """Run offline simulation / reinforcement learning."""
    settings = get_settings()
    simulation, context = create_simulation(settings)

    async def _runner() -> None:
        try:
            result = await simulation.train_policy(episodes)
            typer.echo(result.json(indent=2))
        finally:
            await context.shutdown()

    asyncio.run(_runner())


def main() -> None:
    app()


if __name__ == "__main__":
    main()


