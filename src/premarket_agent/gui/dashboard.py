"""Rich-based dashboard for monitoring virtual trading performance."""

from __future__ import annotations

import asyncio
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..config.schemas import PortfolioSnapshot, TradeMode
from ..services.portfolio import PortfolioService


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _format_percent(value: float) -> str:
    return f"{value:+.2f}%"


def _summary_table(snapshot: PortfolioSnapshot) -> Table:
    table = Table.grid(expand=True)
    table.add_column(justify="left")
    table.add_column(justify="right")
    net_profit = snapshot.realized_pnl + snapshot.unrealized_pnl
    profit_style = "green" if net_profit >= 0 else "red"
    table.add_row("Cash", _format_currency(snapshot.cash))
    table.add_row("Net Profit", Text(_format_currency(net_profit), style=profit_style))
    table.add_row("Unrealized PnL", Text(_format_currency(snapshot.unrealized_pnl), style="green" if snapshot.unrealized_pnl >= 0 else "red"))
    table.add_row("Realized PnL", Text(_format_currency(snapshot.realized_pnl), style="green" if snapshot.realized_pnl >= 0 else "red"))
    table.add_row("Net Liquidation", _format_currency(snapshot.net_liquidation))
    table.add_row("Updated", snapshot.timestamp.strftime("%H:%M:%S UTC"))
    return table


def _positions_table(snapshot: PortfolioSnapshot) -> Table:
    table = Table(
        title="Open Positions",
        expand=True,
        show_lines=False,
    )
    table.add_column("Symbol", justify="center")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Gain", justify="right")
    table.add_column("Change", justify="right")
    if not snapshot.positions:
        table.add_row("-", "-", "-", "-", "-", "-")
        return table
    for position in snapshot.positions:
        gain_style = "green" if position.gain >= 0 else "red"
        table.add_row(
            position.symbol,
            str(position.quantity),
            _format_currency(position.avg_price),
            _format_currency(position.current_price),
            Text(_format_currency(position.gain), style=gain_style),
            Text(_format_percent(position.gain_percent), style=gain_style),
        )
    return table


def _build_renderable(snapshot: PortfolioSnapshot) -> Panel:
    summary = _summary_table(snapshot)
    positions = _positions_table(snapshot)
    content = Group(summary, positions)
    return Panel(content, title="Virtual Trading Dashboard", border_style="blue")


def render_snapshot(snapshot: PortfolioSnapshot) -> Panel:
    """Expose a single renderable for ad-hoc printing."""
    return _build_renderable(snapshot)


async def run_dashboard(
    portfolio_service: PortfolioService,
    mode: TradeMode = TradeMode.PAPER,
    refresh_interval: float = 2.0,
    console: Optional[Console] = None,
) -> None:
    """Continuously render the live dashboard until cancelled."""
    console = console or Console()
    try:
        with Live(console=console, refresh_per_second=2, screen=False) as live:
            while True:
                snapshot = await portfolio_service.snapshot(mode)
                live.update(_build_renderable(snapshot))
                await asyncio.sleep(refresh_interval)
    except asyncio.CancelledError:
        return

