"""Desktop dashboard for the premarket agent."""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QMainWindow,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..agents import AgentController
from ..bootstrap import ApplicationContext, create_system
from ..config.schemas import (
    OrderEvent,
    PortfolioSnapshot,
    ResearchArtifact,
    TradeMode,
    TrendingSymbol,
)
from ..config.settings import get_settings
from ..services.portfolio import PortfolioService

METRIC_FORMAT = "${:,.2f}"


class DashboardWindow(QMainWindow):
    """Qt window that renders live trading status."""

    def __init__(self, controller: AgentController, context: ApplicationContext) -> None:
        super().__init__()
        self.setWindowTitle("Premarket Agent - Paper Trading Dashboard")
        self.resize(1200, 720)

        self._controller = controller
        self._context = context
        self._portfolio: PortfolioService = context.portfolio_service

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        self._build_ui()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(2000)
        self._refresh_timer.timeout.connect(self._refresh_dashboard)
        self._refresh_timer.start()

    # ------------------------------------------------------------------ UI setup

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # Metrics grid
        metrics_grid = QGridLayout()
        self._metrics_labels: Dict[str, QLabel] = {}
        metric_keys = [
            ("cash", "Cash"),
            ("net_profit", "Net Profit"),
            ("unrealized", "Unrealized PnL"),
            ("realized", "Realized PnL"),
            ("net_liquidation", "Net Liquidation"),
            ("open_positions", "Open Positions"),
            ("pending_orders", "Recent Orders"),
        ]
        for index, (key, label) in enumerate(metric_keys):
            title = QLabel(label)
            title.setStyleSheet("font-weight: bold;")
            value = QLabel("—")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            metrics_grid.addWidget(title, index // 3, (index % 3) * 2)
            metrics_grid.addWidget(value, index // 3, (index % 3) * 2 + 1)
            self._metrics_labels[key] = value
        layout.addLayout(metrics_grid)

        # Tabbed tables
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._positions_table = self._create_table(
            ["Symbol", "Qty", "Entry", "Current", "Gain", "Change", "Entry Time"]
        )
        self._tabs.addTab(self._positions_table, "Positions")

        self._orders_table = self._create_table(
            ["Time", "Symbol", "Status", "Quantity", "Avg Fill"]
        )
        self._tabs.addTab(self._orders_table, "Recent Orders")

        self._trending_table = self._create_table(
            ["Symbol", "Price", "Premarket %", "Day %", "Reason", "Summary"]
        )
        self._tabs.addTab(self._trending_table, "Trending Symbols")

        self._gainers_table = self._create_table(
            ["Symbol", "Price", "Day %", "Premarket %", "Reason", "Summary"]
        )
        self._tabs.addTab(self._gainers_table, "Top Gainers (Day %)")

    def _create_table(self, headers: List[str]) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        return table

    # ------------------------------------------------------------------ Controller loop

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._controller.run_forever())
        self._loop.run_forever()

    async def _gather_dashboard_state(self) -> Dict[str, object]:
        snapshot = await self._portfolio.snapshot(TradeMode.PAPER)
        orders = list(self._controller.state.orders)[-50:]
        trending = list(self._controller.state.trending)[:25]
        day_gainers = list(getattr(self._controller.state, "day_gainers", []))[:50]
        research = dict(self._controller.state.research)
        return {
            "snapshot": snapshot,
            "orders": orders,
            "trending": trending,
            "day_gainers": day_gainers,
            "research": research,
        }

    def _refresh_dashboard(self) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self._gather_dashboard_state(), self._loop
        )
        try:
            data = future.result(timeout=5)
        except Exception as exc:  # pragma: no cover - UI resilience
            print(f"[dashboard] refresh failed: {exc}")
            return

        snapshot: PortfolioSnapshot = data["snapshot"]  # type: ignore[assignment]
        orders: List[OrderEvent] = data["orders"]  # type: ignore[assignment]
        trending: List[TrendingSymbol] = data["trending"]  # type: ignore[assignment]
        day_gainers: List[TrendingSymbol] = data["day_gainers"]  # type: ignore[assignment]
        research: Dict[str, object] = data["research"]  # type: ignore[assignment]

        self._update_metrics(snapshot, len(orders))
        self._update_positions_table(snapshot)
        self._update_orders_table(orders)
        self._update_trending_table(trending, research)
        self._update_gainers_table(day_gainers or trending, research)

    # ------------------------------------------------------------------ Update helpers

    def _update_metrics(self, snapshot: PortfolioSnapshot, orders_count: int) -> None:
        net_profit = snapshot.realized_pnl + snapshot.unrealized_pnl
        self._metrics_labels["cash"].setText(METRIC_FORMAT.format(snapshot.cash))
        self._metrics_labels["net_profit"].setText(METRIC_FORMAT.format(net_profit))
        self._metrics_labels["unrealized"].setText(METRIC_FORMAT.format(snapshot.unrealized_pnl))
        self._metrics_labels["realized"].setText(METRIC_FORMAT.format(snapshot.realized_pnl))
        self._metrics_labels["net_liquidation"].setText(
            METRIC_FORMAT.format(snapshot.net_liquidation)
        )
        self._metrics_labels["open_positions"].setText(str(len(snapshot.positions)))
        self._metrics_labels["pending_orders"].setText(str(orders_count))

    def _update_positions_table(self, snapshot: PortfolioSnapshot) -> None:
        table = self._positions_table
        table.setRowCount(len(snapshot.positions))
        for row, position in enumerate(snapshot.positions):
            data = [
                position.symbol,
                f"{position.quantity}",
                METRIC_FORMAT.format(position.avg_price),
                METRIC_FORMAT.format(position.current_price),
                METRIC_FORMAT.format(position.gain),
                f"{position.gain_percent:.2f}%",
                position.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            ]
            for col, value in enumerate(data):
                item = QTableWidgetItem(value)
                table.setItem(row, col, item)

    def _update_orders_table(self, orders: List[OrderEvent]) -> None:
        table = self._orders_table
        table.setRowCount(len(orders))
        for row, order in enumerate(reversed(orders)):
            data = [
                order.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                order.symbol,
                order.status,
                f"{order.filled_quantity}",
                METRIC_FORMAT.format(order.avg_fill_price or 0.0),
            ]
            for col, value in enumerate(data):
                item = QTableWidgetItem(value)
                table.setItem(row, col, item)

    def _update_trending_table(self, trending: List[TrendingSymbol], research: Dict[str, object]) -> None:
        table = self._trending_table
        table.setRowCount(len(trending))
        for row, symbol_data in enumerate(trending):
            artifact = research.get(symbol_data.symbol)
            summary = ""
            if isinstance(artifact, ResearchArtifact):
                summary_text = artifact.summary
                summary = summary_text if len(summary_text) <= 120 else summary_text[:117] + "…"
            data = [
                symbol_data.symbol,
                METRIC_FORMAT.format(symbol_data.last_price),
                f"{symbol_data.premarket_change_percent:.2f}%",
                f"{(symbol_data.day_change_percent or 0.0):.2f}%",
                symbol_data.reason or "—",
                summary or "Research limited",
            ]
            for col, value in enumerate(data):
                item = QTableWidgetItem(value)
                table.setItem(row, col, item)

    def _update_gainers_table(self, trending: List[TrendingSymbol], research: Dict[str, object]) -> None:
        # Sort by day change descending
        sorted_syms = sorted(trending, key=lambda s: (s.day_change_percent or 0.0), reverse=True)
        table = self._gainers_table
        table.setRowCount(len(sorted_syms))
        for row, symbol_data in enumerate(sorted_syms):
            artifact = research.get(symbol_data.symbol)
            summary = ""
            if isinstance(artifact, ResearchArtifact):
                summary_text = artifact.summary
                summary = summary_text if len(summary_text) <= 120 else summary_text[:117] + "…"
            data = [
                symbol_data.symbol,
                METRIC_FORMAT.format(symbol_data.last_price),
                f"{(symbol_data.day_change_percent or 0.0):.2f}%",
                f"{symbol_data.premarket_change_percent:.2f}%",
                symbol_data.reason or "—",
                summary or "Research limited",
            ]
            for col, value in enumerate(data):
                item = QTableWidgetItem(value)
                table.setItem(row, col, item)

    # ------------------------------------------------------------------ Shutdown

    def closeEvent(self, event) -> None:  # pragma: no cover - GUI interaction
        self._refresh_timer.stop()
        self._controller.stop()
        try:
            future = asyncio.run_coroutine_threadsafe(self._context.shutdown(), self._loop)
            future.result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)
        super().closeEvent(event)


def main() -> None:
    """Launch the Qt dashboard."""
    settings = get_settings()
    # Ensure fast UI feedback and paper start at $10,000 on launch
    try:
        settings.refresh_interval_seconds = 5  # type: ignore[attr-defined]
        settings.paper_starting_cash = 10_000.0  # type: ignore[attr-defined]
    except Exception:
        pass
    controller, context = create_system(settings)

    app = QApplication(sys.argv)
    window = DashboardWindow(controller=controller, context=context)
    # Prime one immediate cycle so the window isn't empty on first paint
    try:
        future = asyncio.run_coroutine_threadsafe(controller.run_once(), window._loop)  # type: ignore[attr-defined]
        future.result(timeout=10)
        window._refresh_dashboard()
    except Exception:
        pass
    window.show()
    exit_code = app.exec()
    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()


