"""Portfolio snapshot utilities for dashboards."""

from __future__ import annotations

from typing import List

from ..config.schemas import PortfolioPosition, PortfolioSnapshot, TradeMode
from .broker import BrokerService
from .market_data import MarketDataService


class PortfolioService:
    """Builds aggregated portfolio views from broker data."""

    def __init__(self, broker: BrokerService, market_data: MarketDataService) -> None:
        self._broker = broker
        self._market_data = market_data

    async def snapshot(self, mode: TradeMode = TradeMode.PAPER) -> PortfolioSnapshot:
        positions = await self._broker.list_positions(mode)
        cash = self._broker.cash_balance(mode)
        realized = self._broker.realized_pnl(mode)

        summary_positions: List[PortfolioPosition] = []
        unrealized_total = 0.0

        for position in positions:
            snapshot = await self._market_data.get_snapshot(position.symbol)
            current_price = snapshot.last_price if snapshot else position.avg_price
            gain = (current_price - position.avg_price) * position.quantity
            gain_percent = (
                (current_price - position.avg_price) / position.avg_price * 100
                if position.avg_price
                else 0.0
            )
            unrealized_total += gain
            summary_positions.append(
                PortfolioPosition(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_price=position.avg_price,
                    current_price=current_price,
                    gain=gain,
                    gain_percent=gain_percent,
                )
            )

        current_value = sum(
            (pos.current_price) * pos.quantity for pos in summary_positions
        )
        net_liq = cash + current_value

        return PortfolioSnapshot(
            cash=cash,
            realized_pnl=realized,
            unrealized_pnl=unrealized_total,
            net_liquidation=net_liq,
            positions=summary_positions,
        )


