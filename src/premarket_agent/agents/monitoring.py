"""Agent that monitors open positions and triggers exits."""

from __future__ import annotations

import logging

from .base import Agent, AgentState
from ..services.analysis import RiskManager
from ..services.broker import BrokerService
from ..services.market_data import MarketDataService

LOGGER = logging.getLogger(__name__)


class MonitoringAgent(Agent):
    def __init__(
        self,
        broker_service: BrokerService,
        market_data: MarketDataService,
        risk_manager: RiskManager,
    ) -> None:
        super().__init__("monitoring")
        self._broker = broker_service
        self._market_data = market_data
        self._risk = risk_manager

    async def run(self, state: AgentState) -> AgentState:
        if not state.plans:
            return state
        for plan in state.plans:
            snapshot = await self._market_data.get_snapshot(plan.symbol)
            if not snapshot:
                continue
            price = snapshot.last_price
            if self._risk.should_exit(plan, price):
                LOGGER.info("monitoring triggers exit for %s at %.2f", plan.symbol, price)
                order = await self._broker.close_position(plan.symbol, price, plan.mode)
                if order:
                    state.orders.append(order)
        return state


