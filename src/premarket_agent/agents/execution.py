"""Agent responsible for trade execution."""

from __future__ import annotations

import logging

from .base import Agent, AgentState
from ..services.broker import BrokerService

LOGGER = logging.getLogger(__name__)


class ExecutionAgent(Agent):
    def __init__(self, broker_service: BrokerService) -> None:
        super().__init__("execution")
        self._broker = broker_service

    async def run(self, state: AgentState) -> AgentState:
        if not state.plans:
            LOGGER.info("no trade plans to execute")
            return state
        orders = []
        for plan in state.plans:
            try:
                order = await self._broker.execute(plan)
                orders.append(order)
                LOGGER.info(
                    "executed order %s qty=%s price=%.2f",
                    order.order_id,
                    order.filled_quantity,
                    order.avg_fill_price or plan.entry_price,
                )
            except RuntimeError as exc:
                LOGGER.warning("failed to execute plan for %s: %s", plan.symbol, exc)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.exception("unexpected failure executing plan for %s: %s", plan.symbol, exc)
        state.orders.extend(orders)
        return state


