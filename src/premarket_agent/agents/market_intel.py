"""Agent that continuously scans for trending premarket symbols."""

from __future__ import annotations

import logging

from .base import Agent, AgentState
from ..services.market_data import MarketDataService

LOGGER = logging.getLogger(__name__)


class MarketIntelAgent(Agent):
    def __init__(self, market_data: MarketDataService) -> None:
        super().__init__("market_intel")
        self._market_data = market_data

    async def run(self, state: AgentState) -> AgentState:
        trending, gainers = await asyncio.gather(
            self._market_data.fetch_trending(),
            self._market_data.fetch_day_gainers(),
        )
        LOGGER.info("intel agent identified %s trending symbols", len(trending))
        LOGGER.info("intel agent identified %s day gainers", len(gainers))
        state.trending = trending
        state.day_gainers = gainers
        return state


