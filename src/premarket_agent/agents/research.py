"""Agent that enriches trending symbols with research artifacts."""

from __future__ import annotations

import logging
from typing import Dict

from .base import Agent, AgentState
from ..config.schemas import ResearchArtifact
from ..services.research import ResearchService

LOGGER = logging.getLogger(__name__)


class ResearchAgent(Agent):
    def __init__(self, research_service: ResearchService, max_remote_symbols: int) -> None:
        super().__init__("research")
        self._research = research_service
        self._max_remote_symbols = max_remote_symbols

    async def run(self, state: AgentState) -> AgentState:
        if not state.trending:
            LOGGER.info("no trending symbols; skipping research")
            return state
        research_map: Dict[str, ResearchArtifact] = {}
        remaining_remote = self._max_remote_symbols
        for item in state.trending:
            symbol = item.symbol
            use_remote = remaining_remote > 0 and not (item.reason and "fallback" in item.reason)
            try:
                if use_remote:
                    artifact = await self._research.build_artifact(symbol, trending=item)
                    remaining_remote -= 1
                else:
                    artifact = await self._research.placeholder_artifact(symbol, trending=item)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("research failed for %s: %s", symbol, exc)
                artifact = await self._research.placeholder_artifact(symbol, trending=item)
            research_map[symbol] = artifact
        LOGGER.info("research agent produced artifacts for %s symbols", len(research_map))
        state.research = research_map
        return state


