"""Coordinates agent pipeline execution."""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable, List

from .base import Agent, AgentState
from ..config.settings import Settings

LOGGER = logging.getLogger(__name__)


class AgentController:
    """Runs agents sequentially in a loop."""

    def __init__(self, settings: Settings, agents: Iterable[Agent]) -> None:
        self._settings = settings
        self._agents: List[Agent] = list(agents)
        self._state = AgentState()
        self._running = False

    async def run_once(self) -> AgentState:
        state = self._state
        for agent in self._agents:
            LOGGER.debug("running agent %s", agent.name)
            state = await agent.run(state)
        self._state = state
        return state

    async def run_forever(self) -> None:
        self._running = True
        try:
            while self._running:
                try:
                    await self.run_once()
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.exception("controller iteration failed: %s", exc)
                await asyncio.sleep(self._settings.refresh_interval_seconds)
        except asyncio.CancelledError:
            self._running = False
            raise

    def stop(self) -> None:
        self._running = False

    @property
    def state(self) -> AgentState:
        return self._state


