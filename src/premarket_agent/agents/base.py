"""Agent base classes and context containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config.schemas import OrderEvent, ResearchArtifact, TradePlan, TrendingSymbol


@dataclass
class AgentState:
    trending: List[TrendingSymbol] = field(default_factory=list)
    day_gainers: List[TrendingSymbol] = field(default_factory=list)
    research: Dict[str, ResearchArtifact] = field(default_factory=dict)
    plans: List[TradePlan] = field(default_factory=list)
    orders: List[OrderEvent] = field(default_factory=list)


class Agent:
    """Base class for all agents."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def __call__(self, state: AgentState) -> AgentState:
        return await self.run(state)

    async def run(self, state: AgentState) -> AgentState:  # pragma: no cover - interface
        raise NotImplementedError


