"""Simulation and reinforcement learning workflows."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from ..config.schemas import TradeMode, TradePlan, TrainingResult, TrendingSymbol
from ..config.settings import Settings
from .analysis import ScoringService
from .broker import BrokerService
from .learning import LearningService
from .market_data import MarketDataService
from .research import ResearchService

LOGGER = logging.getLogger(__name__)


@dataclass
class SimulationEpisode:
    plans: List[TradePlan]
    pnl: float
    sharpe: float
    max_drawdown: float


class SimulationService:
    """Orchestrates paper trading simulations for training policies."""

    def __init__(
        self,
        settings: Settings,
        market_data: MarketDataService,
        research: ResearchService,
        scoring: ScoringService,
        broker: BrokerService,
        learning: Optional[LearningService] = None,
        portfolio: Optional["PortfolioService"] = None,
    ) -> None:
        self._settings = settings
        self._market_data = market_data
        self._research = research
        self._scoring = scoring
        self._broker = broker
        self._learning = learning
        self._portfolio = portfolio

    async def run_episode(self) -> SimulationEpisode:
        trending = await self._market_data.fetch_trending()
        research_map = {}
        for item in trending:
            research_map[item.symbol] = await self._research.build_artifact(item.symbol)

        ranked = self._scoring.rank_candidates(trending, research_map)
        plans: List[TradePlan] = []
        for rank, (symbol, _) in enumerate(ranked[: self._settings.max_positions]):
            plan = self._scoring.create_trade_plan(
                trending=[ts for ts in trending if ts.symbol == symbol][0],
                artifact=research_map[symbol],
                rank=rank,
                total_candidates=len(ranked),
                mode=TradeMode.PAPER,
            )
            plans.append(plan)

        pnl = 0.0
        equity_curve: List[float] = [0.0]
        trending_map = {item.symbol: item for item in trending}
        for plan in plans:
            await self._broker.execute(plan)
            simulated_exit = plan.entry_price * (1 + random.uniform(-0.02, 0.05))
            await self._broker.close_position(plan.symbol, simulated_exit, plan.mode)
            pnl += (simulated_exit - plan.entry_price) * plan.position_size
            equity_curve.append(pnl)
            if self._learning:
                trending_item = trending_map.get(plan.symbol)
                research_item = research_map.get(plan.symbol)
                if trending_item and research_item:
                    realized = (simulated_exit - plan.entry_price) * plan.position_size
                    self._learning.record_outcome(plan, trending_item, research_item, realized)
        if self._learning:
            self._learning.train()
        sharpe = _sharpe_ratio(equity_curve)
        drawdown = _max_drawdown(equity_curve)
        return SimulationEpisode(plans=plans, pnl=pnl, sharpe=sharpe, max_drawdown=drawdown)

    async def train_policy(self, episodes: Optional[int] = None) -> TrainingResult:
        episodes = episodes or self._settings.rl_training_episodes
        pnls: List[float] = []
        sharpes: List[float] = []
        drawdowns: List[float] = []
        for idx in range(episodes):
            episode = await self.run_episode()
            pnls.append(episode.pnl)
            sharpes.append(episode.sharpe)
            drawdowns.append(episode.max_drawdown)
            LOGGER.info(
                "episode %s/%s pnl=%.2f sharpe=%.2f max_drawdown=%.2f",
                idx + 1,
                episodes,
                episode.pnl,
                episode.sharpe,
                episode.max_drawdown,
            )
        return TrainingResult(
            strategy_name="premarket_rl",
            episodes=episodes,
            sharpe_ratio=float(np.nanmean(sharpes)),
            max_drawdown=float(np.nanmax(drawdowns)),
            win_rate=float(sum(1 for pnl in pnls if pnl > 0) / max(len(pnls), 1)),
            annual_return=float(np.nanmean(pnls) * 252 / self._settings.max_position_size_usd),
            notes="Prototype simulation with random exits. Replace with market replay for production.",
        )


def _sharpe_ratio(equity_curve: Iterable[float]) -> float:
    returns = np.diff(list(equity_curve))
    if returns.size == 0 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))


def _max_drawdown(equity_curve: Iterable[float]) -> float:
    curve = np.array(list(equity_curve))
    if curve.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(curve)
    drawdown = running_max - curve
    return float(np.max(drawdown))


