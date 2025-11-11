"""Agent that converts research artifacts into actionable trade plans."""

from __future__ import annotations

import logging

from .base import Agent, AgentState
from ..config.schemas import TradeMode
from ..services.analysis import RiskManager, ScoringService

LOGGER = logging.getLogger(__name__)


class ScoringAgent(Agent):
    def __init__(
        self,
        scoring_service: ScoringService,
        risk_manager: RiskManager,
        mode: TradeMode = TradeMode.PAPER,
    ) -> None:
        super().__init__("scoring")
        self._scoring = scoring_service
        self._risk = risk_manager
        self._mode = mode

    async def run(self, state: AgentState) -> AgentState:
        if not state.trending or not state.research:
            LOGGER.info("insufficient data for scoring; skipping")
            return state
        ranked = self._scoring.rank_candidates(state.trending, state.research)
        plans = []
        for idx, (symbol, _) in enumerate(ranked[: self._risk.max_positions]):
            trending = next(item for item in state.trending if item.symbol == symbol)
            plan = self._scoring.create_trade_plan(
                trending=trending,
                artifact=state.research[symbol],
                rank=idx,
                total_candidates=len(ranked),
                mode=self._mode,
            )
            if not self._risk.validate_plan(plan):
                LOGGER.warning("risk manager rejected plan for %s", plan.symbol)
                continue
            plans.append(plan)
        LOGGER.info("scoring agent produced %s trade plans", len(plans))
        state.plans = plans
        return state


