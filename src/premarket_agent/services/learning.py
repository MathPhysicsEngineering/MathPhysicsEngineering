"""Learning service coordinating neural policy updates."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np

from ..config.schemas import ResearchArtifact, TradePlan, TrendingSymbol
from ..models import FeatureVectorBuilder, NeuralPolicy

LOGGER = logging.getLogger(__name__)


class LearningService:
    """Collects experiences from virtual trades and trains the neural policy."""

    def __init__(
        self,
        feature_builder: FeatureVectorBuilder,
        policy: Optional[NeuralPolicy] = None,
    ) -> None:
        self._features = feature_builder
        self._policy = policy or NeuralPolicy(feature_builder)

    @property
    def policy(self) -> NeuralPolicy:
        return self._policy

    def score(self, trending: TrendingSymbol, research: Optional[ResearchArtifact]) -> float:
        return self._policy.predict(trending, research)

    def record_outcome(
        self,
        plan: TradePlan,
        trending: TrendingSymbol,
        research: Optional[ResearchArtifact],
        realized_pnl: float,
    ) -> None:
        entry_cost = max(plan.entry_price * plan.position_size, 1.0)
        reward = np.tanh(realized_pnl / entry_cost)
        features = self._features.build(trending, research)
        self._policy.record(features, float(reward))
        LOGGER.debug(
            "recorded experience symbol=%s reward=%.4f pnl=%.2f cost=%.2f",
            plan.symbol,
            reward,
            realized_pnl,
            entry_cost,
        )

    def train(self) -> float:
        loss = self._policy.train()
        if loss:
            LOGGER.info("neural policy training loss=%.5f", loss)
        return loss


