"""Scoring, allocation, and risk management services."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..config.schemas import ResearchArtifact, TradeMode, TradePlan, TrendingSymbol
from ..config.settings import Settings
from .learning import LearningService  # type: ignore[import]

LOGGER = logging.getLogger(__name__)


class ScoringService:
    """Converts research artifacts into ranked trade plans."""

    def __init__(
        self,
        settings: Settings,
        risk_manager: "RiskManager",
        learning: Optional[LearningService] = None,
    ) -> None:
        self._settings = settings
        self._risk = risk_manager
        self._learning = learning

    def rank_candidates(
        self, trending: Iterable[TrendingSymbol], research: Dict[str, ResearchArtifact]
    ) -> List[Tuple[str, float]]:
        scores: List[Tuple[str, float]] = []
        for item in trending:
            if item.symbol not in research:
                continue
            artifact = research[item.symbol]
            score, _, _ = self._score_components(item, artifact)
            scores.append((item.symbol, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def create_trade_plan(
        self,
        trending: TrendingSymbol,
        artifact: ResearchArtifact,
        rank: int,
        total_candidates: int,
        mode: TradeMode = TradeMode.PAPER,
    ) -> TradePlan:
        confidence, heuristic, model_score = self._score_components(trending, artifact)
        confidence = min(max(confidence, 0.0), 1.0)
        entry = trending.last_price
        target_change = min(max(trending.premarket_change_percent / 100, 0.03), 0.25)
        target_price = entry * (1 + target_change + confidence * 0.05)
        stop_loss = entry * (1 - max(target_change * 0.6, 0.015))

        capital_allocation = self._risk.allocate_capital(confidence, rank, total_candidates)
        position_size = max(int(capital_allocation // entry), 0)
        position_size = max(1, position_size)

        return TradePlan(
            symbol=trending.symbol,
            entry_price=entry,
            target_price=float(target_price),
            stop_loss=float(stop_loss),
            position_size=position_size,
            expected_holding_minutes=90,
            confidence=confidence,
            rationale=artifact.summary[:512],
            mode=mode,
            metadata={
                "heuristic_score": float(heuristic),
                "neural_score": float(model_score) if model_score is not None else None,
            },
        )

    def _heuristic_score(self, trending: TrendingSymbol, artifact: ResearchArtifact) -> float:
        change_score = np.tanh(trending.premarket_change_percent / 5)
        volume_score = np.tanh(np.log1p(trending.premarket_volume) / 10)
        sentiment_score = {
            "positive": 0.8,
            "neutral": 0.5,
            "negative": 0.2,
        }[artifact.sentiment.value]
        analyst_bonus = 0.1 if artifact.analyst_consensus in {"upgrade", "buy"} else 0.0
        risk_penalty = 0.1 * len(artifact.risk_flags)
        score = change_score * 0.4 + volume_score * 0.3 + sentiment_score * 0.3 + analyst_bonus
        score = max(score - risk_penalty, 0.0)
        return float(min(score, 1.0))

    def _score_components(
        self, trending: TrendingSymbol, artifact: ResearchArtifact
    ) -> Tuple[float, float, Optional[float]]:
        heuristic = self._heuristic_score(trending, artifact)
        model_score: Optional[float] = None
        if self._learning:
            try:
                model_score = self._learning.score(trending, artifact)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("neural policy scoring failed for %s: %s", trending.symbol, exc)
        if model_score is not None:
            combined = 0.6 * heuristic + 0.4 * model_score
        else:
            combined = heuristic
        return float(min(combined, 1.0)), heuristic, model_score


class RiskManager:
    """Risk management heuristics for capital allocation and guardrails."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def max_positions(self) -> int:
        return self._settings.max_positions

    def allocate_capital(self, confidence: float, rank: int, total: int) -> float:
        base = self._settings.max_position_size_usd
        ladder = max(total, 1)
        rank_factor = (ladder - rank) / ladder
        return base * (0.5 + 0.5 * confidence) * (0.5 + 0.5 * rank_factor)

    def validate_plan(self, plan: TradePlan) -> bool:
        if plan.position_size * plan.entry_price > self._settings.max_position_size_usd * 1.5:
            return False
        if plan.stop_loss >= plan.entry_price:
            return False
        if plan.target_price <= plan.entry_price:
            return False
        return True

    def should_exit(self, plan: TradePlan, current_price: float) -> bool:
        if current_price <= plan.stop_loss:
            return True
        if current_price >= plan.target_price:
            return True
        return False


