"""Research aggregation service."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Union

import numpy as np
from openai import AsyncOpenAI  # type: ignore

from ..config.schemas import ResearchArtifact, Sentiment, TrendingSymbol
from ..config.settings import Settings
from .news import FundamentalsService, NewsService

LOGGER = logging.getLogger(__name__)


class ResearchService:
    """Aggregates news, fundamentals, and analyst data into a research artifact."""

    POSITIVE_WORDS = {"beat", "strong", "growth", "upgrade", "bullish", "outperform"}
    NEGATIVE_WORDS = {"miss", "weak", "downgrade", "bearish", "underperform", "lawsuit"}

    def __init__(
        self,
        settings: Settings,
        news_service: NewsService,
        fundamentals_service: FundamentalsService,
        summarizer: Optional["Summarizer"] = None,
    ) -> None:
        self._settings = settings
        self._news = news_service
        self._fundamentals = fundamentals_service
        self._summarizer = summarizer or Summarizer(settings.openai_api_key, settings.openai_model)

    async def build_artifact(
        self,
        symbol: str,
        trending: Optional[TrendingSymbol] = None,
    ) -> ResearchArtifact:
        headlines: List[Dict[str, str]] = []
        fundamentals: Dict[str, Optional[Union[float, str]]] = {}
        try:
            headlines = await self._news.fetch_recent(symbol)
        except Exception as exc:  # pragma: no cover - network resilience
            LOGGER.warning("news fetch failed for %s: %s", symbol, exc)
        try:
            fundamentals = await self._fundamentals.fetch_snapshot(symbol)
        except Exception as exc:  # pragma: no cover - network resilience
            LOGGER.warning("fundamentals fetch failed for %s: %s", symbol, exc)
        sentiment = self._infer_sentiment(headlines)
        summary = await self._summarizer.summarize(symbol, headlines, fundamentals, trending=trending)

        analyst_consensus = fundamentals.get("last_upgrade_action")
        price_target = _to_float(fundamentals.get("target_mean_price"))
        cashflow = _to_float(fundamentals.get("cashflow"))
        risk_flags: List[str] = []
        if fundamentals.get("debt_to_equity"):
            dte = _to_float(fundamentals["debt_to_equity"])
            if dte and dte > 200:
                risk_flags.append(f"Elevated debt-to-equity ({dte:.1f})")
        if fundamentals.get("last_upgrade_action") == "downgrade":
            risk_flags.append("Recent analyst downgrade")
        if not headlines:
            risk_flags.append("No recent headlines")

        return ResearchArtifact(
            symbol=symbol,
            sentiment=sentiment,
            summary=summary,
            news_sources=[item.get("link", "") for item in headlines],
            analyst_consensus=analyst_consensus,
            analyst_price_target=price_target,
            earnings_date=_to_str(fundamentals.get("earnings_date")),
            cashflow_strength=cashflow,
            risk_flags=risk_flags,
            metadata=fundamentals,
        )
    async def placeholder_artifact(
        self,
        symbol: str,
        trending: Optional[TrendingSymbol] = None,
    ) -> ResearchArtifact:
        sentiment = Sentiment.NEUTRAL
        change = (
            f"{trending.premarket_change_percent:.2f}%"
            if trending is not None
            else "N/A"
        )
        summary = (
            f"{symbol} flagged by fallback universe with premarket change {change}. "
            "Full research unavailable due to rate limits; proceeding with heuristic scores only."
        )
        return ResearchArtifact(
            symbol=symbol,
            sentiment=sentiment,
            summary=summary,
            news_sources=[],
            risk_flags=["Limited data (fallback universe)"],
            metadata={"change_percent": trending.premarket_change_percent if trending else None},
        )

    def _infer_sentiment(self, headlines: List[Dict[str, str]]) -> Sentiment:
        if not headlines:
            return Sentiment.NEUTRAL
        scores: List[int] = []
        for item in headlines:
            text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
            score = 0
            for word in self.POSITIVE_WORDS:
                if word in text:
                    score += 1
            for word in self.NEGATIVE_WORDS:
                if word in text:
                    score -= 1
            scores.append(score)
        avg = np.mean(scores)
        if avg > 0.5:
            return Sentiment.POSITIVE
        if avg < -0.5:
            return Sentiment.NEGATIVE
        return Sentiment.NEUTRAL


class Summarizer:
    """Optional LLM-based summarizer with graceful fallback."""

    def __init__(self, api_key: Optional[str], model: str) -> None:
        self._client: Optional[AsyncOpenAI]
        if api_key:
            self._client = AsyncOpenAI(api_key=api_key)
        else:
            self._client = None
        self._model = model

    async def summarize(
        self,
        symbol: str,
        headlines: List[Dict[str, str]],
        fundamentals: Dict[str, Optional[Union[float, str]]],
        trending: Optional[TrendingSymbol] = None,
    ) -> str:
        if not self._client:
            return self._heuristic_summary(symbol, headlines, fundamentals, trending)

        prompt = (
            f"You are a trading analyst. Summarize the bullish and bearish factors for {symbol} "
            "based on the following news and fundamentals. Provide a concise 3 sentence summary. "
            "Highlight catalysts, risks, and notable analyst commentary."
        )
        news_lines = [
            f"- {item.get('title', '')} ({item.get('publisher', 'unknown')})"
            for item in headlines[:6]
        ]
        fundamentals_lines = [f"{key}: {value}" for key, value in fundamentals.items() if value]
        content = "\n".join([prompt, "\nNews:\n"] + news_lines + ["\nFundamentals:\n"] + fundamentals_lines)

        try:
            response = await self._client.responses.create(
                model=self._model,
                input=content,
                max_output_tokens=256,
            )
            summary = response.output_text.strip()
            if trending:
                summary += (
                    f" Premarket change: {trending.premarket_change_percent:.2f}% "
                    f"with last price {trending.last_price:.2f}."
                )
            return summary
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("openai summarizer failed; falling back. error=%s", exc)
            return self._heuristic_summary(symbol, headlines, fundamentals, trending)

    def _heuristic_summary(
        self,
        symbol: str,
        headlines: List[Dict[str, str]],
        fundamentals: Dict[str, Optional[Union[float, str]]],
        trending: Optional[TrendingSymbol] = None,
    ) -> str:
        head = headlines[0]["title"] if headlines else "No major headlines."
        target = fundamentals.get("target_mean_price")
        cashflow = fundamentals.get("cashflow")
        change = (
            f"{trending.premarket_change_percent:.2f}%"
            if trending is not None
            else "N/A"
        )
        price = f"{trending.last_price:.2f}" if trending else "N/A"
        return (
            f"{symbol} premarket catalysts include '{head}'. "
            f"Analyst target price: {target or 'N/A'}. "
            f"Operating cashflow: {cashflow or 'N/A'}. "
            f"Premarket change: {change} at price {price}."
        )


def _to_float(value: Optional[Union[float, str]]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Optional[Union[float, str]]) -> Optional[str]:
    if value is None:
        return None
    return str(value)


