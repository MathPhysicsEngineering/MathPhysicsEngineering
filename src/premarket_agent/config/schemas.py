"""Shared data schemas and enums."""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class TradeMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class TrendingSymbol(BaseModel):
    symbol: str
    last_price: float
    premarket_change_percent: float
    premarket_volume: int
    reason: Optional[str] = None
    timestamp: dt.datetime = Field(default_factory=dt.datetime.utcnow)


class ResearchArtifact(BaseModel):
    symbol: str
    sentiment: Sentiment
    summary: str
    news_sources: List[str] = Field(default_factory=list)
    analyst_consensus: Optional[str] = None
    analyst_price_target: Optional[float] = None
    earnings_date: Optional[dt.date] = None
    cashflow_strength: Optional[float] = None
    risk_flags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Union[float, str, int, None]] = Field(default_factory=dict)
    generated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)


class TradePlan(BaseModel):
    symbol: str
    entry_price: float
    target_price: float
    stop_loss: float
    position_size: int
    expected_holding_minutes: int
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    mode: TradeMode = TradeMode.PAPER
    metadata: Dict[str, Union[float, str, int, None]] = Field(default_factory=dict)


class OrderEvent(BaseModel):
    symbol: str
    order_id: str
    status: str
    filled_quantity: int
    avg_fill_price: Optional[float] = None
    timestamp: dt.datetime = Field(default_factory=dt.datetime.utcnow)


class PositionState(BaseModel):
    symbol: str
    quantity: int
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float
    entry_time: dt.datetime
    last_update: dt.datetime
    metadata: Dict[str, Union[float, str, int, None]] = Field(default_factory=dict)


class TrainingResult(BaseModel):
    strategy_name: str
    episodes: int
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    annual_return: float
    notes: str = ""


class PortfolioPosition(BaseModel):
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    gain: float
    gain_percent: float


class PortfolioSnapshot(BaseModel):
    cash: float
    realized_pnl: float
    unrealized_pnl: float
    net_liquidation: float
    positions: List[PortfolioPosition] = Field(default_factory=list)
    timestamp: dt.datetime = Field(default_factory=dt.datetime.utcnow)


