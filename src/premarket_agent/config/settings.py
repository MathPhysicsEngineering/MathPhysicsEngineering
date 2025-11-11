"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global configuration for the trading system."""

    app_name: str = "Premarket Agent"
    environment: str = Field(default="development", description="Environment name")

    # Interactive Brokers (IBKR) / TWS / Gateway
    ib_host: str = Field(default="127.0.0.1", description="IBKR host")
    ib_port: int = Field(default=7497, description="IBKR paper trading port")
    ib_client_id: int = Field(default=1, description="IBKR client id")

    # Optional Alpaca (as secondary broker or data source)
    alpaca_api_key: Optional[str] = None
    alpaca_api_secret: Optional[str] = None
    alpaca_paper: bool = True

    # OpenAI / LLM integration for research summarization
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1"

    # Data provider keys (extendable)
    polygon_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None

    # Market data configuration
    trending_symbol_limit: int = Field(default=20, ge=1, le=100)
    min_premarket_gap_percent: float = Field(default=3.0, ge=0.0)
    min_dollar_volume: float = Field(default=1_000_000.0, ge=0.0)
    refresh_interval_seconds: int = Field(default=120, ge=30)
    yahoo_requests_per_second: float = Field(default=0.5, gt=0.0)

    # Research configuration
    news_window_minutes: int = Field(default=180, ge=30)
    filings_lookback_days: int = Field(default=30, ge=1)
    max_remote_research_symbols: int = Field(default=3, ge=0)

    # Risk controls
    max_position_size_usd: float = Field(default=25_000.0, ge=0.0)
    max_positions: int = Field(default=5, ge=1)
    max_drawdown_percent: float = Field(default=5.0, ge=0.0)
    paper_starting_cash: float = Field(default=10_000.0, ge=1.0)

    # Simulation / training
    replay_days: int = Field(default=60, ge=1)
    rl_training_episodes: int = Field(default=500, ge=1)

    # Storage
    duckdb_path: str = Field(default="data/premarket.duckdb")

    # Logging
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _empty_string_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value == "":
            return None
        return value

    @property
    def broker_adapters(self) -> List[str]:
        adapters = ["interactive_brokers"]
        if self.alpaca_api_key and self.alpaca_api_secret:
            adapters.append("alpaca")
        return adapters


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings."""
    return Settings()


