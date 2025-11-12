"""Market data ingestion and feature computation."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import Iterable, List, Optional, Sequence

import httpx
import pandas as pd
import yfinance as yf

from ..config.schemas import TrendingSymbol
from ..config.settings import Settings
from ..utils import AsyncRateLimiter
from .broker import InteractiveBrokersClient

LOGGER = logging.getLogger(__name__)

try:  # Python 3.9+
    to_thread = asyncio.to_thread  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - Python 3.8 fallback

    async def to_thread(func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


class MarketDataService:
    """Aggregates premarket data from multiple sources."""

    def __init__(
        self,
        settings: Settings,
        ib_client: Optional[InteractiveBrokersClient] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        rate_limiter: Optional[AsyncRateLimiter] = None,
    ) -> None:
        self._settings = settings
        self._ib_client = ib_client
        self._http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))
        self._limiter = rate_limiter or AsyncRateLimiter(settings.yahoo_requests_per_second)

    async def fetch_trending(self) -> List[TrendingSymbol]:
        """Fetch and merge trending symbols across providers."""
        providers = [
            self._fetch_yahoo_trending(),
            self._fetch_ib_trending() if self._ib_client else asyncio.sleep(0, result=[]),
        ]
        results = await asyncio.gather(*providers, return_exceptions=True)
        symbols: List[TrendingSymbol] = []
        for result in results:
            if isinstance(result, Exception):
                LOGGER.warning("market data provider failed: %s", result)
                continue
            symbols.extend(result)

        merged = self._merge_symbols(symbols)
        filtered = self._apply_filters(merged)
        if not filtered:
            LOGGER.info("no trending symbols found; using fallback universe")
            fallback = await self._fallback_symbols()
            filtered = self._apply_filters(fallback)
            if not filtered:
                filtered = fallback
        LOGGER.info("found %s trending symbols after filters", len(filtered))
        return filtered[: self._settings.trending_symbol_limit]

    async def _fetch_yahoo_trending(self) -> List[TrendingSymbol]:
        url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
        await self._limiter.wait()
        response = await self._http.get(url, params={"count": 50})
        response.raise_for_status()
        payload = response.json()
        quotes = payload.get("finance", {}).get("result", [])
        if not quotes:
            return []
        symbols: List[str] = []
        for entry in quotes:
            symbols.extend([item["symbol"] for item in entry.get("quotes", [])])
        unique = list(dict.fromkeys(symbols))
        tasks = [self._symbol_snapshot(symbol) for symbol in unique]
        snapshots = await asyncio.gather(*tasks, return_exceptions=True)
        results: List[TrendingSymbol] = []
        for snapshot in snapshots:
            if isinstance(snapshot, TrendingSymbol):
                results.append(snapshot)
        return results

    async def _fetch_ib_trending(self) -> List[TrendingSymbol]:
        if not self._ib_client:
            return []
        symbols = await self._ib_client.fetch_top_movers(
            limit=self._settings.trending_symbol_limit * 2
        )
        tasks = [self._symbol_snapshot(symbol.symbol) for symbol in symbols]
        snapshots = await asyncio.gather(*tasks, return_exceptions=True)
        results: List[TrendingSymbol] = []
        for snapshot in snapshots:
            if isinstance(snapshot, TrendingSymbol):
                results.append(snapshot)
        return results

    async def get_snapshot(self, symbol: str) -> Optional[TrendingSymbol]:
        """Obtain a current snapshot for a single symbol."""
        return await self._symbol_snapshot(symbol)

    async def _symbol_snapshot(self, symbol: str) -> Optional[TrendingSymbol]:
        try:
            return await to_thread(self._build_snapshot, symbol)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.debug("failed to build snapshot for %s: %s", symbol, exc)
            return None

    def _build_snapshot(self, symbol: str) -> Optional[TrendingSymbol]:
        ticker = yf.Ticker(symbol)
        fast_info = getattr(ticker, "fast_info", None)

        def _fi_get(*keys: str) -> Optional[float]:
            if not fast_info:
                return None
            for key in keys:
                try:
                    value = fast_info.get(key)
                except Exception:  # pragma: no cover - defensive
                    value = None
                if value not in (None, 0):
                    return value
            return None

        last_price_value = _fi_get(
            "lastPrice",
            "lastClose",
            "regularMarketPreviousClose",
            "previousClose",
            "open",
        )
        last_price = float(last_price_value or 0.0)

        prev_close_value = _fi_get(
            "previousClose",
            "regularMarketPreviousClose",
            "lastClose",
        )
        prev_close = float(prev_close_value or last_price)

        volume_value = _fi_get(
            "lastVolume",
            "tenDayAverageVolume",
            "threeMonthAverageVolume",
            "twoHundredDayAverageVolume",
        )
        volume = int(volume_value or 0)
        # Compute day change percent based on previous close vs current
        day_change = ((last_price - prev_close) / prev_close * 100) if prev_close else 0.0
        timestamp = dt.datetime.utcnow()

        if not last_price or last_price <= 0 or volume <= 0:
            hist = ticker.history(period="5d", interval="5m", prepost=True)
            if hist.empty:
                return None
            hist = hist.sort_index()
            window = hist.tail(60)
            first = window.iloc[0]
            last = window.iloc[-1]
            last_price = float(last["Close"])
            open_price = float(first["Open"]) or last_price
            day_change = (last_price - open_price) / open_price * 100 if open_price else 0.0
            volume = int(window["Volume"].sum())
            timestamp = window.index.max().to_pydatetime()

        return TrendingSymbol(
            symbol=symbol,
            last_price=last_price,
            premarket_change_percent=float(day_change),
            premarket_volume=volume,
            day_change_percent=float(day_change),
            reason="yahoo_trending",
            timestamp=timestamp,
        )

    def _merge_symbols(self, symbols: Sequence[TrendingSymbol]) -> List[TrendingSymbol]:
        merged: dict[str, TrendingSymbol] = {}
        for entry in symbols:
            if entry.symbol not in merged:
                merged[entry.symbol] = entry
                continue
            existing = merged[entry.symbol]
            change = max(existing.premarket_change_percent, entry.premarket_change_percent)
            volume = max(existing.premarket_volume, entry.premarket_volume)
            merged[entry.symbol] = existing.copy(
                update={
                    "premarket_change_percent": change,
                    "premarket_volume": volume,
                    "reason": ", ".join(sorted(set(filter(None, [existing.reason, entry.reason])))),
                    "timestamp": max(existing.timestamp, entry.timestamp),
                }
            )
        return list(merged.values())

    def _apply_filters(self, symbols: Iterable[TrendingSymbol]) -> List[TrendingSymbol]:
        filtered: List[TrendingSymbol] = []
        for entry in symbols:
            if entry.premarket_change_percent < self._settings.min_premarket_gap_percent:
                continue
            if entry.premarket_volume * entry.last_price < self._settings.min_dollar_volume:
                continue
            filtered.append(entry)
        return filtered

    async def close(self) -> None:
        await self._http.aclose()

    async def _fallback_symbols(self) -> List[TrendingSymbol]:
        universe = ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "TSLA", "AMD", "AVGO"]
        tasks = [self._symbol_snapshot(symbol) for symbol in universe]
        snapshots = await asyncio.gather(*tasks, return_exceptions=True)
        results: List[TrendingSymbol] = []
        for snapshot in snapshots:
            if isinstance(snapshot, TrendingSymbol):
                results.append(snapshot.copy(update={"reason": "fallback_universe"}))
        return results


