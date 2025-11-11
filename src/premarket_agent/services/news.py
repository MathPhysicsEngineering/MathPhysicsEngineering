"""News and fundamental data services."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import httpx  # type: ignore[import]

from ..utils import AsyncRateLimiter

LOGGER = logging.getLogger(__name__)


class NewsService:
    """Fetches recent headlines and metadata for a symbol."""

    def __init__(
        self,
        http_client: Optional[httpx.AsyncClient] = None,
        rate_limiter: Optional[AsyncRateLimiter] = None,
    ) -> None:
        self._http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))
        self._limiter = rate_limiter or AsyncRateLimiter(0.5)

    async def fetch_recent(self, symbol: str, limit: int = 10) -> List[Dict[str, str]]:
        url = "https://query2.finance.yahoo.com/v2/finance/news"
        params = {"symbols": symbol, "region": "US", "lang": "en-US"}
        await self._limiter.wait()
        response = await self._http.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("data", [])
        headlines: List[Dict[str, str]] = []
        for item in items[:limit]:
            headline = {
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
            }
            headlines.append(headline)
        return headlines

    async def close(self) -> None:
        await self._http.aclose()


class FundamentalsService:
    """Fetches basic fundamental and analyst data via Yahoo Finance APIs."""

    def __init__(
        self,
        http_client: Optional[httpx.AsyncClient] = None,
        rate_limiter: Optional[AsyncRateLimiter] = None,
    ) -> None:
        self._http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))
        self._limiter = rate_limiter or AsyncRateLimiter(0.5)

    async def fetch_snapshot(self, symbol: str) -> Dict[str, Optional[Union[float, str]]]:
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        modules = [
            "financialData",
            "defaultKeyStatistics",
            "earnings",
            "calendarEvents",
            "recommendationTrend",
            "upgradeDowngradeHistory",
        ]
        params = {"modules": ",".join(modules)}
        await self._limiter.wait()
        response = await self._http.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        result = payload.get("quoteSummary", {}).get("result", [])
        if not result:
            return {}
        data = result[0]
        snapshot: Dict[str, Optional[Union[float, str]]] = {}

        fin_data = data.get("financialData", {})
        snapshot["current_ratio"] = _safe_get(fin_data, "currentRatio", "raw")
        snapshot["cashflow"] = _safe_get(fin_data, "operatingCashflow", "raw")
        snapshot["debt_to_equity"] = _safe_get(fin_data, "debtToEquity", "raw")
        snapshot["target_mean_price"] = _safe_get(fin_data, "targetMeanPrice", "raw")
        snapshot["recommendation_mean"] = _safe_get(fin_data, "recommendationMean", "raw")

        rec_trend = data.get("recommendationTrend", {}).get("trend", [])
        if rec_trend:
            latest = rec_trend[0]
            snapshot["analyst_trend_buy"] = latest.get("buy")
            snapshot["analyst_trend_sell"] = latest.get("sell")

        earnings = data.get("earnings", {})
        if earnings.get("earningsDate"):
            snapshot["earnings_date"] = earnings["earningsDate"][0].get("fmt")

        upgrades = data.get("upgradeDowngradeHistory", {}).get("history", [])
        if upgrades:
            snapshot["last_upgrade_action"] = upgrades[0].get("action")
            snapshot["last_upgrade_grade"] = upgrades[0].get("toGrade")

        cal_events = data.get("calendarEvents", {})
        if cal_events.get("dividendDate"):
            snapshot["dividend_date"] = cal_events["dividendDate"].get("fmt")

        return snapshot

    async def close(self) -> None:
        await self._http.aclose()


def _safe_get(container: Dict, key: str, inner: str) -> Optional[Union[float, str]]:
    if key not in container:
        return None
    value = container[key]
    if isinstance(value, dict):
        return value.get(inner)
    return None


