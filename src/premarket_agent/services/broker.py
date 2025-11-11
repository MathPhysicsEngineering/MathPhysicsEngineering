"""Broker adapters and execution services."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

try:  # pragma: no cover - optional dependency
    from ib_insync import IB, MarketOrder, ScannerSubscription, Stock  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    IB = None  # type: ignore[assignment]
    MarketOrder = None  # type: ignore[assignment]
    Stock = None  # type: ignore[assignment]
    ScannerSubscription = None  # type: ignore[assignment]

from ..config.schemas import OrderEvent, PositionState, TradeMode, TradePlan, TrendingSymbol
from ..config.settings import Settings

LOGGER = logging.getLogger(__name__)


class InteractiveBrokersClient:
    """Thin async wrapper around ib_insync for background threads."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ib = IB() if IB else None
        self._connected = False

    async def connect(self) -> None:
        if self._ib is None:
            raise RuntimeError("ib_insync is not installed; cannot establish IBKR connection")
        if self._connected:
            return
        LOGGER.info(
            "connecting to IBKR host=%s port=%s client_id=%s",
            self._settings.ib_host,
            self._settings.ib_port,
            self._settings.ib_client_id,
        )
        await asyncio.to_thread(
            self._ib.connect,
            self._settings.ib_host,
            self._settings.ib_port,
            clientId=self._settings.ib_client_id,
        )
        self._connected = self._ib.isConnected()
        LOGGER.info("ibkr connected=%s", self._connected)

    async def disconnect(self) -> None:
        if not self._connected or not self._ib:
            return
        await asyncio.to_thread(self._ib.disconnect)
        self._connected = False

    async def fetch_top_movers(self, limit: int = 20) -> List[TrendingSymbol]:
        """Fetch top gainers using IBKR scanners. Fallback to Yahoo if unavailable."""
        if not self._connected or not self._ib:
            LOGGER.debug("IBKR not connected; returning empty movers")
            return []
        if not ScannerSubscription:
            LOGGER.warning("ScannerSubscription unavailable; returning empty movers")
            return []

        sub = ScannerSubscription(
            instrument="STK",
            locationCode="STK.US.MAJOR",
            scanCode="TOP_PERC_GAIN",
        )
        LOGGER.debug("requesting IBKR scanner for top gainers (limit=%s)", limit)
        contracts = await asyncio.to_thread(self._ib.reqScannerData, sub, [], [])
        movers: List[TrendingSymbol] = []
        for contract_details in contracts[:limit]:
            try:
                symbol = contract_details.contract.symbol
                snapshot = await self._snapshot(symbol)
                if snapshot:
                    movers.append(snapshot)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.debug("failed to build IBKR mover snapshot: %s", exc)
        return movers

    async def place_market_order(self, plan: TradePlan) -> OrderEvent:
        if not self._connected or not self._ib:
            raise RuntimeError("IBKR not connected. Call connect() before placing orders.")
        if not Stock or not MarketOrder:
            raise RuntimeError("ib_insync not available; cannot place live orders.")
        contract = Stock(plan.symbol, "SMART", "USD")
        order = MarketOrder("BUY", plan.position_size)
        trade = await asyncio.to_thread(self._ib.placeOrder, contract, order)
        order_id = str(trade.order.orderId)
        LOGGER.info("submitted IBKR order %s for %s", order_id, plan.symbol)
        return OrderEvent(
            symbol=plan.symbol,
            order_id=order_id,
            status=trade.orderStatus.status,
            filled_quantity=int(trade.orderStatus.filled),
            avg_fill_price=trade.orderStatus.avgFillPrice or plan.entry_price,
        )

    async def close(self) -> None:
        await self.disconnect()

    async def _snapshot(self, symbol: str) -> Optional[TrendingSymbol]:
        from yfinance import Ticker

        ticker = await asyncio.to_thread(Ticker, symbol)
        hist = await asyncio.to_thread(ticker.history, "1d", "1m", True)
        if hist.empty:
            return None
        hist = hist.sort_index()
        first = hist.iloc[0]
        last = hist.iloc[-1]
        change = (last["Close"] - first["Open"]) / first["Open"] * 100
        volume = int(hist["Volume"].sum())
        last_price = float(last["Close"])
        return TrendingSymbol(
            symbol=symbol,
            last_price=last_price,
            premarket_change_percent=float(change),
            premarket_volume=volume,
            reason="ibkr_scanner",
            timestamp=hist.index.max().to_pydatetime(),
        )


@dataclass
class PaperPosition:
    symbol: str
    quantity: int
    avg_price: float
    entry_time: pd.Timestamp
    realized_pnl: float = 0.0
    metadata: dict = field(default_factory=dict)


class PaperBroker:
    """In-memory execution simulator."""

    def __init__(self, starting_cash: float = 1_000_000.0) -> None:
        self._starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: Dict[str, PaperPosition] = {}
        self._realized_pnl_total: float = 0.0

    async def execute(self, plan: TradePlan) -> OrderEvent:
        max_affordable = int(self.cash // plan.entry_price)
        if max_affordable <= 0:
            raise RuntimeError(f"insufficient cash for {plan.symbol}")
        quantity = min(plan.position_size, max_affordable)
        if quantity < plan.position_size:
            LOGGER.warning(
                "adjusting position size for %s due to cash limits: requested=%s filled=%s",
                plan.symbol,
                plan.position_size,
                quantity,
            )
        cost = quantity * plan.entry_price
        self.cash -= cost
        now = pd.Timestamp.utcnow()
        position = self.positions.get(plan.symbol)
        if position:
            total_quantity = position.quantity + quantity
            avg_price = (
                position.avg_price * position.quantity + plan.entry_price * quantity
            ) / total_quantity
            position.quantity = total_quantity
            position.avg_price = avg_price
            position.entry_time = now
        else:
            self.positions[plan.symbol] = PaperPosition(
                symbol=plan.symbol,
                quantity=quantity,
                avg_price=plan.entry_price,
                entry_time=now,
                metadata={"confidence": plan.confidence, "rationale": plan.rationale},
            )
        order_id = f"PAPER-{plan.symbol}-{int(now.timestamp())}"
        return OrderEvent(
            symbol=plan.symbol,
            order_id=order_id,
            status="Filled",
            filled_quantity=quantity,
            avg_fill_price=plan.entry_price,
        )

    async def close_position(self, symbol: str, price: float) -> Optional[OrderEvent]:
        position = self.positions.get(symbol)
        if not position:
            return None
        proceeds = position.quantity * price
        pnl = proceeds - position.quantity * position.avg_price
        self.cash += proceeds
        position.realized_pnl += pnl
        self._realized_pnl_total += pnl
        order_id = f"PAPER-EXIT-{symbol}"
        del self.positions[symbol]
        return OrderEvent(
            symbol=symbol,
            order_id=order_id,
            status="Filled",
            filled_quantity=-position.quantity,
            avg_fill_price=price,
        )

    async def list_positions(self) -> List[PositionState]:
        states: List[PositionState] = []
        now = pd.Timestamp.utcnow()
        for position in self.positions.values():
            states.append(
                PositionState(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_price=position.avg_price,
                    unrealized_pnl=0.0,
                    realized_pnl=position.realized_pnl,
                    entry_time=position.entry_time.to_pydatetime(),
                    last_update=now.to_pydatetime(),
                    metadata=position.metadata,
                )
            )
        return states

    def cash_balance(self) -> float:
        return self.cash

    def realized_pnl(self) -> float:
        return self._realized_pnl_total

    def reset(self) -> None:
        self.cash = self._starting_cash
        self.positions.clear()
        self._realized_pnl_total = 0.0


class BrokerService:
    """Facade for routing trade execution between paper and live brokers."""

    def __init__(
        self,
        settings: Settings,
        ib_client: Optional[InteractiveBrokersClient] = None,
        paper_broker: Optional[PaperBroker] = None,
    ) -> None:
        self._settings = settings
        self._ib_client = ib_client
        self._paper = paper_broker or PaperBroker(starting_cash=settings.paper_starting_cash)

    async def execute(self, plan: TradePlan) -> OrderEvent:
        LOGGER.info("executing trade plan for %s mode=%s", plan.symbol, plan.mode.value)
        if plan.mode == TradeMode.LIVE:
            if not self._ib_client:
                raise RuntimeError("Live trading requested but Interactive Brokers not configured")
            await self._ib_client.connect()
            return await self._ib_client.place_market_order(plan)
        return await self._paper.execute(plan)

    async def close_position(self, symbol: str, price: float, mode: TradeMode) -> Optional[OrderEvent]:
        if mode == TradeMode.LIVE:
            LOGGER.warning("live position management not implemented; skipping close for %s", symbol)
            return None
        return await self._paper.close_position(symbol, price)

    async def list_positions(self, mode: TradeMode) -> List[PositionState]:
        if mode == TradeMode.LIVE:
            LOGGER.warning("live positions not implemented; returning empty list")
            return []
        return await self._paper.list_positions()

    async def shutdown(self) -> None:
        if self._ib_client:
            await self._ib_client.close()

    def cash_balance(self, mode: TradeMode) -> float:
        if mode == TradeMode.LIVE:
            LOGGER.warning("live cash balance not implemented; returning 0")
            return 0.0
        return self._paper.cash_balance()

    def realized_pnl(self, mode: TradeMode) -> float:
        if mode == TradeMode.LIVE:
            LOGGER.warning("live realized pnl not implemented; returning 0")
            return 0.0
        return self._paper.realized_pnl()


