"""Simple asynchronous rate limiter utilities."""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    """Token-less rate limiter that spaces calls over time."""

    def __init__(self, max_calls_per_sec: float = 1.0) -> None:
        if max_calls_per_sec <= 0:
            raise ValueError("max_calls_per_sec must be positive")
        self._interval = 1.0 / max_calls_per_sec
        self._lock = asyncio.Lock()
        self._next_time = time.monotonic()

    async def wait(self) -> None:
        """Wait until the next slot is available."""
        async with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
            self._next_time = time.monotonic() + self._interval


