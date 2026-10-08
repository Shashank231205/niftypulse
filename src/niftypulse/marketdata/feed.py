"""Venue-agnostic market data feed (ports-and-adapters).

The recorder depends only on `MarketFeed`. Each venue (Binance spot, Binance USD-M,
NSE via Upstox) is an adapter that resolves its instruments and yields normalised
`BookSnapshot`, `Trade` and `Alert` events.
"""

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from prometheus_client import Counter, Gauge, Histogram

from niftypulse.core.config import FeedSettings
from niftypulse.domain.events import Alert, BookSnapshot, Trade
from niftypulse.domain.instruments import Instrument, Venue

FeedItem = BookSnapshot | Trade | Alert

FEED_MESSAGES = Counter("feed_messages_total", "Normalised feed events", ["venue", "kind"])
FEED_RESYNCS = Counter("feed_resyncs_total", "Order book resyncs", ["venue", "reason"])
FEED_CONNECTED = Gauge("feed_connected", "1 while the venue socket is up", ["venue"])
FEED_LATENCY = Histogram(
    "feed_latency_seconds",
    "Venue timestamp to local receipt",
    ["venue"],
    buckets=(1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 0.1, 0.25, 0.5, 1.0, 2.5),
)


class MarketFeed(ABC):
    venue: Venue

    def __init__(self, settings: FeedSettings) -> None:
        self._feed = settings
        self._queue: asyncio.Queue[FeedItem] = asyncio.Queue(maxsize=settings.queue_maxsize)

    @abstractmethod
    async def resolve_instruments(self) -> list[Instrument]:
        """Look up today's tradable instruments and their tick and lot sizes."""

    @abstractmethod
    async def _run(self) -> None:
        """Connect, keep the connection alive and push events with `_emit`."""

    async def stream(self) -> AsyncIterator[FeedItem]:
        runner = asyncio.create_task(self._run(), name=f"feed-{self.venue}")
        try:
            while True:
                if runner.done():
                    runner.result()
                    return
                get = asyncio.create_task(self._queue.get())
                done, _ = await asyncio.wait({get, runner}, return_when=asyncio.FIRST_COMPLETED)
                if get in done:
                    yield get.result()
                else:
                    get.cancel()
        finally:
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)

    async def close(self) -> None:
        """Release adapter resources (HTTP clients, sessions)."""
        return

    async def _emit(self, item: FeedItem) -> None:
        """Bounded queue: a slow consumer applies backpressure to the socket reader."""
        await self._queue.put(item)
        FEED_MESSAGES.labels(self.venue.value, type(item).__name__).inc()
        if not isinstance(item, Alert):
            FEED_LATENCY.labels(self.venue.value).observe(
                max(item.ts_recv - item.ts_exchange, 0) / 1e9
            )

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter."""
        cap = min(self._feed.reconnect_max_s, self._feed.reconnect_base_s * 2**attempt)
        return random.uniform(0, cap)  # noqa: S311 - jitter, not cryptography
