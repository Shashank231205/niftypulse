"""Recorder: venue feeds -> JetStream.

Runs one adapter per configured venue. Each normalised event is published on a subject
partitioned by instrument (`md.book.<key>`, `md.trade.<key>`), feed alerts go to
`alert.feed.<key>`, and a watchdog raises a critical alert when a venue goes quiet.
"""

import asyncio
from collections.abc import Callable

from prometheus_client import Gauge

from niftypulse.core import telemetry
from niftypulse.core.bus import EventBus, subject
from niftypulse.core.config import FeedSettings, binance_settings, bus_settings, feed_settings
from niftypulse.domain.events import Alert, BookSnapshot, InstrumentsResolved, now_ns
from niftypulse.domain.instruments import Venue
from niftypulse.marketdata.binance import BinanceSpotFeed, BinanceUsdmFeed
from niftypulse.marketdata.feed import FeedItem, MarketFeed
from niftypulse.services.runtime import Service

log = telemetry.logger(__name__)

SECONDS_SINCE_EVENT = Gauge("recorder_seconds_since_event", "Feed silence", ["venue"])

FeedFactory = Callable[[FeedSettings], MarketFeed]

FEEDS: dict[Venue, FeedFactory] = {
    Venue.BINANCE_SPOT: lambda f: BinanceSpotFeed(f, binance_settings()),
    Venue.BINANCE_USDM: lambda f: BinanceUsdmFeed(f, binance_settings()),
}


def build_feeds(settings: FeedSettings) -> list[MarketFeed]:
    feeds = []
    for name in settings.venues:
        venue = Venue(name)
        if venue not in FEEDS:
            raise ValueError(f"no feed adapter registered for {venue}")
        feeds.append(FEEDS[venue](settings))
    return feeds


class Recorder(Service):
    name = "recorder"

    def __init__(self) -> None:
        super().__init__()
        self._settings = feed_settings()
        self._bus = EventBus(bus_settings(), client_name=self.name, on_fatal=self.fail)
        self._feeds = build_feeds(self._settings)
        self._last_event: dict[Venue, int] = {}

    async def start(self) -> None:
        await self._bus.connect()
        for feed in self._feeds:
            instruments = await feed.resolve_instruments()
            await self._bus.publish(
                f"md.universe.{feed.venue.value}",
                InstrumentsResolved(
                    instrument=feed.venue.value,
                    ts_exchange=now_ns(),
                    ts_recv=now_ns(),
                    instruments=tuple(instruments),
                ),
            )
            self._last_event[feed.venue] = now_ns()
            self.spawn(self._pump(feed), name=f"pump-{feed.venue}")
        self.spawn(self._watchdog(), name="watchdog")

    async def _pump(self, feed: MarketFeed) -> None:
        async for item in feed.stream():
            self._last_event[feed.venue] = now_ns()
            await self._bus.publish(self._subject(item), item)

    @staticmethod
    def _subject(item: FeedItem) -> str:
        if isinstance(item, Alert):
            return subject("alert.feed", item.instrument)
        root = "md.book" if isinstance(item, BookSnapshot) else "md.trade"
        return subject(root, item.instrument)

    async def _watchdog(self) -> None:
        stale_ns = int(self._settings.stale_after_s * 1e9)
        alerted: set[Venue] = set()
        while True:
            await asyncio.sleep(self._settings.stale_after_s / 3)
            now = now_ns()
            for venue, last in self._last_event.items():
                silence = now - last
                SECONDS_SINCE_EVENT.labels(venue.value).set(silence / 1e9)
                if silence > stale_ns and venue not in alerted:
                    alerted.add(venue)
                    log.error("recorder.feed_stale", venue=venue.value, silence_s=silence / 1e9)
                    await self._bus.publish(
                        subject("alert.feed", venue.value),
                        Alert(
                            instrument=venue.value,
                            ts_exchange=now,
                            ts_recv=now,
                            severity="critical",
                            code="feed_stale",
                            message=f"no events for {silence / 1e9:.0f}s",
                        ),
                    )
                elif silence <= stale_ns:
                    alerted.discard(venue)

    async def stop(self) -> None:
        for feed in self._feeds:
            await feed.close()
        await self._bus.close()


def main() -> None:
    Recorder().run()
