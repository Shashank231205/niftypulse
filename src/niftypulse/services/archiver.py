"""Archiver: JetStream market data -> Parquet lake.

Acks only after a batch is durably on disk (see EventBus.subscribe_buffered), so the
lake never silently misses ticks; the redelivery window is sized from the flush interval.
"""

from niftypulse.core.bus import EventBus
from niftypulse.core.config import bus_settings, feed_settings, lake_settings
from niftypulse.marketdata.tick_store import TickWriter
from niftypulse.services.runtime import Service


class Archiver(Service):
    name = "archiver"

    def __init__(self) -> None:
        super().__init__()
        self._lake = lake_settings()
        self._bus = EventBus(bus_settings(), client_name=self.name, on_fatal=self.fail)
        self._writer = TickWriter(self._lake, depth=feed_settings().publish_depth)

    async def start(self) -> None:
        await self._bus.connect()
        self._bus.subscribe_buffered(
            "md.>",
            durable=self.name,
            sink=self._writer,
            ack_wait_s=self._lake.flush_interval_s * self._lake.ack_wait_multiplier,
            max_pending=self._lake.flush_rows * 2,
        )

    async def stop(self) -> None:
        await self._bus.close()


def main() -> None:
    Archiver().run()
