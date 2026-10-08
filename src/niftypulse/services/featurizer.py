"""Featurizer: market data -> feature vectors, using the same engine as research replay.

Values are published as (mid, spread, *features) in the engine's fixed order; consumers
check `feature_set` against their model bundle instead of shipping names on every message.
"""

from niftypulse.core.bus import EventBus, subject
from niftypulse.core.config import binance_settings, bus_settings, feature_settings
from niftypulse.domain.events import AnyEvent, BookSnapshot, FeatureVector, Trade
from niftypulse.domain.instruments import Venue, instrument_key
from niftypulse.features.engine import FeatureEngine
from niftypulse.services.runtime import Service


def live_instruments() -> list[str]:
    bn = binance_settings()
    return [
        *(instrument_key(Venue.BINANCE_SPOT, s) for s in bn.spot_symbols),
        *(instrument_key(Venue.BINANCE_USDM, s) for s in bn.usdm_symbols),
    ]


class Featurizer(Service):
    name = "featurizer"

    def __init__(self) -> None:
        super().__init__()
        self._bus = EventBus(bus_settings(), client_name=self.name, on_fatal=self.fail)
        self._engine = FeatureEngine(feature_settings(), live_instruments())

    async def start(self) -> None:
        await self._bus.connect()
        self._bus.subscribe("md.>", durable=self.name, handler=self._on_event)

    async def _on_event(self, event: AnyEvent) -> None:
        if not isinstance(event, BookSnapshot | Trade):
            return
        row = self._engine.on_event(event)
        if row is None:
            return
        await self._bus.publish(
            subject("feat", row.instrument),
            FeatureVector(
                instrument=row.instrument,
                ts_exchange=row.ts_exchange,
                ts_recv=event.ts_recv,
                feature_set=self._engine.feature_set,
                names=(),
                values=(row.mid, row.spread, *row.values),
                source_event_id=row.source_event_id,
            ),
        )

    async def stop(self) -> None:
        await self._bus.close()


def main() -> None:
    Featurizer().run()
