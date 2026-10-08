"""Tradable instruments and the daily universe snapshot.

An instrument key is `<VENUE>|<venue symbol>`, e.g. `BINANCE_USDM|BTCUSDT` or
`NSE_FO|48704`, so keys are unique across venues and safe to use as partition keys.
"""

from datetime import date
from enum import StrEnum

import msgspec


class Venue(StrEnum):
    BINANCE_SPOT = "BINANCE_SPOT"
    BINANCE_USDM = "BINANCE_USDM"
    NSE = "NSE"


class InstrumentKind(StrEnum):
    INDEX = "index"
    EQUITY = "equity"
    SPOT = "spot"
    PERPETUAL = "perpetual"
    FUTURE = "future"
    CALL = "call"
    PUT = "put"


class Instrument(msgspec.Struct, frozen=True, kw_only=True):
    key: str
    venue: Venue
    symbol: str
    kind: InstrumentKind
    underlying: str | None = None
    expiry: date | None = None
    strike: float | None = None
    lot_size: float = 1.0
    tick_size: float


class Universe(msgspec.Struct, frozen=True, kw_only=True):
    """Everything subscribed on one trading day. Stored with the ticks for lineage."""

    trading_day: date
    instruments: tuple[Instrument, ...]
    source: str
    source_sha256: str

    @property
    def keys(self) -> list[str]:
        return [i.key for i in self.instruments]


def instrument_key(venue: Venue, symbol: str) -> str:
    return f"{venue.value}|{symbol}"
