"""Tradable instruments and the daily universe snapshot."""

from datetime import date
from enum import StrEnum

import msgspec


class InstrumentKind(StrEnum):
    INDEX = "index"
    EQUITY = "equity"
    FUTURE = "future"
    CALL = "call"
    PUT = "put"


class Instrument(msgspec.Struct, frozen=True, kw_only=True):
    key: str
    symbol: str
    kind: InstrumentKind
    underlying: str | None = None
    expiry: date | None = None
    strike: float | None = None
    lot_size: int = 1
    tick_size: float = 0.05


class Universe(msgspec.Struct, frozen=True, kw_only=True):
    """Everything subscribed on one trading day. Stored with the ticks for lineage."""

    trading_day: date
    instruments: tuple[Instrument, ...]
    source_url: str
    source_sha256: str

    @property
    def keys(self) -> list[str]:
        return [i.key for i in self.instruments]

    def by_key(self) -> dict[str, Instrument]:
        return {i.key: i for i in self.instruments}
