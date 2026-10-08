"""Resolve the daily subscription universe from the Upstox instrument master.

Contracts roll every month and option strikes follow the market, so nothing is pinned
in config. Each morning this picks the front-month futures, the listed equities and
indices, and the nearest-expiry options within N strikes of the index's last price.
"""

import gzip
import hashlib
from collections.abc import Iterable
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import msgspec

from niftypulse.core import telemetry
from niftypulse.core.config import BrokerSettings, UniverseSettings
from niftypulse.domain.instruments import Instrument, InstrumentKind, Universe

log = telemetry.logger(__name__)

IST = ZoneInfo("Asia/Kolkata")
_PAISE_PER_RUPEE = 100.0


class _Row(msgspec.Struct, kw_only=True):
    instrument_key: str
    trading_symbol: str
    segment: str
    instrument_type: str
    underlying_symbol: str | None = None
    expiry: int | None = None
    strike_price: float | None = None
    lot_size: int = 1
    tick_size: float = 5.0


def _expiry_day(ms: int | None) -> date | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=UTC).astimezone(IST).date()


def _to_instrument(row: _Row, kind: InstrumentKind) -> Instrument:
    return Instrument(
        key=row.instrument_key,
        symbol=row.trading_symbol,
        kind=kind,
        underlying=row.underlying_symbol,
        expiry=_expiry_day(row.expiry),
        strike=row.strike_price or None,
        lot_size=row.lot_size,
        tick_size=row.tick_size / _PAISE_PER_RUPEE,
    )


def front_month_futures(
    rows: Iterable[_Row], underlyings: set[str], today: date
) -> list[Instrument]:
    nearest: dict[str, _Row] = {}
    for r in rows:
        if r.segment != "NSE_FO" or r.instrument_type != "FUT":
            continue
        if r.underlying_symbol not in underlyings:
            continue
        exp = _expiry_day(r.expiry)
        if exp is None or exp < today:
            continue
        best = nearest.get(r.underlying_symbol)
        if best is None or (r.expiry or 0) < (best.expiry or 0):
            nearest[r.underlying_symbol] = r
    missing = underlyings - nearest.keys()
    if missing:
        raise LookupError(f"no live futures contract for {sorted(missing)}")
    return [_to_instrument(r, InstrumentKind.FUTURE) for r in nearest.values()]


def option_strip(
    rows: Iterable[_Row], underlying: str, spot: float, strikes_each_side: int, today: date
) -> list[Instrument]:
    """Calls and puts of the nearest expiry, centred on the strike closest to spot."""
    chain = [
        r
        for r in rows
        if r.segment == "NSE_FO"
        and r.instrument_type in ("CE", "PE")
        and r.underlying_symbol == underlying
        and (_expiry_day(r.expiry) or date.min) >= today
    ]
    if not chain:
        raise LookupError(f"no live options for {underlying}")
    expiry = min(r.expiry or 0 for r in chain)
    chain = [r for r in chain if r.expiry == expiry]
    strikes = sorted({r.strike_price or 0.0 for r in chain})
    atm = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
    selected = set(strikes[max(0, atm - strikes_each_side) : atm + strikes_each_side + 1])
    kinds = {"CE": InstrumentKind.CALL, "PE": InstrumentKind.PUT}
    return [
        _to_instrument(r, kinds[r.instrument_type])
        for r in sorted(chain, key=lambda r: (r.strike_price or 0.0, r.instrument_type))
        if r.strike_price in selected
    ]


def by_symbol(
    rows: Iterable[_Row], segment: str, symbols: set[str], kind: InstrumentKind
) -> list[Instrument]:
    found = {
        r.trading_symbol: r for r in rows if r.segment == segment and r.trading_symbol in symbols
    }
    missing = symbols - found.keys()
    if missing:
        raise LookupError(f"{segment}: unknown symbols {sorted(missing)}")
    return [_to_instrument(found[s], kind) for s in sorted(symbols)]


class InstrumentMaster:
    def __init__(
        self, broker: BrokerSettings, universe: UniverseSettings, access_token: str
    ) -> None:
        self._broker = broker
        self._universe = universe
        self._client = httpx.AsyncClient(
            base_url=broker.api_base_url,
            timeout=broker.http_timeout_s,
            headers={"Accept": "application/json", "Authorization": f"Bearer {access_token}"},
        )

    async def resolve(self, today: date | None = None) -> Universe:
        today = today or datetime.now(IST).date()
        raw = await self._download()
        rows = msgspec.json.decode(gzip.decompress(raw), type=list[_Row], strict=False)
        spot = await self._last_price(self._universe.option_index_key)

        instruments = [
            *by_symbol(rows, "NSE_INDEX", set(self._universe.indices), InstrumentKind.INDEX),
            *by_symbol(rows, "NSE_EQ", set(self._universe.equities), InstrumentKind.EQUITY),
            *front_month_futures(rows, set(self._universe.futures), today),
            *option_strip(
                rows,
                self._universe.option_underlying,
                spot,
                self._universe.option_strikes_each_side,
                today,
            ),
        ]
        universe = Universe(
            trading_day=today,
            instruments=tuple(instruments),
            source_url=self._broker.instrument_master_url,
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )
        log.info("universe.resolved", day=str(today), instruments=len(instruments), spot=spot)
        return universe

    async def _download(self) -> bytes:
        async with httpx.AsyncClient(timeout=self._broker.http_timeout_s * 6) as plain:
            resp = await plain.get(self._broker.instrument_master_url)
            resp.raise_for_status()
            return resp.content

    async def _last_price(self, instrument_key: str) -> float:
        resp = await self._client.get(
            "/v3/market-quote/ltp", params={"instrument_key": instrument_key}
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()["data"]
        if not data:
            raise LookupError(f"no quote returned for {instrument_key}")
        return float(next(iter(data.values()))["last_price"])

    async def close(self) -> None:
        await self._client.aclose()
