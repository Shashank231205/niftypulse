"""Binance spot and USD-M perpetual adapters.

Each symbol's local book follows Binance's documented diff-depth procedure:
open the stream, buffer updates, fetch a REST snapshot, drop updates the snapshot
already covers, require the first applied update to bridge the snapshot id, then require
every later update to continue the sequence. Any gap or a crossed book triggers a
resync of that symbol only, and an alert, so the published book is never silently wrong.

Spot and USD-M differ only in the sequencing rule and the trade stream, captured by
`_SequenceRule`.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx
import msgspec
import websockets

from niftypulse.core import telemetry
from niftypulse.core.config import BinanceSettings, FeedSettings
from niftypulse.domain.events import Alert, BookSnapshot, Trade, now_ns
from niftypulse.domain.instruments import Instrument, InstrumentKind, Venue, instrument_key
from niftypulse.marketdata.feed import FEED_CONNECTED, FEED_RESYNCS, MarketFeed
from niftypulse.marketdata.order_book import OrderBook

log = telemetry.logger(__name__)

_MS_TO_NS = 1_000_000


class _DepthUpdate(msgspec.Struct):
    event_time: int = msgspec.field(name="E")
    symbol: str = msgspec.field(name="s")
    first_id: int = msgspec.field(name="U")
    final_id: int = msgspec.field(name="u")
    bids: list[tuple[str, str]] = msgspec.field(name="b")
    asks: list[tuple[str, str]] = msgspec.field(name="a")
    prev_final_id: int | None = msgspec.field(default=None, name="pu")
    transaction_time: int | None = msgspec.field(default=None, name="T")


class _TradeMsg(msgspec.Struct):
    symbol: str = msgspec.field(name="s")
    price: str = msgspec.field(name="p")
    qty: str = msgspec.field(name="q")
    trade_time: int = msgspec.field(name="T")
    buyer_is_maker: bool = msgspec.field(name="m")
    trade_id: int | None = msgspec.field(default=None, name="t")
    agg_trade_id: int | None = msgspec.field(default=None, name="a")


class _Envelope(msgspec.Struct):
    stream: str
    data: msgspec.Raw


_envelope = msgspec.json.Decoder(_Envelope)
_depth = msgspec.json.Decoder(_DepthUpdate)
_trade = msgspec.json.Decoder(_TradeMsg)


class _SequenceRule(ABC):
    @abstractmethod
    def is_stale(self, ev: _DepthUpdate, snapshot_id: int) -> bool: ...

    @abstractmethod
    def bridges(self, ev: _DepthUpdate, snapshot_id: int) -> bool: ...

    @abstractmethod
    def continues(self, ev: _DepthUpdate, last_u: int) -> bool: ...


class _SpotRule(_SequenceRule):
    def is_stale(self, ev: _DepthUpdate, snapshot_id: int) -> bool:
        return ev.final_id <= snapshot_id

    def bridges(self, ev: _DepthUpdate, snapshot_id: int) -> bool:
        return ev.first_id <= snapshot_id + 1 <= ev.final_id

    def continues(self, ev: _DepthUpdate, last_u: int) -> bool:
        return ev.first_id == last_u + 1


class _UsdmRule(_SequenceRule):
    def is_stale(self, ev: _DepthUpdate, snapshot_id: int) -> bool:
        return ev.final_id < snapshot_id

    def bridges(self, ev: _DepthUpdate, snapshot_id: int) -> bool:
        return ev.first_id <= snapshot_id <= ev.final_id

    def continues(self, ev: _DepthUpdate, last_u: int) -> bool:
        return ev.prev_final_id == last_u


@dataclass
class _SymbolState:
    key: str
    book: OrderBook = field(default_factory=OrderBook)
    buffer: list[_DepthUpdate] = field(default_factory=list)
    synced: bool = False
    fetching: bool = False
    # Set after a snapshot load until the first update that bridges it is applied.
    bridge_id: int | None = None


def _levels(raw: list[tuple[str, str]]) -> list[tuple[float, float]]:
    return [(float(p), float(q)) for p, q in raw]


class BinanceFeed(MarketFeed):
    _rule: _SequenceRule
    _kind: InstrumentKind
    _trade_stream: str
    _depth_path: str
    _info_path: str

    def __init__(
        self,
        feed: FeedSettings,
        binance: BinanceSettings,
        ws_url: str,
        rest_url: str,
        symbols: list[str],
    ) -> None:
        super().__init__(feed)
        self._bn = binance
        self._ws_url = ws_url
        self._symbols = [s.upper() for s in symbols]
        self._http = httpx.AsyncClient(base_url=rest_url, timeout=binance.http_timeout_s)
        self._state: dict[str, _SymbolState] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    async def resolve_instruments(self) -> list[Instrument]:
        resp = await self._http.get(self._info_path)
        resp.raise_for_status()
        listed = {s["symbol"]: s for s in resp.json()["symbols"]}
        missing = set(self._symbols) - listed.keys()
        if missing:
            raise LookupError(f"{self.venue}: unknown symbols {sorted(missing)}")
        out = []
        for sym in self._symbols:
            filters = {f["filterType"]: f for f in listed[sym]["filters"]}
            out.append(
                Instrument(
                    key=instrument_key(self.venue, sym),
                    venue=self.venue,
                    symbol=sym,
                    kind=self._kind,
                    underlying=listed[sym]["baseAsset"],
                    tick_size=float(filters["PRICE_FILTER"]["tickSize"]),
                    lot_size=float(filters["LOT_SIZE"]["stepSize"]),
                )
            )
        return out

    def _url(self) -> str:
        streams = []
        for s in self._symbols:
            low = s.lower()
            streams += [
                f"{low}@depth@{self._bn.depth_interval_ms}ms",
                f"{low}@{self._trade_stream}",
            ]
        return f"{self._ws_url}/stream?streams={'/'.join(streams)}"

    async def _run(self) -> None:
        attempt = 0
        while True:
            self._state = {
                s: _SymbolState(key=instrument_key(self.venue, s)) for s in self._symbols
            }
            try:
                async with websockets.connect(
                    self._url(), ping_interval=self._feed.ping_interval_s, max_size=2**22
                ) as ws:
                    FEED_CONNECTED.labels(self.venue.value).set(1)
                    log.info("feed.connected", venue=self.venue.value, symbols=self._symbols)
                    attempt = 0
                    for sym in self._symbols:
                        self._request_snapshot(sym)
                    async for raw in ws:
                        await self._on_message(raw if isinstance(raw, bytes) else raw.encode())
            except (OSError, websockets.WebSocketException, httpx.HTTPError) as exc:
                log.warning("feed.disconnected", venue=self.venue.value, error=repr(exc))
            finally:
                FEED_CONNECTED.labels(self.venue.value).set(0)
                for t in self._tasks:
                    t.cancel()
            delay = self._backoff(attempt)
            attempt += 1
            await asyncio.sleep(delay)

    async def _on_message(self, raw: bytes) -> None:
        recv = now_ns()
        env = _envelope.decode(raw)
        if "@depth" in env.stream:
            await self._on_depth(_depth.decode(env.data), recv)
        else:
            await self._on_trade(_trade.decode(env.data), recv)

    async def _on_trade(self, t: _TradeMsg, recv: int) -> None:
        await self._emit(
            Trade(
                instrument=instrument_key(self.venue, t.symbol),
                ts_exchange=t.trade_time * _MS_TO_NS,
                ts_recv=recv,
                trade_id=t.trade_id if t.trade_id is not None else (t.agg_trade_id or 0),
                price=float(t.price),
                qty=float(t.qty),
                aggressor="sell" if t.buyer_is_maker else "buy",
            )
        )

    async def _on_depth(self, ev: _DepthUpdate, recv: int) -> None:
        st = self._state[ev.symbol]
        if not st.synced:
            st.buffer.append(ev)
            return
        await self._sequence(st, ev, recv)

    async def _sequence(self, st: _SymbolState, ev: _DepthUpdate, recv: int) -> None:
        """Apply one update to a synced book, enforcing Binance's sequencing rules."""
        if st.bridge_id is not None:
            if self._rule.is_stale(ev, st.bridge_id):
                return
            if not self._rule.bridges(ev, st.bridge_id):
                await self._resync(
                    ev.symbol, "bridge_missed", f"snapshot {st.bridge_id}, update {ev.first_id}"
                )
                return
            st.bridge_id = None
        elif not self._rule.continues(ev, st.book.seq):
            await self._resync(
                ev.symbol, "sequence_gap", f"expected after {st.book.seq}, got {ev.first_id}"
            )
            return
        await self._apply(st, ev, recv)

    async def _apply(self, st: _SymbolState, ev: _DepthUpdate, recv: int) -> None:
        st.book.apply(_levels(ev.bids), _levels(ev.asks), ev.final_id)
        if st.book.is_crossed():
            await self._resync(
                ev.symbol, "crossed_book", f"bid {st.book.best_bid} >= ask {st.book.best_ask}"
            )
            return
        bids, asks = st.book.top(self._feed.publish_depth)
        await self._emit(
            BookSnapshot(
                instrument=st.key,
                ts_exchange=(ev.transaction_time or ev.event_time) * _MS_TO_NS,
                ts_recv=recv,
                seq=ev.final_id,
                bids=bids,
                asks=asks,
            )
        )

    def _request_snapshot(self, sym: str) -> None:
        st = self._state[sym]
        if st.fetching:
            return
        st.fetching = True
        task = asyncio.create_task(self._load_snapshot(sym))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _load_snapshot(self, sym: str) -> None:
        st = self._state[sym]
        try:
            resp = await self._http.get(
                self._depth_path, params={"symbol": sym, "limit": self._bn.snapshot_limit}
            )
            resp.raise_for_status()
            snap: dict[str, Any] = resp.json()
        except httpx.HTTPError as exc:
            st.fetching = False
            log.warning("feed.snapshot_failed", venue=self.venue.value, symbol=sym, error=repr(exc))
            await asyncio.sleep(self._backoff(1))
            self._request_snapshot(sym)
            return
        snapshot_id = int(snap["lastUpdateId"])
        st.book.reset(_levels(snap["bids"]), _levels(snap["asks"]), snapshot_id)
        buffered, st.buffer = st.buffer, []
        st.fetching = False
        st.synced = True
        st.bridge_id = snapshot_id
        recv = now_ns()
        for ev in buffered:
            await self._sequence(st, ev, recv)
            if not st.synced:
                return
        log.info("feed.book_synced", venue=self.venue.value, symbol=sym, snapshot_id=snapshot_id)

    async def _resync(self, sym: str, reason: str, detail: str) -> None:
        st = self._state[sym]
        st.synced = False
        st.bridge_id = None
        st.buffer.clear()
        FEED_RESYNCS.labels(self.venue.value, reason).inc()
        log.warning("feed.resync", venue=self.venue.value, symbol=sym, reason=reason, detail=detail)
        await self._emit(
            Alert(
                instrument=st.key,
                ts_exchange=now_ns(),
                ts_recv=now_ns(),
                severity="warning",
                code=f"book_{reason}",
                message=detail,
            )
        )
        self._request_snapshot(sym)

    async def close(self) -> None:
        await self._http.aclose()


class BinanceSpotFeed(BinanceFeed):
    venue = Venue.BINANCE_SPOT
    _rule = _SpotRule()
    _kind = InstrumentKind.SPOT
    _trade_stream = "trade"
    _depth_path = "/api/v3/depth"
    _info_path = "/api/v3/exchangeInfo"

    def __init__(self, feed: FeedSettings, binance: BinanceSettings) -> None:
        super().__init__(
            feed, binance, binance.spot_ws_url, binance.spot_rest_url, binance.spot_symbols
        )


class BinanceUsdmFeed(BinanceFeed):
    venue = Venue.BINANCE_USDM
    _rule = _UsdmRule()
    _kind = InstrumentKind.PERPETUAL
    _trade_stream = "trade"
    _depth_path = "/fapi/v1/depth"
    _info_path = "/fapi/v1/exchangeInfo"

    def __init__(self, feed: FeedSettings, binance: BinanceSettings) -> None:
        super().__init__(
            feed, binance, binance.usdm_ws_url, binance.usdm_rest_url, binance.usdm_symbols
        )
