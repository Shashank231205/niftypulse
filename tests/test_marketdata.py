import json

import httpx
import pytest

from niftypulse.domain.events import Alert, BookSnapshot, decode, encode
from niftypulse.marketdata.binance import (
    BinanceFeed,
    BinanceSpotFeed,
    BinanceUsdmFeed,
    _DepthUpdate,
    _SpotRule,
    _SymbolState,
    _UsdmRule,
)
from niftypulse.marketdata.order_book import OrderBook
from tests.conftest import book, trade


def test_order_book_apply_delete_and_top() -> None:
    ob = OrderBook()
    ob.reset([(100.0, 1.0), (99.0, 2.0)], [(101.0, 1.0), (102.0, 3.0)], seq=10)
    ob.apply([(100.0, 0.0), (99.5, 4.0)], [(101.0, 2.0)], seq=11)
    bids, asks = ob.top(2)
    assert [(lv.price, lv.qty) for lv in bids] == [(99.5, 4.0), (99.0, 2.0)]
    assert [(lv.price, lv.qty) for lv in asks] == [(101.0, 2.0), (102.0, 3.0)]
    assert ob.seq == 11 and not ob.is_crossed()
    ob.apply([(101.5, 1.0)], [], seq=12)
    assert ob.is_crossed()


def test_events_roundtrip_all_kinds() -> None:
    for ev in (book(1.0, 100.0, 100.1), trade(1.0, 100.0, 0.5, "buy")):
        assert decode(encode(ev)) == ev


def _depth(
    symbol: str, first: int, final: int, prev: int | None = None, bid: float = 100.0
) -> bytes:
    data = {
        "e": "depthUpdate",
        "E": 1_700_000_000_000,
        "s": symbol,
        "U": first,
        "u": final,
        "b": [[str(bid), "1.0"]],
        "a": [["100.2", "1.0"]],
    }
    if prev is not None:
        data["pu"] = prev
    return json.dumps({"stream": f"{symbol.lower()}@depth@100ms", "data": data}).encode()


def _feed(cls: type[BinanceFeed], feed_settings, binance_settings, snapshot_id: int) -> BinanceFeed:  # type: ignore[no-untyped-def]
    feed = cls(feed_settings, binance_settings)  # type: ignore[call-arg]
    snapshot = {"lastUpdateId": snapshot_id, "bids": [["99.9", "2.0"]], "asks": [["100.3", "2.0"]]}
    feed._http = httpx.AsyncClient(
        base_url="https://example.invalid",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=snapshot)),
    )
    feed._state = {"BTCUSDT": _SymbolState(key=f"{feed.venue.value}|BTCUSDT")}
    return feed


async def _drain(feed) -> list:  # type: ignore[no-untyped-def]
    out = []
    while not feed._queue.empty():
        out.append(feed._queue.get_nowait())
    return out


async def test_spot_sync_drops_stale_bridges_then_detects_gap(
    feed_settings, binance_settings
) -> None:  # type: ignore[no-untyped-def]
    feed = _feed(BinanceSpotFeed, feed_settings, binance_settings, snapshot_id=100)
    await feed._on_message(_depth("BTCUSDT", 90, 95))  # buffered, stale after snapshot
    await feed._on_message(_depth("BTCUSDT", 96, 101))  # bridges 100 + 1
    await feed._load_snapshot("BTCUSDT")
    await feed._on_message(_depth("BTCUSDT", 102, 105))  # continues
    out = await _drain(feed)
    assert [type(e) for e in out] == [BookSnapshot, BookSnapshot]
    assert out[-1].seq == 105
    await feed._on_message(_depth("BTCUSDT", 110, 112))  # gap: 106..109 missing
    alerts = [e for e in await _drain(feed) if isinstance(e, Alert)]
    assert alerts and alerts[0].code == "book_sequence_gap"
    assert not feed._state["BTCUSDT"].synced


async def test_usdm_first_update_after_empty_buffer_must_bridge(
    feed_settings, binance_settings
) -> None:  # type: ignore[no-untyped-def]
    feed = _feed(BinanceUsdmFeed, feed_settings, binance_settings, snapshot_id=500)
    await feed._load_snapshot("BTCUSDT")  # nothing buffered
    await feed._on_message(_depth("BTCUSDT", 490, 499, prev=480))  # stale: dropped silently
    await feed._on_message(_depth("BTCUSDT", 495, 505, prev=489))  # bridges 500
    await feed._on_message(_depth("BTCUSDT", 506, 510, prev=505))  # pu continues
    out = await _drain(feed)
    assert [e.seq for e in out if isinstance(e, BookSnapshot)] == [505, 510]


def test_sequence_rules() -> None:
    ev = _DepthUpdate(1, "X", 101, 110, [], [], 100)
    assert _SpotRule().bridges(ev, 100) and _SpotRule().continues(ev, 100)
    assert _UsdmRule().bridges(_DepthUpdate(1, "X", 95, 105, [], [], 90), 100)
    assert _UsdmRule().continues(ev, 100)
    with pytest.raises(AssertionError):
        assert _SpotRule().continues(ev, 99)
