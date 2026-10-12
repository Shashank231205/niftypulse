"""Shared builders. Tests construct settings explicitly and never read a developer's .env."""

import itertools

import pytest

from niftypulse.core.config import BinanceSettings, FeatureSettings, FeedSettings, LabelSettings
from niftypulse.domain.events import BookSnapshot, Level, Trade

_ids = itertools.count()
NS = 1_000_000_000


def book(
    ts_s: float,
    bid: float,
    ask: float,
    bid_qty: float = 1.0,
    ask_qty: float = 1.0,
    key: str = "X|BTC",
    levels: int = 5,
) -> BookSnapshot:
    tick = (ask - bid) or 0.1
    return BookSnapshot(
        event_id=f"b{next(_ids)}",
        instrument=key,
        ts_exchange=int(ts_s * NS),
        ts_recv=int(ts_s * NS) + 1,
        seq=next(_ids),
        bids=tuple(Level(bid - i * tick, bid_qty + i) for i in range(levels)),
        asks=tuple(Level(ask + i * tick, ask_qty + i) for i in range(levels)),
    )


def trade(ts_s: float, price: float, qty: float, aggressor: str, key: str = "X|BTC") -> Trade:
    return Trade(
        event_id=f"t{next(_ids)}",
        instrument=key,
        ts_exchange=int(ts_s * NS),
        ts_recv=int(ts_s * NS) + 1,
        trade_id=next(_ids),
        price=price,
        qty=qty,
        aggressor=aggressor,  # type: ignore[arg-type]
    )


@pytest.fixture
def feature_settings() -> FeatureSettings:
    return FeatureSettings(leader_instrument="X|BTC", _env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def label_settings() -> LabelSettings:
    return LabelSettings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def feed_settings() -> FeedSettings:
    return FeedSettings(venues=["BINANCE_SPOT"], publish_depth=5, _env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def binance_settings() -> BinanceSettings:
    return BinanceSettings(
        spot_ws_url="wss://example.invalid",
        spot_rest_url="https://example.invalid",
        usdm_ws_url="wss://example.invalid",
        usdm_rest_url="https://example.invalid",
        spot_symbols=["BTCUSDT"],
        usdm_symbols=["BTCUSDT"],
        _env_file=None,  # type: ignore[call-arg]
    )
