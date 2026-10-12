from datetime import UTC, datetime

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from niftypulse.core.config import GatewaySettings, LakeSettings
from niftypulse.domain.events import InstrumentsResolved
from niftypulse.domain.instruments import Instrument, InstrumentKind, Venue
from niftypulse.marketdata.tick_store import TickReader, TickWriter
from niftypulse.services.gateway.auth import (
    Principal,
    RateLimiter,
    decode_token,
    hash_secret,
    issue_token,
    new_api_key,
    verify_secret,
)
from tests.conftest import book, trade


async def test_tick_store_roundtrip_dedupes_redeliveries(tmp_path) -> None:  # type: ignore[no-untyped-def]
    writer = TickWriter(LakeSettings(root=str(tmp_path), flush_rows=5, _env_file=None), depth=5)  # type: ignore[call-arg]
    b1, b2 = (
        book(1.0, 100.0, 100.1, key="BINANCE_SPOT|BTCUSDT"),
        book(2.0, 100.1, 100.2, key="BINANCE_SPOT|BTCUSDT"),
    )
    t1 = trade(1.5, 100.1, 0.2, "buy", key="BINANCE_SPOT|BTCUSDT")
    for ev in (b1, b2, t1, b1):  # b1 delivered twice (at-least-once)
        writer.add(ev)
    writer.add(
        InstrumentsResolved(
            instrument="BINANCE_SPOT",
            ts_exchange=1,
            ts_recv=1,
            instruments=(
                Instrument(
                    key="BINANCE_SPOT|BTCUSDT",
                    venue=Venue.BINANCE_SPOT,
                    symbol="BTCUSDT",
                    kind=InstrumentKind.SPOT,
                    tick_size=0.01,
                ),
            ),
        )
    )
    assert writer.should_flush()
    await writer.flush()
    reader = TickReader(str(tmp_path))
    day = datetime.fromtimestamp(1, tz=UTC).date()
    books = reader.load("book", "BINANCE_SPOT", "BTCUSDT", day, day)
    trades = reader.load("trade", "BINANCE_SPOT", "BTCUSDT", day, day)
    assert books.num_rows == 2 and trades.num_rows == 1
    assert books.column("bid_px_0").to_pylist() == [100.0, 100.1]
    assert list(tmp_path.glob("universe/*/*/*.json"))


def _gw() -> GatewaySettings:
    return GatewaySettings(
        host="127.0.0.1", port=1, jwt_secret=SecretStr("x" * 48), jwt_ttl_s=60, _env_file=None
    )  # type: ignore[call-arg]


def test_api_keys_are_hashed_and_verified_in_constant_time() -> None:
    key, stored = new_api_key("alice")
    principal, _, secret = key.partition(".")
    assert principal == "alice" and secret not in stored and stored.startswith("scrypt$")
    assert verify_secret(secret, stored)
    assert not verify_secret(secret + "x", stored)
    assert not verify_secret(secret, hash_secret("other"))


def test_jwt_roundtrip_and_tamper_rejection() -> None:
    s = _gw()
    token = issue_token(Principal("alice", "operator"), s)
    assert decode_token(token, s) == Principal("alice", "operator")
    with pytest.raises(HTTPException):
        decode_token(token[:-2] + "aa", s)
    other = GatewaySettings(host="h", port=1, jwt_secret=SecretStr("y" * 48), _env_file=None)  # type: ignore[call-arg]
    with pytest.raises(HTTPException):
        decode_token(token, other)


def test_rate_limiter_blocks_bursts() -> None:
    limiter = RateLimiter(per_minute=5)
    assert all(limiter.allow("k") for _ in range(5))
    assert not limiter.allow("k")
    assert limiter.allow("other")
