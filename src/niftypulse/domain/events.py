"""Events that flow between services.

Every event is immutable and versioned, and has a unique id. The id doubles as the
JetStream de-duplication key, so a retried publish never creates a second copy, and it
lets any prediction be traced back to the exact ticks that produced it.

Timestamps are integer nanoseconds: `ts_exchange` is when NSE stamped the event and
`ts_recv` is when this system first saw it. The gap between them is feed latency.
"""

import time
import uuid
from typing import Literal

import msgspec

SCHEMA_VERSION = 1


def new_id() -> str:
    return uuid.uuid4().hex


def now_ns() -> int:
    return time.time_ns()


class Event(msgspec.Struct, frozen=True, kw_only=True, tag_field="kind"):
    event_id: str = msgspec.field(default_factory=new_id)
    schema_version: int = SCHEMA_VERSION
    instrument: str
    ts_exchange: int
    ts_recv: int


class Level(msgspec.Struct, frozen=True, array_like=True):
    price: float
    qty: int
    orders: int


class BookTick(Event, frozen=True, kw_only=True, tag="book"):
    """Top-of-book snapshot plus the last trade, as delivered by the broker feed."""

    ltp: float
    ltq: int
    volume: int
    oi: int
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]


class FeatureVector(Event, frozen=True, kw_only=True, tag="features"):
    feature_set: str
    names: tuple[str, ...]
    values: tuple[float, ...]
    source_event_id: str


class Prediction(Event, frozen=True, kw_only=True, tag="prediction"):
    horizon_s: int
    p_up: float
    p_flat: float
    p_down: float
    vol_forecast: float
    regime: int
    model_version: str
    source_event_id: str


class Decision(Event, frozen=True, kw_only=True, tag="decision"):
    """A paper-trading decision. The system never sends real orders."""

    action: Literal["buy", "sell", "hold"]
    order_type: Literal["limit", "market", "none"]
    qty: int
    limit_price: float | None
    fill_probability: float | None
    policy_version: str
    source_event_id: str


class Alert(Event, frozen=True, kw_only=True, tag="alert"):
    severity: Literal["info", "warning", "critical"]
    code: str
    message: str


AnyEvent = BookTick | FeatureVector | Prediction | Decision | Alert

_encoder = msgspec.msgpack.Encoder()
_decoder: msgspec.msgpack.Decoder[AnyEvent] = msgspec.msgpack.Decoder(AnyEvent)


def encode(event: Event) -> bytes:
    return _encoder.encode(event)


def decode(data: bytes) -> AnyEvent:
    return _decoder.decode(data)
