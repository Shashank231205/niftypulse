"""Market tape and queue-aware order simulation.

`build_tape` compresses one instrument's replayed events into fixed steps: top of book at
the end of each step plus the trade flow during it (volume that hit the bid, lifted the
ask, and the traded price range). The backtest engine and the RL execution environment
both run on the tape, so they share one fill model.

Fill model for a resting limit order (same approximation as the fill-probability labels):
the order joins the back of the queue; trades at its price consume the queue ahead;
displayed size dropping below the queue ahead is treated as cancellations ahead of us;
a trade through the price fills it outright. Market orders fill at the touch.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from niftypulse.domain.events import BookSnapshot, MarketEvent, Trade

_NS = 1_000_000_000

Side = Literal["buy", "sell"]


def build_tape(events: Iterable[MarketEvent], step_ms: int) -> pd.DataFrame:
    step = step_ms * 1_000_000
    rows: list[dict[str, float | int]] = []
    book: BookSnapshot | None = None
    bucket_end = 0
    hit_bid = lift_ask = 0.0
    lo, hi = np.inf, -np.inf

    def flush(ts: int) -> None:
        if book is None or not book.bids or not book.asks:
            return
        rows.append(
            {
                "ts": ts,
                "bid": book.bids[0].price,
                "ask": book.asks[0].price,
                "bid_qty": book.bids[0].qty,
                "ask_qty": book.asks[0].qty,
                "sell_volume": hit_bid,
                "buy_volume": lift_ask,
                "trade_low": lo if np.isfinite(lo) else np.nan,
                "trade_high": hi if np.isfinite(hi) else np.nan,
            }
        )

    for ev in events:
        ts = ev.ts_exchange
        if bucket_end == 0:
            bucket_end = ts - ts % step + step
        while ts >= bucket_end:
            flush(bucket_end)
            bucket_end += step
            hit_bid = lift_ask = 0.0
            lo, hi = np.inf, -np.inf
        if isinstance(ev, Trade):
            if ev.aggressor == "sell":
                hit_bid += ev.qty
            else:
                lift_ask += ev.qty
            lo, hi = min(lo, ev.price), max(hi, ev.price)
        else:
            book = ev
    return pd.DataFrame(rows)


@dataclass(slots=True)
class RestingOrder:
    side: Side
    price: float
    qty: float
    queue_ahead: float
    filled: bool = False


class TapeSimulator:
    """Steps a resting order through the tape; one instance per instrument tape."""

    def __init__(self, tape: pd.DataFrame) -> None:
        self.ts = tape["ts"].to_numpy(dtype=np.int64)
        self.bid = tape["bid"].to_numpy()
        self.ask = tape["ask"].to_numpy()
        self.bid_qty = tape["bid_qty"].to_numpy()
        self.ask_qty = tape["ask_qty"].to_numpy()
        self.sell_volume = tape["sell_volume"].to_numpy()
        self.buy_volume = tape["buy_volume"].to_numpy()
        self.low = tape["trade_low"].to_numpy()
        self.high = tape["trade_high"].to_numpy()

    def __len__(self) -> int:
        return len(self.ts)

    def mid(self, i: int) -> float:
        return float(0.5 * (self.bid[i] + self.ask[i]))

    def join(self, i: int, side: Side, qty: float) -> RestingOrder:
        price = self.bid[i] if side == "buy" else self.ask[i]
        queue = self.bid_qty[i] if side == "buy" else self.ask_qty[i]
        return RestingOrder(side, float(price), qty, float(queue))

    def advance(self, order: RestingOrder, i: int) -> bool:
        """Apply step i's flow to the order. Returns True if it filled during the step."""
        if order.filled:
            return False
        if order.side == "buy":
            through = np.isfinite(self.low[i]) and self.low[i] < order.price
            at_level_flow = self.sell_volume[i] if self.bid[i] <= order.price else 0.0
            shown = self.bid_qty[i] if self.bid[i] == order.price else order.queue_ahead
        else:
            through = np.isfinite(self.high[i]) and self.high[i] > order.price
            at_level_flow = self.buy_volume[i] if self.ask[i] >= order.price else 0.0
            shown = self.ask_qty[i] if self.ask[i] == order.price else order.queue_ahead
        if through or at_level_flow > order.queue_ahead:
            order.filled = True
            return True
        order.queue_ahead = max(0.0, min(order.queue_ahead - at_level_flow, shown))
        return False

    def market_price(self, i: int, side: Side) -> float:
        return float(self.ask[i] if side == "buy" else self.bid[i])


def tape_index(tape_ts: npt.NDArray[np.int64], ts: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Last tape step at or before each timestamp (-1 if none)."""
    return (np.searchsorted(tape_ts, ts, side="right") - 1).astype(np.int64)
