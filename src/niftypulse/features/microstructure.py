"""Per-instrument microstructure state and features.

All windows are exponentially time-decayed accumulators: O(1) per event, no buffers,
identical online and offline, and a direct port to the C++ hot path. A window of
`tau` seconds means an event's weight falls to 1/e after `tau` seconds.

References for the order-flow features:
- Cont, Kukanov, Stoikov (2014), "The Price Impact of Order Book Events" (OFI).
- Xu, Gould, Howison (2019), "Multi-Level Order-Flow Imbalance in a Limit Order Book".
- Stoikov (2018), "The micro-price".
"""

import math
from collections import deque
from dataclasses import dataclass, field

from niftypulse.domain.events import BookSnapshot, Level, Trade

_NS = 1_000_000_000
_BPS = 1e4


@dataclass(slots=True)
class DecayedSum:
    """Sum of x_i * exp(-(t - t_i) / tau)."""

    tau_s: float
    value: float = 0.0
    last_ns: int = 0

    def decay_to(self, ts_ns: int) -> float:
        if self.last_ns and ts_ns > self.last_ns:
            self.value *= math.exp(-(ts_ns - self.last_ns) / (self.tau_s * _NS))
        self.last_ns = max(self.last_ns, ts_ns)
        return self.value

    def add(self, x: float, ts_ns: int) -> None:
        self.decay_to(ts_ns)
        self.value += x


def _ofi_level(prev: Level | None, cur: Level | None, is_bid: bool) -> float:
    """Signed order-flow contribution of one book level between two snapshots."""
    if prev is None or cur is None:
        return 0.0
    if is_bid:
        if cur.price > prev.price:
            return cur.qty
        if cur.price == prev.price:
            return cur.qty - prev.qty
        return -prev.qty
    if cur.price < prev.price:
        return -cur.qty
    if cur.price == prev.price:
        return -(cur.qty - prev.qty)
    return prev.qty


def _imbalance(bids: tuple[Level, ...], asks: tuple[Level, ...], depth: int) -> float:
    b = sum(lv.qty for lv in bids[:depth])
    a = sum(lv.qty for lv in asks[:depth])
    return (b - a) / (b + a) if b + a > 0 else 0.0


def _slope(levels: tuple[Level, ...], mid: float) -> float:
    """Least-squares slope of log cumulative depth against log distance from mid.

    Steeper means liquidity builds quickly away from the touch (a resilient book).
    """
    xs, ys, cum = [], [], 0.0
    for lv in levels:
        dist = abs(lv.price - mid) / mid
        cum += lv.qty
        if dist > 0 and cum > 0:
            xs.append(math.log(dist))
            ys.append(math.log(cum))
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / var if var else 0.0


@dataclass(slots=True)
class MicrostructureState:
    """Rolling state for one instrument. Feed events in exchange-time order."""

    windows_s: tuple[float, ...]
    rv_windows_s: tuple[float, ...]
    ofi_levels: int
    history_s: float
    mid: float = math.nan
    spread: float = math.nan
    last_book: BookSnapshot | None = None
    last_trade_ns: int = 0
    mids: deque[tuple[int, float]] = field(default_factory=deque)
    ofi: list[DecayedSum] = field(default_factory=list)
    mlofi: list[DecayedSum] = field(default_factory=list)
    depth_norm: DecayedSum = field(default_factory=lambda: DecayedSum(30.0))
    depth_weight: DecayedSum = field(default_factory=lambda: DecayedSum(30.0))
    signed_vol: list[DecayedSum] = field(default_factory=list)
    total_vol: list[DecayedSum] = field(default_factory=list)
    trade_count: list[DecayedSum] = field(default_factory=list)
    rv: list[DecayedSum] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ofi = [DecayedSum(w) for w in self.windows_s]
        self.mlofi = [DecayedSum(w) for w in self.windows_s]
        self.signed_vol = [DecayedSum(w) for w in self.windows_s]
        self.total_vol = [DecayedSum(w) for w in self.windows_s]
        self.trade_count = [DecayedSum(w) for w in self.windows_s]
        self.rv = [DecayedSum(w) for w in self.rv_windows_s]

    @property
    def ready(self) -> bool:
        return self.last_book is not None and not math.isnan(self.mid)

    def on_trade(self, t: Trade) -> None:
        sign = 1.0 if t.aggressor == "buy" else -1.0
        for sv, tv, tc in zip(self.signed_vol, self.total_vol, self.trade_count, strict=True):
            sv.add(sign * t.qty, t.ts_exchange)
            tv.add(t.qty, t.ts_exchange)
            tc.add(1.0, t.ts_exchange)
        self.last_trade_ns = t.ts_exchange

    def on_book(self, b: BookSnapshot) -> None:
        if not b.bids or not b.asks:
            return
        ts = b.ts_exchange
        bid, ask = b.bids[0], b.asks[0]
        mid = 0.5 * (bid.price + ask.price)
        prev = self.last_book
        if prev is not None and prev.bids and prev.asks:
            l1 = _ofi_level(prev.bids[0], bid, True) + _ofi_level(prev.asks[0], ask, False)
            multi = sum(
                _ofi_level(_at(prev.bids, i), _at(b.bids, i), True)
                + _ofi_level(_at(prev.asks, i), _at(b.asks, i), False)
                for i in range(self.ofi_levels)
            )
            for o, m in zip(self.ofi, self.mlofi, strict=True):
                o.add(l1, ts)
                m.add(multi, ts)
        if not math.isnan(self.mid) and self.mid > 0:
            r = math.log(mid / self.mid)
            for acc in self.rv:
                acc.add(r * r, ts)
        self.depth_norm.add(0.5 * (bid.qty + ask.qty), ts)
        self.depth_weight.add(1.0, ts)
        self.mid, self.spread, self.last_book = mid, ask.price - bid.price, b
        self.mids.append((ts, mid))
        horizon = ts - int(self.history_s * _NS)
        while len(self.mids) > 1 and self.mids[1][0] <= horizon:
            self.mids.popleft()

    def mid_return_bps(self, lookback_s: float, now_ns: int) -> float:
        """Log return of the mid over the lookback, using the last mid at or before it."""
        target = now_ns - int(lookback_s * _NS)
        past = math.nan
        for ts, m in self.mids:
            if ts > target:
                break
            past = m
        if math.isnan(past) or past <= 0 or math.isnan(self.mid):
            return 0.0
        return _BPS * math.log(self.mid / past)

    def features(self, now_ns: int) -> list[float]:
        b = self.last_book
        assert b is not None
        bid, ask = b.bids[0], b.asks[0]
        mid = self.mid
        weight = self.depth_weight.decay_to(now_ns)
        avg_depth = self.depth_norm.decay_to(now_ns) / weight if weight > 0 else 1.0
        avg_depth = avg_depth or 1.0
        micro = (bid.price * ask.qty + ask.price * bid.qty) / (bid.qty + ask.qty)
        out = [
            _BPS * self.spread / mid,
            _BPS * (micro - mid) / mid,
            _imbalance(b.bids, b.asks, 1),
            _imbalance(b.bids, b.asks, 5),
            _imbalance(b.bids, b.asks, 10),
            _slope(b.bids, mid),
            _slope(b.asks, mid),
        ]
        out += [o.decay_to(now_ns) / avg_depth for o in self.ofi]
        out += [m.decay_to(now_ns) / (avg_depth * self.ofi_levels) for m in self.mlofi]
        for sv, tv in zip(self.signed_vol, self.total_vol, strict=True):
            total = tv.decay_to(now_ns)
            out.append(sv.decay_to(now_ns) / total if total > 0 else 0.0)
        out += [tc.decay_to(now_ns) / tc.tau_s for tc in self.trade_count]
        rvs = [_BPS * math.sqrt(max(acc.decay_to(now_ns), 0.0)) for acc in self.rv]
        out += rvs
        out.append(rvs[0] / rvs[-1] if rvs[-1] > 0 else 1.0)
        out += [self.mid_return_bps(w, now_ns) for w in self.windows_s]
        since = (now_ns - self.last_trade_ns) / _NS if self.last_trade_ns else self.history_s
        out.append(min(since, self.history_s))
        return out

    def feature_names(self) -> list[str]:
        w = [f"{int(x)}s" for x in self.windows_s]
        rw = [f"{int(x)}s" for x in self.rv_windows_s]
        return [
            "spread_bps",
            "microprice_dev_bps",
            "imbalance_l1",
            "imbalance_l5",
            "imbalance_l10",
            "book_slope_bid",
            "book_slope_ask",
            *[f"ofi_l1_{x}" for x in w],
            *[f"ofi_ml{self.ofi_levels}_{x}" for x in w],
            *[f"trade_imbalance_{x}" for x in w],
            *[f"trade_rate_{x}" for x in w],
            *[f"realized_vol_bps_{x}" for x in rw],
            "vol_ratio_short_long",
            *[f"mid_return_bps_{x}" for x in w],
            "seconds_since_trade",
        ]


def _at(levels: tuple[Level, ...], i: int) -> Level | None:
    return levels[i] if i < len(levels) else None
