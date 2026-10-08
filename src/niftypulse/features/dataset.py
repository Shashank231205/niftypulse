"""Build a research dataset by replaying the tick lake through the live feature engine.

Replay merges every instrument's books and trades in exchange-time order, so cross-asset
features see exactly what a live process would have seen at that moment (point-in-time).
Rows are then sampled on a fixed time grid, which removes most of the autocorrelation of
per-update rows, and labelled with strictly forward-looking targets.
"""

import hashlib
import heapq
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

import msgspec
import numpy as np
import pandas as pd
import pyarrow as pa

from niftypulse.core.config import FeatureSettings, LabelSettings
from niftypulse.domain.events import BookSnapshot, Level, MarketEvent, Trade
from niftypulse.features.engine import FeatureEngine, FeatureRow
from niftypulse.features.labels import direction, forward_realized_vol_bps, forward_returns_bps
from niftypulse.marketdata.tick_store import TickReader

_NS = 1_000_000_000
_BPS = 1e4


@dataclass(frozen=True, slots=True)
class Dataset:
    frame: pd.DataFrame
    feature_names: list[str]
    feature_set: str
    dataset_id: str

    def label_col(self, horizon_s: float) -> str:
        return f"label_{_h(horizon_s)}"

    def return_col(self, horizon_s: float) -> str:
        return f"ret_bps_{_h(horizon_s)}"


def _h(horizon_s: float) -> str:
    return f"{horizon_s:g}s"


def _book_events(key: str, table: pa.Table, depth: int) -> Iterator[BookSnapshot]:
    cols = {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}
    for i in range(table.num_rows):
        bids = tuple(
            Level(float(cols[f"bid_px_{j}"][i]), float(cols[f"bid_qty_{j}"][i]))
            for j in range(depth)
            if not np.isnan(cols[f"bid_px_{j}"][i])
        )
        asks = tuple(
            Level(float(cols[f"ask_px_{j}"][i]), float(cols[f"ask_qty_{j}"][i]))
            for j in range(depth)
            if not np.isnan(cols[f"ask_px_{j}"][i])
        )
        yield BookSnapshot(
            event_id=str(cols["event_id"][i]),
            instrument=key,
            ts_exchange=int(cols["ts_exchange"][i]),
            ts_recv=int(cols["ts_recv"][i]),
            seq=int(cols["seq"][i]),
            bids=bids,
            asks=asks,
        )


def _trade_events(key: str, table: pa.Table) -> Iterator[Trade]:
    c = {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}
    for i in range(table.num_rows):
        yield Trade(
            event_id=str(c["event_id"][i]),
            instrument=key,
            ts_exchange=int(c["ts_exchange"][i]),
            ts_recv=int(c["ts_recv"][i]),
            trade_id=int(c["trade_id"][i]),
            price=float(c["price"][i]),
            qty=float(c["qty"][i]),
            aggressor="buy" if c["aggressor"][i] > 0 else "sell",
        )


def replay(
    reader: TickReader, instruments: list[str], start: date, end: date, depth: int
) -> Iterator[MarketEvent]:
    """All market events for the instruments, merged in (exchange time, receipt time) order."""
    streams: list[Iterator[MarketEvent]] = []
    for key in instruments:
        venue, symbol = key.split("|", 1)
        streams.append(_book_events(key, reader.load("book", venue, symbol, start, end), depth))
        streams.append(_trade_events(key, reader.load("trade", venue, symbol, start, end)))
    yield from heapq.merge(*streams, key=lambda e: (e.ts_exchange, e.ts_recv))


def build_dataset(
    reader: TickReader,
    instruments: list[str],
    start: date,
    end: date,
    features: FeatureSettings,
    labels: LabelSettings,
    depth: int,
) -> Dataset:
    engine = FeatureEngine(features, instruments)
    rows: dict[str, list[FeatureRow]] = {k: [] for k in instruments}
    for event in replay(reader, instruments, start, end, depth):
        row = engine.on_event(event)
        if row is not None:
            rows[row.instrument].append(row)

    frames = [
        _label_instrument(key, rows[key], engine.names, labels) for key in instruments if rows[key]
    ]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    frame = frame.sort_values(["ts", "instrument"], kind="stable").reset_index(drop=True)
    manifest = {
        "instruments": instruments,
        "start": str(start),
        "end": str(end),
        "feature_set": engine.feature_set,
        "labels": labels.model_dump(mode="json"),
        "rows": len(frame),
    }
    dataset_id = hashlib.sha256(msgspec.json.encode(manifest)).hexdigest()[:12]
    return Dataset(frame, engine.names, engine.feature_set, dataset_id)


def _label_instrument(
    key: str, rows: list[FeatureRow], names: list[str], s: LabelSettings
) -> pd.DataFrame:
    ts = np.array([r.ts_exchange for r in rows], dtype=np.int64)
    mid = np.array([r.mid for r in rows], dtype=np.float64)
    step = s.sample_interval_ms * 1_000_000
    grid = np.arange(ts[0] - ts[0] % step + step, ts[-1], step, dtype=np.int64)
    idx = np.searchsorted(ts, grid, side="right") - 1
    keep = idx >= 0
    grid, idx = grid[keep], idx[keep]

    values = np.array([rows[i].values for i in idx], dtype=np.float64)
    frame = pd.DataFrame(values, columns=names)
    frame.insert(0, "instrument", key)
    frame.insert(1, "ts", grid)
    frame.insert(2, "mid", mid[idx])
    spread = np.array([rows[i].spread for i in idx])
    frame.insert(3, "spread", spread)
    frame["source_event_id"] = [rows[i].source_event_id for i in idx]

    half_spread_bps = _BPS * 0.5 * spread / mid[idx]
    for h in s.horizons_s:
        ret = forward_returns_bps(ts, mid, grid, h)
        frame[f"ret_bps_{_h(h)}"] = ret
        frame[f"label_{_h(h)}"] = direction(ret, half_spread_bps, s)
    frame[f"fwd_rv_bps_{_h(s.vol_horizon_s)}"] = forward_realized_vol_bps(
        ts, mid, grid, s.vol_horizon_s
    )
    longest = max(*s.horizons_s, s.vol_horizon_s)
    kept: pd.DataFrame = frame[grid + int(longest * _NS) <= ts[-1]]
    return kept.reset_index(drop=True)
