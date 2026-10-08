"""Parquet tick lake: write path (archiver) and read path (research, replay).

Layout, Hive-partitioned so DuckDB prunes by venue, date and instrument:

    <root>/<kind>/venue=<venue>/date=<YYYY-MM-DD>/symbol=<symbol>/part-<first_ts>-<id>.parquet

kind is `book`, `trade` or `universe`. Book levels are flattened into fixed columns
(bid_px_0 .. ask_qty_{N-1}) so feature code reads contiguous float arrays. Delivery is
at-least-once, so the reader de-duplicates on event_id.
"""

import asyncio
import time
import uuid
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import msgspec
import pyarrow as pa
import pyarrow.parquet as pq

from niftypulse.core import telemetry
from niftypulse.core.config import LakeSettings
from niftypulse.domain.events import AnyEvent, BookSnapshot, InstrumentsResolved, Trade

log = telemetry.logger(__name__)

_NS_PER_S = 1_000_000_000


def book_columns(depth: int) -> list[str]:
    cols = []
    for side in ("bid", "ask"):
        for i in range(depth):
            cols += [f"{side}_px_{i}", f"{side}_qty_{i}"]
    return cols


def _partition(event: BookSnapshot | Trade) -> tuple[str, str, date, str]:
    venue, symbol = event.instrument.split("|", 1)
    day = datetime.fromtimestamp(event.ts_exchange / _NS_PER_S, tz=UTC).date()
    return ("book" if isinstance(event, BookSnapshot) else "trade", venue, day, symbol)


class TickWriter:
    """BufferedSink for the archiver: buffers rows per partition, flushes atomically."""

    def __init__(self, settings: LakeSettings, depth: int) -> None:
        self._root = Path(settings.root)
        self._flush_rows = settings.flush_rows
        self._flush_interval = settings.flush_interval_s
        self._depth = depth
        self._rows: dict[tuple[str, str, date, str], list[BookSnapshot | Trade]] = defaultdict(list)
        self._universes: list[InstrumentsResolved] = []
        self._count = 0
        self._last_flush = time.monotonic()

    def add(self, event: AnyEvent) -> None:
        if isinstance(event, BookSnapshot | Trade):
            self._rows[_partition(event)].append(event)
            self._count += 1
        elif isinstance(event, InstrumentsResolved):
            self._universes.append(event)
            self._count += 1

    def should_flush(self) -> bool:
        due = time.monotonic() - self._last_flush >= self._flush_interval
        return self._count >= self._flush_rows or (self._count > 0 and due)

    def discard(self) -> None:
        self._rows.clear()
        self._universes.clear()
        self._count = 0

    async def flush(self) -> None:
        written = 0
        for (kind, venue, day, symbol), events in self._rows.items():
            table = self._book_table(events) if kind == "book" else self._trade_table(events)
            folder = self._root / kind / f"venue={venue}" / f"date={day}" / f"symbol={symbol}"
            # Parquet encoding is CPU and disk bound; keep it off the event loop.
            await asyncio.to_thread(self._write, folder, events[0].ts_exchange, table)
            written += len(events)
        for u in self._universes:
            day = datetime.fromtimestamp(u.ts_recv / _NS_PER_S, tz=UTC).date()
            folder = self._root / "universe" / f"venue={u.instrument}" / f"date={day}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"universe-{u.ts_recv}.json").write_bytes(msgspec.json.encode(u))
        log.info("lake.flushed", rows=written, partitions=len(self._rows))
        self.discard()
        self._last_flush = time.monotonic()

    @staticmethod
    def _write(folder: Path, first_ts: int, table: pa.Table) -> None:
        """Write to a temp name, then rename: readers never see a half-written file."""
        folder.mkdir(parents=True, exist_ok=True)
        name = f"part-{first_ts}-{uuid.uuid4().hex[:8]}.parquet"
        tmp = folder / f".{name}.tmp"
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(folder / name)

    def _book_table(self, events: list[BookSnapshot | Trade]) -> pa.Table:
        books = [e for e in events if isinstance(e, BookSnapshot)]
        cols: dict[str, list[str] | list[int] | list[float]] = {
            "event_id": [e.event_id for e in books],
            "ts_exchange": [e.ts_exchange for e in books],
            "ts_recv": [e.ts_recv for e in books],
            "seq": [e.seq for e in books],
        }
        nan = float("nan")
        for side in ("bid", "ask"):
            for i in range(self._depth):
                px: list[float] = []
                qty: list[float] = []
                for e in books:
                    levels = e.bids if side == "bid" else e.asks
                    if i < len(levels):
                        px.append(levels[i].price)
                        qty.append(levels[i].qty)
                    else:
                        px.append(nan)
                        qty.append(0.0)
                cols[f"{side}_px_{i}"] = px
                cols[f"{side}_qty_{i}"] = qty
        return pa.table(cols)

    @staticmethod
    def _trade_table(events: list[BookSnapshot | Trade]) -> pa.Table:
        trades = [e for e in events if isinstance(e, Trade)]
        return pa.table(
            {
                "event_id": [e.event_id for e in trades],
                "ts_exchange": [e.ts_exchange for e in trades],
                "ts_recv": [e.ts_recv for e in trades],
                "trade_id": [e.trade_id for e in trades],
                "price": [e.price for e in trades],
                "qty": [e.qty for e in trades],
                "aggressor": pa.array(
                    [1 if e.aggressor == "buy" else -1 for e in trades], pa.int8()
                ),
            }
        )


class TickReader:
    """Point-in-time reads from the lake, de-duplicated and ordered by exchange time."""

    def __init__(self, root: str) -> None:
        self._root = Path(root)

    def load(self, kind: str, venue: str, symbol: str, start: date, end: date) -> pa.Table:
        pattern = self._root / kind / f"venue={venue}" / "date=*" / f"symbol={symbol}" / "*.parquet"
        con = duckdb.connect()
        try:
            return con.execute(
                """
                SELECT * EXCLUDE (venue, symbol, date)
                FROM read_parquet(?, hive_partitioning = true)
                WHERE date BETWEEN ? AND ?
                QUALIFY row_number() OVER (PARTITION BY event_id ORDER BY ts_recv) = 1
                ORDER BY ts_exchange, ts_recv
                """,
                [pattern.as_posix(), start, end],
            ).arrow()
        finally:
            con.close()

    def available(self, kind: str) -> list[tuple[str, str, str]]:
        """(venue, date, symbol) partitions present for a kind."""
        out = []
        for folder in sorted((self._root / kind).glob("venue=*/date=*/symbol=*")):
            venue, day, symbol = (p.split("=", 1)[1] for p in folder.parts[-3:])
            out.append((venue, day, symbol))
        return out
