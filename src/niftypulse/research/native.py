"""Python vs C++ feature engine: parity and speed on real recorded ticks.

Exports one instrument's events to the NPEV1 binary format, runs `replay_bench`, checks
that every C++ feature matches the Python reference (relative tolerance), and times the
Python engine on the same events, so the speed-up is measured, not assumed.
"""

import json
import struct
import subprocess
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from niftypulse.core.config import FeatureSettings
from niftypulse.domain.events import BookSnapshot, MarketEvent, Trade
from niftypulse.features.dataset import replay
from niftypulse.features.microstructure import MicrostructureState
from niftypulse.marketdata.tick_store import TickReader


def export_events(events: list[MarketEvent], path: Path, depth: int) -> None:
    with path.open("wb") as f:
        f.write(b"NPEV1")
        f.write(struct.pack("<I", depth))
        for ev in events:
            if isinstance(ev, BookSnapshot):
                f.write(struct.pack("<BqHH", 0, ev.ts_exchange, len(ev.bids), len(ev.asks)))
                for lv in (*ev.bids, *ev.asks):
                    f.write(struct.pack("<dd", lv.price, lv.qty))
            else:
                agg = 1 if ev.aggressor == "buy" else -1
                f.write(struct.pack("<Bqddb", 1, ev.ts_exchange, ev.price, ev.qty, agg))


def python_features(events: list[MarketEvent], s: FeatureSettings) -> tuple[np.ndarray, list[int]]:
    state = MicrostructureState(
        tuple(s.windows_s), tuple(s.rv_windows_s), s.ofi_levels, s.history_s
    )
    rows: list[list[float]] = []
    times: list[int] = []
    for ev in events:
        if isinstance(ev, Trade):
            state.on_trade(ev)
            continue
        t0 = time.perf_counter_ns()
        state.on_book(ev)
        row = state.features(ev.ts_exchange) if state.ready else None
        times.append(time.perf_counter_ns() - t0)
        if row is not None:
            rows.append(row)
    return np.array(rows), times


def run(
    binary: Path,
    reader: TickReader,
    instrument: str,
    start: date,
    end: date,
    s: FeatureSettings,
    depth: int,
    workdir: Path,
    mlp_weights: Path | None,
) -> dict[str, Any]:
    events = list(replay(reader, [instrument], start, end, depth))
    workdir.mkdir(parents=True, exist_ok=True)
    events_path, features_path = workdir / "events.npev", workdir / "features_cpp.bin"
    export_events(events, events_path, depth)
    argv = [
        str(binary),
        str(events_path),
        str(features_path),
        ",".join(f"{w:g}" for w in s.windows_s),
        ",".join(f"{w:g}" for w in s.rv_windows_s),
        str(s.ofi_levels),
        f"{s.history_s:g}",
    ]
    if mlp_weights is not None:
        argv.append(str(mlp_weights))
    proc = subprocess.run(argv, capture_output=True, text=True, check=True)  # noqa: S603 - our binary
    cpp_stats = json.loads(proc.stdout)

    raw = features_path.read_bytes()
    n_rows, n_cols = struct.unpack("<II", raw[:8])
    cpp = np.frombuffer(raw[8:], dtype="<f8").reshape(n_rows, n_cols)
    py, py_times = python_features(events, s)
    same_shape = py.shape == cpp.shape
    max_rel = (
        float(np.max(np.abs(py - cpp) / np.maximum(np.abs(py), 1e-9)))
        if same_shape
        else float("inf")
    )
    py_p50 = float(np.percentile(py_times, 50))
    return {
        "instrument": instrument,
        "events": len(events),
        "parity": {
            "shape_match": same_shape,
            "max_relative_error": max_rel,
            "passed": same_shape and max_rel < 1e-6,
        },
        "python_feature_update_ns_p50": py_p50,
        "python_feature_update_ns_p99": float(np.percentile(py_times, 99)),
        "cpp": cpp_stats,
        "speedup_p50": py_p50 / cpp_stats["feature_update_ns_p50"]
        if cpp_stats["feature_update_ns_p50"]
        else None,
    }
