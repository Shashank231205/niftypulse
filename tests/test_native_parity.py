"""C++ feature engine must match the Python reference on the same events.

Needs the compiled replay_bench; set NIFTYPULSE_REPLAY_BENCH to its path (CI does).
"""

import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from niftypulse.research.native import export_events, python_features
from tests.conftest import book, trade

BENCH = os.environ.get("NIFTYPULSE_REPLAY_BENCH")


@pytest.mark.skipif(not BENCH, reason="NIFTYPULSE_REPLAY_BENCH not set")
def test_cpp_features_match_python(tmp_path: Path, feature_settings) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(42)
    events, mid = [], 100.0
    for i in range(3000):
        t = i * 0.05
        if rng.random() < 0.4:
            events.append(
                trade(t, mid, float(rng.uniform(0.01, 2)), "buy" if rng.random() < 0.5 else "sell")
            )
        else:
            mid += float(rng.normal(0, 0.05))
            events.append(
                book(
                    t,
                    round(mid - 0.05, 2),
                    round(mid + 0.05, 2),
                    float(rng.uniform(0.5, 5)),
                    float(rng.uniform(0.5, 5)),
                    levels=10,
                )
            )
    s = feature_settings
    ev_path, out_path = tmp_path / "e.npev", tmp_path / "f.bin"
    export_events(events, ev_path, depth=10)
    subprocess.run(  # noqa: S603 - test binary
        [
            str(Path(BENCH).resolve()),
            str(ev_path),
            str(out_path),
            ",".join(f"{w:g}" for w in s.windows_s),
            ",".join(f"{w:g}" for w in s.rv_windows_s),
            str(s.ofi_levels),
            f"{s.history_s:g}",
        ],
        check=True,
        capture_output=True,
    )
    raw = out_path.read_bytes()
    rows, cols = struct.unpack("<II", raw[:8])
    cpp = np.frombuffer(raw[8:], dtype="<f8").reshape(rows, cols)
    py, _ = python_features(events, s)
    assert py.shape == cpp.shape
    np.testing.assert_allclose(cpp, py, rtol=1e-9, atol=1e-9)
