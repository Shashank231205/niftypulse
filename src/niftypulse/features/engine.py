"""Feature engine: one code path for research replay and the live featurizer.

Feed every market event in exchange-time order; each book update of a ready
instrument yields a feature row. The feature-set id is a hash of the configuration,
so a model is always tied to the exact feature definitions it was trained on.
"""

import hashlib
import math
from dataclasses import dataclass

import msgspec

from niftypulse.core.config import FeatureSettings
from niftypulse.domain.events import BookSnapshot, MarketEvent, Trade
from niftypulse.features.microstructure import MicrostructureState

_FAMILY = "micro"


@dataclass(frozen=True, slots=True)
class FeatureRow:
    instrument: str
    ts_exchange: int
    source_event_id: str
    mid: float
    spread: float
    values: list[float]


def feature_set_id(settings: FeatureSettings) -> str:
    digest = hashlib.sha256(msgspec.json.encode(settings.model_dump(mode="json"))).hexdigest()
    return f"{_FAMILY}-{digest[:10]}"


def default_counterparts(instruments: list[str]) -> dict[str, str]:
    """Pair instruments that trade the same symbol on two venues (spot vs perpetual)."""
    by_symbol: dict[str, list[str]] = {}
    for key in instruments:
        by_symbol.setdefault(key.split("|", 1)[1], []).append(key)
    pairs: dict[str, str] = {}
    for keys in by_symbol.values():
        if len(keys) == 2:
            a, b = keys
            pairs[a], pairs[b] = b, a
    return pairs


class FeatureEngine:
    def __init__(self, settings: FeatureSettings, instruments: list[str]) -> None:
        self._s = settings
        self.feature_set = feature_set_id(settings)
        self._states = {key: self._new_state() for key in instruments}
        self._counterparts = default_counterparts(instruments)
        self._leader = settings.leader_instrument
        if self._leader not in self._states:
            raise ValueError(f"leader {self._leader} is not in the instrument universe")
        self.names = [*self._new_state().feature_names(), *self._cross_names()]

    def _new_state(self) -> MicrostructureState:
        return MicrostructureState(
            windows_s=tuple(self._s.windows_s),
            rv_windows_s=tuple(self._s.rv_windows_s),
            ofi_levels=self._s.ofi_levels,
            history_s=self._s.history_s,
        )

    def _cross_names(self) -> list[str]:
        w = [f"{int(x)}s" for x in self._s.windows_s]
        return [
            *[f"leader_return_bps_{x}" for x in w],
            "basis_bps",
            f"counterpart_return_bps_{w[0]}",
        ]

    def on_event(self, ev: MarketEvent) -> FeatureRow | None:
        state = self._states.get(ev.instrument)
        if state is None:
            return None
        if isinstance(ev, Trade):
            state.on_trade(ev)
            return None
        state.on_book(ev)
        if not state.ready:
            return None
        now = ev.ts_exchange
        values = state.features(now) + self._cross(ev, now)
        return FeatureRow(ev.instrument, now, ev.event_id, state.mid, state.spread, values)

    def _cross(self, ev: BookSnapshot, now: int) -> list[float]:
        leader = self._states[self._leader]
        lead = [leader.mid_return_bps(w, now) if leader.ready else 0.0 for w in self._s.windows_s]
        mine = self._states[ev.instrument]
        other_key = self._counterparts.get(ev.instrument)
        other = self._states.get(other_key) if other_key else None
        if other is not None and other.ready and not math.isnan(other.mid) and other.mid > 0:
            basis = 1e4 * (mine.mid - other.mid) / other.mid
            other_ret = other.mid_return_bps(self._s.windows_s[0], now)
        else:
            basis, other_ret = 0.0, 0.0
        return [*lead, basis, other_ret]
