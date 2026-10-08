"""Strategy: predictions -> convex sizing -> paper trades.

Paper trading only: decisions are simulated as marketable orders against the latest
recorded book (pay the spread plus venue fees) and stored in `paper_fills`. There is no
code path that sends an order to any venue.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from niftypulse.backtest.costs import Side, cost_model_for
from niftypulse.core import telemetry
from niftypulse.core.bus import EventBus, current_trace_id, subject
from niftypulse.core.config import bus_settings, cost_settings, db_settings, policy_settings
from niftypulse.core.db import Database
from niftypulse.domain.events import AnyEvent, BookSnapshot, Decision, Prediction, now_ns
from niftypulse.domain.instruments import Venue
from niftypulse.policy.sizing import ConvexSizer, expected_return_bps, shrunk_covariance
from niftypulse.research.bundle import load_production
from niftypulse.services.featurizer import live_instruments
from niftypulse.services.runtime import Service
from niftypulse.storage.repositories import FillRepository

log = telemetry.logger(__name__)

_NS = 1_000_000_000


@dataclass
class _Market:
    bid: float = math.nan
    ask: float = math.nan
    last_sample_ns: int = 0
    mids: deque[float] = field(
        default_factory=lambda: deque(maxlen=policy_settings().covariance_lookback)
    )
    proba: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))


class Strategy(Service):
    name = "strategy"

    def __init__(self) -> None:
        super().__init__()
        self._p = policy_settings()
        self._bus = EventBus(bus_settings(), client_name=self.name, on_fatal=self.fail)
        self._db = Database(db_settings())
        self._instruments = live_instruments()
        self._markets = {k: _Market() for k in self._instruments}
        self._weights = np.zeros(len(self._instruments))
        self._sizer = ConvexSizer(self._p, len(self._instruments))
        self._costs = {
            k: cost_model_for(Venue(k.split("|", 1)[0]), cost_settings()) for k in self._instruments
        }
        self._last_rebalance = 0

    async def start(self) -> None:
        self._bundle, _ = await load_production(self._db)
        self._rebalance_ns = int(self._bundle.horizon_s * _NS)
        await self._bus.connect()
        self._bus.subscribe("md.book.>", durable=f"{self.name}-book", handler=self._on_book)
        self._bus.subscribe("pred.>", durable=f"{self.name}-pred", handler=self._on_prediction)

    async def _on_book(self, event: AnyEvent) -> None:
        if not isinstance(event, BookSnapshot) or event.instrument not in self._markets:
            return
        if not event.bids or not event.asks:
            return
        m = self._markets[event.instrument]
        m.bid, m.ask = event.bids[0].price, event.asks[0].price
        if event.ts_exchange - m.last_sample_ns >= _NS:
            m.last_sample_ns = event.ts_exchange
            m.mids.append(0.5 * (m.bid + m.ask))

    async def _on_prediction(self, event: AnyEvent) -> None:
        if not isinstance(event, Prediction) or event.instrument not in self._markets:
            return
        self._markets[event.instrument].proba = np.array([event.p_down, event.p_flat, event.p_up])
        if event.ts_exchange - self._last_rebalance >= self._rebalance_ns:
            self._last_rebalance = event.ts_exchange
            await self._rebalance(event)

    async def _rebalance(self, trigger: Prediction) -> None:
        series = [np.array(m.mids) for m in self._markets.values()]
        n = min(len(s) for s in series)
        if n < 3 * len(series):
            return
        returns = np.column_stack([1e4 * np.diff(np.log(s[-n:])) for s in series])
        mu = np.zeros(len(self._instruments))
        for j, key in enumerate(self._instruments):
            proba = self._markets[key].proba
            if abs(proba[2] - proba[0]) >= self._p.entry_threshold:
                up, down = self._bundle.conditional_means.get(key, (0.0, 0.0))
                mu[j] = expected_return_bps(proba[None, :], up, down)[0]
        target = self._sizer.solve(mu, shrunk_covariance(returns), self._weights)
        for j, key in enumerate(self._instruments):
            delta = float(target[j] - self._weights[j])
            m = self._markets[key]
            if abs(delta) < 1e-4 or math.isnan(m.bid):
                continue
            side: Side = "buy" if delta > 0 else "sell"
            price = m.ask if side == "buy" else m.bid
            notional = abs(delta) * self._p.paper_notional
            decision = Decision(
                instrument=key,
                ts_exchange=trigger.ts_exchange,
                ts_recv=now_ns(),
                action=side,
                order_type="market",
                qty=notional / price,
                limit_price=None,
                fill_probability=None,
                policy_version=f"convex-sizer/{self._bundle.bundle_id}",
                source_event_id=trigger.event_id,
            )
            await self._bus.publish(subject("decision", key), decision)
            async with self._db.session() as s:
                await FillRepository(s).add(
                    {
                        "decision_event_id": decision.event_id,
                        "instrument": key,
                        "side": side,
                        "order_type": "market",
                        "qty": decision.qty,
                        "price": price,
                        "costs": self._costs[key].fee(notional, side, "taker"),
                        "costs_currency": self._costs[key].currency,
                        "policy_version": decision.policy_version,
                        "trace_id": current_trace_id(),
                    }
                )
        self._weights = target

    async def stop(self) -> None:
        await self._bus.close()
        await self._db.close()


def main() -> None:
    Strategy().run()
