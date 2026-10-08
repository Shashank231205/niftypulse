"""Price-level (L2) order book.

Venue adapters own the sequencing rules (when an update may be applied, when to resync);
this class only holds state and answers queries. A C++ implementation with the same
interface replaces it on the hot path (see cpp/).
"""

from collections.abc import Iterable

from sortedcontainers import SortedDict

from niftypulse.domain.events import Level


class OrderBook:
    __slots__ = ("_asks", "_bids", "seq")

    def __init__(self) -> None:
        self._bids: SortedDict[float, float] = SortedDict()
        self._asks: SortedDict[float, float] = SortedDict()
        self.seq = -1

    def reset(
        self, bids: Iterable[tuple[float, float]], asks: Iterable[tuple[float, float]], seq: int
    ) -> None:
        self._bids = SortedDict({p: q for p, q in bids if q > 0})
        self._asks = SortedDict({p: q for p, q in asks if q > 0})
        self.seq = seq

    def apply(
        self, bids: Iterable[tuple[float, float]], asks: Iterable[tuple[float, float]], seq: int
    ) -> None:
        """Apply absolute level quantities; a quantity of zero deletes the level."""
        for side, updates in ((self._bids, bids), (self._asks, asks)):
            for price, qty in updates:
                if qty > 0:
                    side[price] = qty
                else:
                    side.pop(price, None)
        self.seq = seq

    @property
    def best_bid(self) -> float | None:
        return self._bids.peekitem(-1)[0] if self._bids else None

    @property
    def best_ask(self) -> float | None:
        return self._asks.peekitem(0)[0] if self._asks else None

    def is_crossed(self) -> bool:
        bid, ask = self.best_bid, self.best_ask
        return bid is not None and ask is not None and bid >= ask

    def top(self, depth: int) -> tuple[tuple[Level, ...], tuple[Level, ...]]:
        n_bids = min(depth, len(self._bids))
        n_asks = min(depth, len(self._asks))
        bids = tuple(Level(*self._bids.peekitem(-1 - i)) for i in range(n_bids))
        asks = tuple(Level(*self._asks.peekitem(i)) for i in range(n_asks))
        return bids, asks

    def __len__(self) -> int:
        return len(self._bids) + len(self._asks)
