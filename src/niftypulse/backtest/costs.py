"""Transaction cost models (Strategy pattern): one per venue, chosen from the instrument.

Rates are settings, not literals, because exchanges and regulators change them. Defaults
are the published schedules at the time of writing and are documented in .env.example.
"""

from abc import ABC, abstractmethod
from typing import Literal

from niftypulse.core.config import CostSettings
from niftypulse.domain.instruments import Venue

Liquidity = Literal["maker", "taker"]
Side = Literal["buy", "sell"]


class CostModel(ABC):
    currency: str

    @abstractmethod
    def fee(self, notional: float, side: Side, liquidity: Liquidity) -> float:
        """Total explicit cost of one fill, in the instrument's currency."""

    def fee_bps(self, side: Side, liquidity: Liquidity, notional: float) -> float:
        return 1e4 * self.fee(notional, side, liquidity) / notional if notional else 0.0


class BinanceCosts(CostModel):
    currency = "USDT"

    def __init__(self, maker_bps: float, taker_bps: float) -> None:
        self._maker = maker_bps
        self._taker = taker_bps

    def fee(self, notional: float, side: Side, liquidity: Liquidity) -> float:
        rate = self._maker if liquidity == "maker" else self._taker
        return notional * rate / 1e4


class NseFuturesCosts(CostModel):
    """Index/stock futures: brokerage, STT (sell), exchange charges, SEBI fee, GST, stamp duty."""

    currency = "INR"

    def __init__(self, s: CostSettings) -> None:
        self._s = s

    def fee(self, notional: float, side: Side, liquidity: Liquidity) -> float:
        s = self._s
        brokerage = min(s.nse_brokerage_per_order, notional * s.nse_brokerage_cap_pct / 100)
        exchange = notional * s.nse_exchange_txn_pct / 100
        sebi = notional * s.nse_sebi_per_crore / 1e7
        stt = notional * s.nse_stt_sell_pct / 100 if side == "sell" else 0.0
        stamp = notional * s.nse_stamp_buy_pct / 100 if side == "buy" else 0.0
        gst = (brokerage + exchange + sebi) * s.nse_gst_pct / 100
        return brokerage + exchange + sebi + stt + stamp + gst


def cost_model_for(venue: Venue, s: CostSettings) -> CostModel:
    if venue == Venue.BINANCE_SPOT:
        return BinanceCosts(s.binance_spot_maker_bps, s.binance_spot_taker_bps)
    if venue == Venue.BINANCE_USDM:
        return BinanceCosts(s.binance_usdm_maker_bps, s.binance_usdm_taker_bps)
    return NseFuturesCosts(s)
