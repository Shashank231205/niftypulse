"""Signal backtest: calibrated probabilities -> positions -> fills -> PnL after costs.

Every `rebalance_s` the convex sizer turns expected returns into target weights across
instruments. Changes in position are executed as market orders at the touch (so the
spread is paid) plus venue fees. PnL is marked to mid each step. Gross and net results
are both reported so the cost of trading is visible, and results are broken down by
regime. Weights are fractions of a fixed notional per instrument.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from niftypulse.backtest.costs import CostModel, Side
from niftypulse.backtest.simulator import TapeSimulator, tape_index
from niftypulse.core.config import PolicySettings
from niftypulse.policy.sizing import ConvexSizer, expected_return_bps, shrunk_covariance
from niftypulse.validation.metrics import block_bootstrap_ci

FloatArray = npt.NDArray[np.float64]


@dataclass
class BacktestResult:
    equity_bps: pd.Series
    gross_equity_bps: pd.Series
    trades: int
    turnover: float
    by_regime: dict[str, float] = field(default_factory=dict)
    per_instrument: dict[str, float] = field(default_factory=dict)

    def summary(
        self, steps_per_year: float, bootstrap_block: int
    ) -> dict[str, float | int | list[float] | dict[str, float]]:
        """Annualised Sharpe comes with a block-bootstrap 95% interval: over a few hours of
        one-second steps the point estimate alone is mostly noise."""
        r = self.equity_bps.diff().fillna(0.0).to_numpy()

        def sharpe_of(x: FloatArray) -> float:
            sd = np.std(x)
            return float(np.mean(x) / sd * np.sqrt(steps_per_year)) if sd > 0 else 0.0

        eq = self.equity_bps.to_numpy()
        index = self.equity_bps.index
        hours = (index[-1] - index[0]).total_seconds() / 3600 if len(index) > 1 else 0.0
        drawdown = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0
        return {
            "net_pnl_bps": float(eq[-1]) if len(eq) else 0.0,
            "gross_pnl_bps": float(self.gross_equity_bps.iloc[-1]) if len(eq) else 0.0,
            "hours": hours,
            "sharpe": sharpe_of(r),
            "sharpe_ci95": list(block_bootstrap_ci(r, sharpe_of, bootstrap_block)),
            "max_drawdown_bps": drawdown,
            "trades": self.trades,
            "turnover": self.turnover,
            "pnl_by_regime_bps": self.by_regime,
            "pnl_by_instrument_bps": self.per_instrument,
        }


def run_signal_backtest(
    predictions: pd.DataFrame,
    tapes: dict[str, TapeSimulator],
    costs: dict[str, CostModel],
    conditional_means: dict[str, tuple[float, float]],
    policy: PolicySettings,
    rebalance_s: float,
) -> BacktestResult:
    """`predictions`: rows with instrument, ts, p_down, p_flat, p_up, regime (test period only).

    `conditional_means[instrument] = (E[r|up], E[r|down])` in bps, estimated on training data.
    """
    instruments = sorted(tapes)
    n = len(instruments)
    grid = np.unique(predictions["ts"].to_numpy(dtype=np.int64))
    step_ns = int(rebalance_s * 1e9)
    sizer = ConvexSizer(policy, n)
    pivot = {
        col: predictions.pivot_table(index="ts", columns="instrument", values=col).reindex(grid)
        for col in ("p_down", "p_flat", "p_up", "regime")
    }
    idx = {k: tape_index(tapes[k].ts, grid) for k in instruments}
    mids = np.array(
        [[tapes[k].mid(i) if i >= 0 else np.nan for i in idx[k]] for k in instruments]
    ).T

    w = np.zeros(n)
    equity = gross = 0.0
    eq_curve, gross_curve = np.zeros(len(grid)), np.zeros(len(grid))
    trades, turnover = 0, 0.0
    regime_pnl: dict[str, float] = {}
    inst_pnl = dict.fromkeys(instruments, 0.0)
    last_rebalance = -np.inf
    returns_window: list[FloatArray] = []

    for t in range(len(grid)):
        if t > 0:
            step_ret = np.nan_to_num(1e4 * np.log(mids[t] / mids[t - 1]))
            pnl = w * step_ret
            equity += pnl.sum()
            gross += pnl.sum()
            returns_window.append(step_ret)
            returns_window = returns_window[-policy.covariance_lookback :]
            for j, k in enumerate(instruments):
                inst_pnl[k] += pnl[j]
                reg = pivot["regime"].iloc[t].get(k, np.nan)
                if np.isfinite(reg):
                    regime_pnl[f"regime{int(reg)}"] = (
                        regime_pnl.get(f"regime{int(reg)}", 0.0) + pnl[j]
                    )

        if grid[t] - last_rebalance >= step_ns and len(returns_window) > 2 * n:
            last_rebalance = grid[t]
            proba = np.column_stack(
                [
                    pivot[c].iloc[t].reindex(instruments).fillna(v).to_numpy()
                    for c, v in (("p_down", 0.0), ("p_flat", 1.0), ("p_up", 0.0))
                ]
            )
            mu = np.array(
                [
                    expected_return_bps(proba[j : j + 1], *conditional_means[k])[0]
                    for j, k in enumerate(instruments)
                ]
            )
            mu[np.abs(proba[:, 2] - proba[:, 0]) < policy.entry_threshold] = 0.0
            cov = shrunk_covariance(np.array(returns_window))
            target = sizer.solve(mu, cov, w)
            delta = target - w
            for j, k in enumerate(instruments):
                if abs(delta[j]) < 1e-6 or idx[k][t] < 0:
                    continue
                i = idx[k][t]
                side: Side = "buy" if delta[j] > 0 else "sell"
                half_spread_bps = 1e4 * (tapes[k].ask[i] - tapes[k].bid[i]) / (2 * tapes[k].mid(i))
                fee_bps = costs[k].fee_bps(side, "taker", notional=1.0)
                cost = abs(delta[j]) * (half_spread_bps + fee_bps)
                equity -= cost
                inst_pnl[k] -= cost
                trades += 1
                turnover += abs(delta[j])
            w = target
        eq_curve[t], gross_curve[t] = equity, gross

    index = pd.to_datetime(grid, unit="ns", utc=True)
    return BacktestResult(
        equity_bps=pd.Series(eq_curve, index=index),
        gross_equity_bps=pd.Series(gross_curve, index=index),
        trades=trades,
        turnover=turnover,
        by_regime=regime_pnl,
        per_instrument=inst_pnl,
    )
