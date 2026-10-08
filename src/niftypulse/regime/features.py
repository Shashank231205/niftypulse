"""Causal regime features appended to the base feature set.

For each instrument, in time order: Kalman drift and innovation, BOCPD change
probability and run length, and HMM filtered state probabilities. The HMM is fitted on
the training slice only and then filtered forward over every row, so test rows never
influence their own features. `OnlineRegimeState` is shared by research replay and the
live predictor, which guarantees the two compute identical values.
"""

import numpy as np
import pandas as pd

from niftypulse.core.config import RegimeSettings
from niftypulse.regime.changepoint import BOCPD
from niftypulse.regime.hmm import OnlineRegimeFilter, RegimeHMM, regime_observations
from niftypulse.regime.kalman import LocalLinearTrend

VOL_COL = "realized_vol_bps_30s"
RET_COL = "mid_return_bps_30s"
SPREAD_COL = "spread_bps"
STEP_RET_COL = "mid_return_bps_1s"


def regime_columns(n_states: int) -> list[str]:
    return [
        "kalman_velocity_bps_s",
        "kalman_innovation_z",
        "changepoint_prob",
        "run_length",
        *[f"regime_p{k}" for k in range(n_states)],
        "regime",
    ]


class OnlineRegimeState:
    """Regime features for one instrument, updated once per sampled row."""

    def __init__(self, hmm: RegimeHMM, s: RegimeSettings) -> None:
        self._kf = LocalLinearTrend(s.kalman_level_var, s.kalman_velocity_var, s.kalman_obs_var)
        self._cpd = BOCPD(s.bocpd_hazard_lambda, s.bocpd_max_run, s.bocpd_recent_window)
        self._filter = OnlineRegimeFilter(hmm)

    def update(
        self,
        ts: int,
        mid: float,
        step_ret_bps: float,
        vol_bps: float,
        ret_bps: float,
        spread_bps: float,
    ) -> list[float]:
        self._kf.update(float(np.log(mid)), ts)
        cp, run = self._cpd.update(step_ret_bps)
        obs = regime_observations(np.array([vol_bps]), np.array([ret_bps]), np.array([spread_bps]))
        probs = self._filter.update(obs[0])
        return [
            self._kf.velocity_bps_per_s,
            self._kf.innovation_z,
            cp,
            run,
            *probs.tolist(),
            float(np.argmax(probs)),
        ]


def fit_regime_model(train: pd.DataFrame, s: RegimeSettings) -> RegimeHMM:
    obs = regime_observations(
        train[VOL_COL].to_numpy(), train[RET_COL].to_numpy(), train[SPREAD_COL].to_numpy()
    )
    return RegimeHMM(s.hmm_states, s.seed).fit(obs)


def add_regime_features(frame: pd.DataFrame, hmm: RegimeHMM, s: RegimeSettings) -> pd.DataFrame:
    cols = regime_columns(s.hmm_states)
    out = np.zeros((len(frame), len(cols)))
    ts_all = frame["ts"].to_numpy()
    inputs = ("ts", "mid", STEP_RET_COL, VOL_COL, RET_COL, SPREAD_COL)
    for _, group in frame.groupby("instrument", sort=False).indices.items():
        idx = np.asarray(group)[np.argsort(ts_all[group], kind="stable")]
        state = OnlineRegimeState(hmm, s)
        sub = frame.iloc[idx]
        columns = [sub[c].to_numpy() for c in inputs]
        for j, row in enumerate(idx):
            ts, mid, step, vol, ret, spread = (c[j] for c in columns)
            out[row] = state.update(
                int(ts), float(mid), float(step), float(vol), float(ret), float(spread)
            )
    result = frame.copy()
    result[cols] = out
    return result
