"""Causal regime features appended to the base feature set.

For each instrument, in time order: Kalman drift and innovation, BOCPD change
probability and run length, and HMM filtered state probabilities. The HMM is fitted on
the training slice only and then filtered forward over every row, so test rows never
influence their own features.
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


def fit_regime_model(train: pd.DataFrame, s: RegimeSettings) -> RegimeHMM:
    obs = regime_observations(
        train[VOL_COL].to_numpy(), train[RET_COL].to_numpy(), train[SPREAD_COL].to_numpy()
    )
    return RegimeHMM(s.hmm_states, s.seed).fit(obs)


def add_regime_features(frame: pd.DataFrame, hmm: RegimeHMM, s: RegimeSettings) -> pd.DataFrame:
    cols = regime_columns(s.hmm_states)
    out = np.zeros((len(frame), len(cols)))
    for _, group in frame.groupby("instrument", sort=False).indices.items():
        idx = np.asarray(group)[np.argsort(frame["ts"].to_numpy()[group], kind="stable")]
        kf = LocalLinearTrend(s.kalman_level_var, s.kalman_velocity_var, s.kalman_obs_var)
        cpd = BOCPD(s.bocpd_hazard_lambda, s.bocpd_max_run, s.bocpd_recent_window)
        flt = OnlineRegimeFilter(hmm)
        sub = frame.iloc[idx]
        obs = regime_observations(
            sub[VOL_COL].to_numpy(), sub[RET_COL].to_numpy(), sub[SPREAD_COL].to_numpy()
        )
        mids, ts, steps = sub["mid"].to_numpy(), sub["ts"].to_numpy(), sub[STEP_RET_COL].to_numpy()
        for j, row in enumerate(idx):
            kf.update(float(np.log(mids[j])), int(ts[j]))
            cp, run = cpd.update(float(steps[j]))
            probs = flt.update(obs[j])
            out[row] = [
                kf.velocity_bps_per_s,
                kf.innovation_z,
                cp,
                run,
                *probs,
                float(np.argmax(probs)),
            ]
    result = frame.copy()
    result[cols] = out
    return result
