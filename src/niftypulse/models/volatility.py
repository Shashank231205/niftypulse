"""Short-horizon realized-volatility forecasting.

Target: log of forward realized volatility. Fitting in log space with squared error is
close to minimising QLIKE and keeps heavy-tailed spikes from dominating. Baseline is a
HAR-style linear model on realized vol at several look-backs (Corsi, 2009); the
challenger is LightGBM on the full feature set. Both are scored by QLIKE on variance.

A quiet window has exactly zero realized vol. Taken to log space unfloored, those rows
become extreme outliers and a model learns to forecast ~0 variance, which QLIKE punishes
without limit, so vols are floored at `vol_floor_bps` everywhere: inputs, targets, scoring.
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge

from niftypulse.core.config import ModelSettings
from niftypulse.models.registry import FloatArray, ModelInput


def _log_vol(v: FloatArray, floor_bps: float) -> FloatArray:
    return np.log(np.maximum(v, floor_bps))


@dataclass
class HARVolatility:
    """Ridge on log realized vols: the classic heterogeneous autoregressive structure."""

    rv_features: list[str]
    floor_bps: float
    _model: Ridge = field(default_factory=lambda: Ridge(alpha=1.0))

    def fit(self, data: ModelInput, target_vol: FloatArray) -> None:
        x = _log_vol(data.frame[self.rv_features].to_numpy(dtype=np.float64), self.floor_bps)
        self._model.fit(x, _log_vol(target_vol, self.floor_bps))

    def predict_vol(self, data: ModelInput) -> FloatArray:
        x = _log_vol(data.frame[self.rv_features].to_numpy(dtype=np.float64), self.floor_bps)
        return np.maximum(np.exp(self._model.predict(x)), self.floor_bps)


class GBMVolatility:
    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings
        self._booster: lgb.Booster | None = None

    def fit(
        self, train: ModelInput, y_train: FloatArray, valid: ModelInput, y_valid: FloatArray
    ) -> None:
        s = self.settings
        floor = s.vol_floor_bps
        dtrain = lgb.Dataset(train.matrix(), _log_vol(y_train, floor), feature_name=train.features)
        dvalid = lgb.Dataset(valid.matrix(), _log_vol(y_valid, floor), reference=dtrain)
        self._booster = lgb.train(
            {
                "objective": "regression",
                "learning_rate": s.gbm_learning_rate,
                "num_leaves": s.gbm_num_leaves,
                "min_data_in_leaf": s.gbm_min_child_samples,
                "feature_fraction": s.gbm_colsample,
                "bagging_fraction": s.gbm_subsample,
                "bagging_freq": 1,
                "lambda_l2": s.gbm_l2,
                "seed": s.seed,
                "verbosity": -1,
                "num_threads": s.n_threads,
            },
            dtrain,
            num_boost_round=s.gbm_max_rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(s.early_stopping_rounds, verbose=False)],
        )

    def predict_vol(self, data: ModelInput) -> FloatArray:
        assert self._booster is not None
        pred = self._booster.predict(data.matrix(), num_iteration=self._booster.best_iteration)
        vol = np.exp(np.asarray(pred, dtype=np.float64))
        return np.maximum(vol, self.settings.vol_floor_bps)
