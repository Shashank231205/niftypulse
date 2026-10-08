"""Gradient-boosted trees: LightGBM and XGBoost multiclass with early stopping."""

from typing import Any

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from niftypulse.core.config import ModelSettings
from niftypulse.models.registry import (
    DirectionModel,
    FloatArray,
    ModelInput,
    balanced_weights,
    class_index,
    register,
)


@register
class LightGBMModel(DirectionModel):
    family = "lightgbm"

    def __init__(self, settings: ModelSettings) -> None:
        super().__init__(settings)
        self._booster: lgb.Booster | None = None
        self.feature_importance: dict[str, float] = {}

    def fit(
        self, train: ModelInput, y_train: FloatArray, valid: ModelInput, y_valid: FloatArray
    ) -> None:
        s = self.settings
        yt, yv = class_index(y_train), class_index(y_valid)
        dtrain = lgb.Dataset(
            train.matrix(), yt, weight=balanced_weights(yt), feature_name=train.features
        )
        dvalid = lgb.Dataset(valid.matrix(), yv, weight=balanced_weights(yv), reference=dtrain)
        params: dict[str, Any] = {
            "objective": "multiclass",
            "num_class": 3,
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
        }
        self._booster = lgb.train(
            params,
            dtrain,
            num_boost_round=s.gbm_max_rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(s.early_stopping_rounds, verbose=False)],
        )
        gain = self._booster.feature_importance(importance_type="gain")
        total = float(gain.sum()) or 1.0
        self.feature_importance = {
            n: float(g) / total for n, g in zip(train.features, gain, strict=True)
        }

    def predict_proba(self, data: ModelInput) -> FloatArray:
        assert self._booster is not None, "fit() first"
        out = self._booster.predict(data.matrix(), num_iteration=self._booster.best_iteration)
        return np.asarray(out, dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        best = self._booster.best_iteration if self._booster else None
        return {"family": self.family, "best_iteration": best}


@register
class XGBoostModel(DirectionModel):
    family = "xgboost"

    def __init__(self, settings: ModelSettings) -> None:
        super().__init__(settings)
        s = settings
        self._clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=s.gbm_max_rounds,
            learning_rate=s.gbm_learning_rate,
            max_depth=s.xgb_max_depth,
            min_child_weight=s.gbm_min_child_samples / 10,
            subsample=s.gbm_subsample,
            colsample_bytree=s.gbm_colsample,
            reg_lambda=s.gbm_l2,
            tree_method="hist",
            early_stopping_rounds=s.early_stopping_rounds,
            eval_metric="mlogloss",
            random_state=s.seed,
            n_jobs=s.n_threads,
        )

    def fit(
        self, train: ModelInput, y_train: FloatArray, valid: ModelInput, y_valid: FloatArray
    ) -> None:
        yt, yv = class_index(y_train), class_index(y_valid)
        self._clf.fit(
            train.matrix(),
            yt,
            sample_weight=balanced_weights(yt),
            eval_set=[(valid.matrix(), yv)],
            sample_weight_eval_set=[balanced_weights(yv)],
            verbose=False,
        )

    def predict_proba(self, data: ModelInput) -> FloatArray:
        return np.asarray(self._clf.predict_proba(data.matrix()), dtype=np.float64)
