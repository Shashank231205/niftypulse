"""Regularised multinomial logistic regression: the baseline every other model must beat."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

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
class LogisticModel(DirectionModel):
    family = "logistic"

    def __init__(self, settings: ModelSettings) -> None:
        super().__init__(settings)
        self._pipe = Pipeline(
            [
                ("scale", RobustScaler(quantile_range=(5.0, 95.0))),
                (
                    "clf",
                    LogisticRegression(
                        C=settings.logistic_c, max_iter=2000, random_state=settings.seed
                    ),
                ),
            ]
        )

    def fit(
        self, train: ModelInput, y_train: FloatArray, valid: ModelInput, y_valid: FloatArray
    ) -> None:
        y = class_index(y_train)
        x = np.clip(train.matrix(), -1e6, 1e6)
        self._pipe.fit(x, y, clf__sample_weight=balanced_weights(y))

    def predict_proba(self, data: ModelInput) -> FloatArray:
        proba = self._pipe.predict_proba(np.clip(data.matrix(), -1e6, 1e6))
        out = np.zeros((len(data), 3))
        out[:, self._pipe.classes_] = proba
        return out
