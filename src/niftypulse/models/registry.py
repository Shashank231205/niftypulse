"""Model interface (Strategy) and registry (Factory).

Every direction model, from logistic regression to the transformer, implements
`DirectionModel`, so training, evaluation, calibration, the bandit and the live predictor
treat them interchangeably. Models are created by family name from configuration.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from niftypulse.core.config import ModelSettings

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Rows sorted by (ts, instrument); sequence models also use `instrument` and `ts`."""

    frame: pd.DataFrame
    features: list[str]

    def matrix(self) -> FloatArray:
        x = self.frame[self.features].to_numpy(dtype=np.float64)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self) -> int:
        return len(self.frame)


def class_index(labels: FloatArray) -> npt.NDArray[np.int64]:
    """(-1, 0, +1) -> (0, 1, 2)."""
    return (labels.astype(np.int64) + 1).astype(np.int64)


def balanced_weights(y_idx: npt.NDArray[np.int64]) -> FloatArray:
    """Inverse-frequency sample weights: the rare up/down classes are not drowned by flat."""
    counts = np.bincount(y_idx, minlength=3).astype(np.float64)
    w = len(y_idx) / (3.0 * np.maximum(counts, 1.0))
    return w[y_idx]


class DirectionModel(ABC):
    family: ClassVar[str]

    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings

    @abstractmethod
    def fit(
        self, train: ModelInput, y_train: FloatArray, valid: ModelInput, y_valid: FloatArray
    ) -> None:
        """Labels are -1/0/+1. `valid` is for early stopping only, never for reporting."""

    @abstractmethod
    def predict_proba(self, data: ModelInput) -> FloatArray:
        """(n, 3) probabilities for (down, flat, up)."""

    def describe(self) -> dict[str, Any]:
        return {"family": self.family}


MODELS: dict[str, Callable[[ModelSettings], DirectionModel]] = {}


def register[M: type[DirectionModel]](cls: M) -> M:
    MODELS[cls.family] = cls
    return cls


def create(family: str, settings: ModelSettings) -> DirectionModel:
    if family not in MODELS:
        raise KeyError(f"unknown model family {family!r}; known: {sorted(MODELS)}")
    return MODELS[family](settings)


def load_all_families() -> None:
    """Import model modules so their @register decorators run."""
    from niftypulse.models import gbm, linear, sequence  # noqa: F401
