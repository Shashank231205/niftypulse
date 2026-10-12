"""Model bundle: everything the live predictor needs, saved as one versioned unit.

A bundle holds the calibrated arm models, the regime HMM, feature definitions, class
prior and conditional move sizes. Its directory is content-addressed by sha256 and
registered in Postgres, so the predictor can verify the artifact before loading it and
every live prediction carries the bundle id.
"""

import hashlib
import json
import pickle
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from niftypulse.core.db import Database
from niftypulse.models.calibration import BiasedTemperature
from niftypulse.models.registry import DirectionModel, ModelInput
from niftypulse.models.volatility import GBMVolatility
from niftypulse.regime.hmm import RegimeHMM

FloatArray = npt.NDArray[np.float64]

_BUNDLE_FILE = "bundle.pkl"
_META_FILE = "meta.json"


@dataclass
class CalibratedArm:
    model: DirectionModel
    calibrator: BiasedTemperature

    def predict_proba(self, data: ModelInput) -> FloatArray:
        return self.calibrator.transform(self.model.predict_proba(data))


@dataclass
class ModelBundle:
    bundle_id: str
    horizon_s: float
    feature_set: str
    feature_settings: dict[str, Any]
    base_features: list[str]
    features: list[str]
    arms: dict[str, CalibratedArm]
    hmm: RegimeHMM
    class_prior: FloatArray
    conditional_means: dict[str, tuple[float, float]]
    vol_model: GBMVolatility | None = None
    vol_horizon_s: float = 0.0
    reference_bins: dict[str, list[float]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def git_sha() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        out = subprocess.run([git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)  # noqa: S603 - fixed argv
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def save_bundle(bundle: ModelBundle, root: Path) -> tuple[Path, str]:
    folder = root / "models" / bundle.bundle_id
    folder.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(bundle)
    digest = hashlib.sha256(payload).hexdigest()
    (folder / _BUNDLE_FILE).write_bytes(payload)
    meta = {
        "bundle_id": bundle.bundle_id,
        "sha256": digest,
        "feature_set": bundle.feature_set,
        "horizon_s": bundle.horizon_s,
        "arms": sorted(bundle.arms),
        "created_at": bundle.created_at,
        "metrics": bundle.metrics,
    }
    (folder / _META_FILE).write_text(json.dumps(meta, indent=2, default=float))
    return folder, digest


def load_bundle(folder: Path, expected_sha256: str) -> ModelBundle:
    """Refuse to unpickle anything whose hash does not match the registry entry."""
    payload = (folder / _BUNDLE_FILE).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"bundle {folder.name} hash mismatch: "
            f"registry {expected_sha256[:12]}, file {actual[:12]}"
        )
    bundle = pickle.loads(payload)  # noqa: S301  # nosec B301 - integrity verified above
    if not isinstance(bundle, ModelBundle):
        raise TypeError("artifact is not a ModelBundle")
    return bundle


async def load_production(db: Database) -> tuple[ModelBundle, str]:
    """The promoted bundle and its code commit, verified against the registry hash."""
    from niftypulse.storage.repositories import ModelRepository

    async with db.session() as s:
        mv = await ModelRepository(s).production()
    if mv is None:
        raise LookupError("no production model; run research and `niftypulse promote <id>`")
    return load_bundle(Path(mv.artifact_uri), mv.artifact_sha256), mv.git_sha
