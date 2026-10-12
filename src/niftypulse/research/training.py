"""The research protocol, end to end.

1. Split off the final holdout (last fraction of time, behind a purge gap). It is not
   looked at until every choice has been made.
2. Purged walk-forward CV on the development set for every model family. Per fold, the
   regime HMM is fitted on that fold's training rows only; the fold's training block is
   split again by time into fit / early-stopping / calibration slices.
3. Leakage checks (shifted labels, suspicious-feature scan, split overlap).
4. Model selection on CV log-loss skill only. The holdout is then scored once for every
   family, so the report shows how CV ranking held up out of sample, good or bad.
5. Volatility models (HAR baseline vs GBM) by QLIKE.
6. Distillation of the sequence teacher into a small MLP, then pruning and INT8
   quantization, each step measured for accuracy, size and latency.
7. Bundle the top arms with calibrators and the HMM; register with lineage.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from niftypulse.core import telemetry
from niftypulse.core.config import ModelSettings, RegimeSettings, ResearchSettings
from niftypulse.features.dataset import Dataset
from niftypulse.models.calibration import BiasedTemperature
from niftypulse.models.distillation import DistilledMLP, benchmark_onnx, quantize_onnx
from niftypulse.models.registry import ModelInput, class_index, create, load_all_families
from niftypulse.models.sequence import build_windows
from niftypulse.models.volatility import GBMVolatility, HARVolatility
from niftypulse.regime.features import add_regime_features, fit_regime_model, regime_columns
from niftypulse.research.bundle import CalibratedArm, ModelBundle
from niftypulse.validation.leakage import run_leakage_checks
from niftypulse.validation.metrics import ClassPrior, classification_report, qlike
from niftypulse.validation.purged_cv import PurgedWalkForward, holdout_split

log = telemetry.logger(__name__)

_NS = 1_000_000_000
SEQUENCE_FAMILIES = {"lstm", "tcn", "transformer"}


@dataclass
class ResearchContext:
    model: ModelSettings
    regime: RegimeSettings
    research: ResearchSettings
    horizon_s: float
    families: list[str]


def _time_slices(
    idx: np.ndarray, ts: np.ndarray, valid_fraction: float, gap_ns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split training rows by time into (fit, early-stop, calibration) with purge gaps."""
    order = idx[np.argsort(ts[idx], kind="stable")]
    t = ts[order]
    t0, t1 = t[0], t[-1]
    cut_valid = t1 - int((t1 - t0) * valid_fraction)
    cut_calib = cut_valid + (t1 - cut_valid) // 2
    fit = order[t < cut_valid - gap_ns]
    early = order[(t >= cut_valid) & (t < cut_calib - gap_ns)]
    calib = order[t >= cut_calib]
    return fit, early, calib


class _Predictor(Protocol):
    def predict_proba(self, data: ModelInput) -> np.ndarray: ...


def _predict(
    arm: _Predictor, frame: pd.DataFrame, features: list[str], target: np.ndarray, context_s: float
) -> np.ndarray:
    """Predict target rows with enough preceding rows for sequence windows."""
    ts = frame["ts"].to_numpy()
    start = ts[target].min() - int(context_s * _NS)
    end = ts[target].max()
    window = np.flatnonzero((ts >= start) & (ts <= end))
    proba = arm.predict_proba(ModelInput(frame.iloc[window], features))
    pos = pd.Series(np.arange(len(window)), index=window)
    return proba[pos.loc[target].to_numpy()]


def fit_arm(
    family: str,
    ctx: ResearchContext,
    frame: pd.DataFrame,
    features: list[str],
    label: str,
    rows: np.ndarray,
) -> CalibratedArm:
    ts = frame["ts"].to_numpy()
    gap = int((ctx.horizon_s + ctx.research.embargo_s) * _NS)
    fit, early, calib = _time_slices(rows, ts, 0.3, gap)
    y = frame[label].to_numpy()
    model = create(family, ctx.model)
    model.fit(
        ModelInput(frame.iloc[fit], features),
        y[fit],
        ModelInput(frame.iloc[early], features),
        y[early],
    )
    arm = CalibratedArm(model, BiasedTemperature())
    context = ctx.model.seq_len * 2.0
    raw = _predict(model, frame, features, calib, context)
    arm.calibrator.fit(raw, class_index(y[calib]))
    return arm


def _report(
    ctx: ResearchContext,
    proba: np.ndarray,
    frame: pd.DataFrame,
    rows: np.ndarray,
    label: str,
    ret: str,
    prior: ClassPrior,
) -> dict[str, Any]:
    rep = classification_report(
        proba,
        frame[label].to_numpy()[rows],
        frame[ret].to_numpy()[rows],
        prior.rows(frame["instrument"].to_numpy()[rows]),
        ctx.research.min_abs_score,
        ctx.research.bootstrap_block,
    )
    return rep.as_dict()


def _prior(frame: pd.DataFrame, label: str, rows: np.ndarray) -> ClassPrior:
    return ClassPrior.fit(frame["instrument"].to_numpy()[rows], frame[label].to_numpy()[rows])


def run_protocol(dataset: Dataset, ctx: ResearchContext) -> tuple[ModelBundle, dict[str, Any]]:
    load_all_families()
    t_start = time.time()
    frame = dataset.frame.dropna(subset=[dataset.label_col(ctx.horizon_s)]).reset_index(drop=True)
    label, ret = dataset.label_col(ctx.horizon_s), dataset.return_col(ctx.horizon_s)
    ts = frame["ts"].to_numpy()
    gap_s = ctx.horizon_s + ctx.research.embargo_s
    dev, hold = holdout_split(ts, ctx.research.holdout_fraction, gap_s)
    dev_frame = frame.iloc[dev].reset_index(drop=True)
    features = [*dataset.feature_names, *regime_columns(ctx.regime.hmm_states)]
    report: dict[str, Any] = {
        "dataset_id": dataset.dataset_id,
        "feature_set": dataset.feature_set,
        "horizon_s": ctx.horizon_s,
        "rows": {"total": len(frame), "development": len(dev), "holdout": len(hold)},
        "period": {"start": str(pd.to_datetime(ts.min())), "end": str(pd.to_datetime(ts.max()))},
    }

    cv = PurgedWalkForward(ctx.research.cv_splits, ctx.horizon_s, ctx.research.embargo_s)
    cv_scores: dict[str, list[dict[str, Any]]] = {f: [] for f in ctx.families}
    dev_ts = dev_frame["ts"].to_numpy()
    for fold in cv.split(dev_ts):
        hmm = fit_regime_model(dev_frame.iloc[fold.train], ctx.regime)
        fold_frame = add_regime_features(dev_frame, hmm, ctx.regime)
        prior = _prior(fold_frame, label, fold.train)
        for family in ctx.families:
            t0 = time.time()
            arm = fit_arm(family, ctx, fold_frame, features, label, fold.train)
            proba = _predict(arm, fold_frame, features, fold.test, ctx.model.seq_len * 2.0)
            rep = _report(ctx, proba, fold_frame, fold.test, label, ret, prior)
            rep["fit_seconds"] = time.time() - t0
            cv_scores[family].append(rep)
            log.info(
                "cv.fold",
                fold=fold.number,
                family=family,
                skill=rep["log_loss_skill"],
                ic=rep["ic"],
            )
    report["cv"] = {
        f: {
            "folds": s,
            "mean_log_loss_skill": float(np.mean([r["log_loss_skill"] for r in s]))
            if s
            else float("nan"),
            "sd_log_loss_skill": float(np.std([r["log_loss_skill"] for r in s]))
            if s
            else float("nan"),
            "mean_ic": float(np.nanmean([r["ic"] for r in s])) if s else float("nan"),
        }
        for f, s in cv_scores.items()
    }

    hmm = fit_regime_model(dev_frame, ctx.regime)
    full = add_regime_features(frame, hmm, ctx.regime)
    dev_full = full.iloc[dev].reset_index(drop=True)
    report["leakage"] = run_leakage_checks(
        dev_full,
        features,
        label,
        ret,
        cv,
        lambda tr, te, lab: _lgbm_skill(ctx, tr, te, features, lab),
        ctx.research.leakage_min_shift_fraction,
        ctx.research.leakage_max_abs_ic,
        ctx.research.leakage_max_shuffled_skill,
    ).as_dict()

    ranked = sorted(
        ctx.families, key=lambda f: report["cv"][f]["mean_log_loss_skill"], reverse=True
    )
    prior = _prior(full, label, dev)
    arms: dict[str, CalibratedArm] = {}
    report["holdout"] = {}
    for family in ctx.families:
        arm = fit_arm(family, ctx, full, features, label, dev)
        proba = _predict(arm, full, features, hold, ctx.model.seq_len * 2.0)
        report["holdout"][family] = _report(ctx, proba, full, hold, label, ret, prior)
        arms[family] = arm
    report["selection"] = {"ranked_by_cv": ranked, "chosen": ranked[0]}
    report["volatility"], vol_model = _volatility(ctx, full, dataset, dev, hold)
    teacher = next((f for f in ranked if f in SEQUENCE_FAMILIES), None)
    if teacher:
        report["compression"] = _compress(
            ctx, full, features, label, ret, dev, hold, arms[teacher], prior
        )

    conditional = {}
    for key, g in full.iloc[dev].groupby("instrument"):
        r, y = g[ret].to_numpy(), g[label].to_numpy()
        conditional[str(key)] = (
            float(np.nanmean(r[y == 1])) if (y == 1).any() else 0.0,
            float(np.nanmean(r[y == -1])) if (y == -1).any() else 0.0,
        )
    top = ranked[: max(1, min(3, len(ranked)))]
    bundle = ModelBundle(
        bundle_id=f"{dataset.feature_set}-h{ctx.horizon_s:g}s-{int(time.time())}",
        horizon_s=ctx.horizon_s,
        feature_set=dataset.feature_set,
        feature_settings={},
        base_features=dataset.feature_names,
        features=features,
        arms={f: arms[f] for f in top},
        hmm=hmm,
        class_prior=prior.overall,
        conditional_means=conditional,
        vol_model=vol_model,
        vol_horizon_s=float(
            report["volatility"]["target"].removeprefix("fwd_rv_bps_").removesuffix("s")
        ),
        reference_bins=reference_bins(full.iloc[dev], dataset.feature_names),
        metrics={"cv": {f: report["cv"][f]["mean_log_loss_skill"] for f in top}},
    )
    report["elapsed_s"] = time.time() - t_start
    return bundle, report


def reference_bins(
    frame: pd.DataFrame, features: list[str], n_bins: int = 10
) -> dict[str, list[float]]:
    """Decile edges of each feature on development data: the drift monitor's reference."""
    out = {}
    for name in features:
        values = frame[name].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values):
            out[name] = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1))).tolist()
    return out


def _lgbm_skill(
    ctx: ResearchContext, train: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str
) -> float:
    model = create("lightgbm", ctx.model)
    n = len(train)
    fit, early = train.iloc[: int(n * 0.8)], train.iloc[int(n * 0.8) :]
    model.fit(
        ModelInput(fit, features),
        fit[label].to_numpy(),
        ModelInput(early, features),
        early[label].to_numpy(),
    )
    proba = model.predict_proba(ModelInput(test, features))
    prior = _prior(train, label, np.arange(n)).rows(test["instrument"].to_numpy())
    y = class_index(test[label].to_numpy())
    rows = np.arange(len(y))
    ll = -np.mean(np.log(np.clip(proba[rows, y], 1e-7, 1)))
    base = -np.mean(np.log(np.clip(prior[rows, y], 1e-7, 1)))
    return float(1 - ll / base)


def _volatility(
    ctx: ResearchContext, full: pd.DataFrame, dataset: Dataset, dev: np.ndarray, hold: np.ndarray
) -> tuple[dict[str, Any], GBMVolatility]:
    target = next(c for c in full.columns if c.startswith("fwd_rv_bps_"))
    ok = np.isfinite(full[target].to_numpy())
    dev, hold = dev[ok[dev]], hold[ok[hold]]
    rv_cols = [c for c in dataset.feature_names if c.startswith("realized_vol_bps_")]
    floor = ctx.model.vol_floor_bps
    y = full[target].to_numpy()
    har = HARVolatility(rv_cols, floor)
    har.fit(ModelInput(full.iloc[dev], rv_cols), y[dev])
    gbm = GBMVolatility(ctx.model)
    cut = int(len(dev) * 0.8)
    gbm.fit(
        ModelInput(full.iloc[dev[:cut]], dataset.feature_names),
        y[dev[:cut]],
        ModelInput(full.iloc[dev[cut:]], dataset.feature_names),
        y[dev[cut:]],
    )
    realized = np.maximum(y[hold], floor) ** 2
    out = {
        "target": target,
        "har_qlike": qlike(har.predict_vol(ModelInput(full.iloc[hold], rv_cols)) ** 2, realized),
        "gbm_qlike": qlike(
            gbm.predict_vol(ModelInput(full.iloc[hold], dataset.feature_names)) ** 2, realized
        ),
        "naive_qlike": qlike(np.full(len(hold), np.mean(np.maximum(y[dev], floor) ** 2)), realized),
    }
    return out, gbm


def _compress(
    ctx: ResearchContext,
    full: pd.DataFrame,
    features: list[str],
    label: str,
    ret: str,
    dev: np.ndarray,
    hold: np.ndarray,
    teacher: CalibratedArm,
    prior: ClassPrior,
) -> dict[str, Any]:
    root = Path(ctx.research.artifacts_root) / "compression"
    root.mkdir(parents=True, exist_ok=True)
    gap = int((ctx.horizon_s + ctx.research.embargo_s) * _NS)
    fit, early, _ = _time_slices(dev, full["ts"].to_numpy(), 0.2, gap)
    y = full[label].to_numpy()
    student = DistilledMLP(ctx.model)
    student.teacher_proba = _predict(teacher, full, features, fit, ctx.model.seq_len * 2.0)
    student.fit(
        ModelInput(full.iloc[fit], features),
        y[fit],
        ModelInput(full.iloc[early], features),
        y[early],
    )
    hold_in = ModelInput(full.iloc[hold], features)
    out: dict[str, Any] = {
        "teacher": teacher.model.family,
        "teacher_holdout": _report(
            ctx,
            _predict(teacher, full, features, hold, ctx.model.seq_len * 2.0),
            full,
            hold,
            label,
            ret,
            prior,
        ),
        "student_holdout": _report(
            ctx, student.predict_proba(hold_in), full, hold, label, ret, prior
        ),
    }
    # windows are per instrument, so time on one instrument's consecutive holdout rows
    first = full["instrument"].to_numpy()[hold[0]]
    one = hold[full["instrument"].to_numpy()[hold] == first][: ctx.model.seq_len * 2]
    out["teacher_latency_us"] = _torch_latency_us(teacher, full.iloc[one], features, ctx)
    f32 = root / "student.onnx"
    student.export_onnx(f32)
    student.prune(ctx.model.prune_amount)
    out["pruned_sparsity"] = student.sparsity()
    out["pruned_holdout"] = _report(
        ctx, student.predict_proba(hold_in), full, hold, label, ret, prior
    )
    pruned = root / "student_pruned.onnx"
    student.export_onnx(pruned)
    int8 = root / "student_pruned_int8.onnx"
    quantize_onnx(pruned, int8)
    student.export_cpp_weights(root / "student_pruned.npmlp")
    sample = student._scale(hold_in.matrix()[:256])
    out["latency"] = [benchmark_onnx(p, sample).as_dict() for p in (f32, pruned, int8)]
    out["student_parameters"] = (
        sum(p.numel() for p in student.net.parameters()) if student.net else 0
    )
    return out


def _torch_latency_us(
    arm: CalibratedArm, frame: pd.DataFrame, features: list[str], ctx: ResearchContext
) -> float:
    """Single-window forward pass of the sequence teacher, for comparison with the student."""
    model = arm.model
    net = getattr(model, "_net", None)
    scaler = getattr(model, "_scaler", None)
    if net is None or scaler is None:
        return float("nan")
    windows, ok = build_windows(
        ModelInput(frame, features), ctx.model.seq_len, getattr(model, "_step_ns", _NS)
    )
    if not ok.any():
        return float("nan")
    x = torch.from_numpy(scaler(windows[ok][:1]))
    net.eval()
    with torch.no_grad():
        for _ in range(20):
            net(x)
        times = []
        for _ in range(200):
            t0 = time.perf_counter_ns()
            net(x)
            times.append((time.perf_counter_ns() - t0) / 1e3)
    return float(np.median(times))
