"""Research workflows behind the CLI: dataset -> train -> backtest -> execution study.

Every workflow writes a JSON report under <artifacts>/reports/<run id>/ and records what
it produced (dataset id, bundle id, git commit) so results are reproducible and traceable.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from niftypulse.backtest.costs import cost_model_for
from niftypulse.backtest.engine import run_signal_backtest
from niftypulse.backtest.simulator import TapeSimulator, build_tape, tape_index
from niftypulse.core import telemetry
from niftypulse.core.config import (
    cost_settings,
    db_settings,
    feature_settings,
    feed_settings,
    label_settings,
    lake_settings,
    model_settings,
    policy_settings,
    regime_settings,
    research_settings,
)
from niftypulse.core.db import Database
from niftypulse.domain.instruments import Venue
from niftypulse.features.dataset import Dataset, build_dataset, replay
from niftypulse.marketdata.tick_store import TickReader
from niftypulse.models.fill_probability import FillProbabilityModel, label_fills
from niftypulse.models.registry import ModelInput
from niftypulse.policy.execution_agent import ExecutionEnv, PPOAgent, evaluate
from niftypulse.research.bundle import ModelBundle, git_sha, load_bundle, save_bundle
from niftypulse.research.lead_lag import lead_lag_matrix
from niftypulse.research.training import ResearchContext, run_protocol
from niftypulse.storage.repositories import BacktestRepository, ModelRepository
from niftypulse.storage.schema import BacktestRun, ModelVersion
from niftypulse.validation.metrics import expected_calibration_error

log = telemetry.logger(__name__)

_NS = 1_000_000_000
FILL_FEATURES = [
    "spread_bps",
    "imbalance_l1",
    "imbalance_l5",
    "ofi_l1_5s",
    "trade_imbalance_5s",
    "trade_rate_5s",
    "realized_vol_bps_30s",
]
EXEC_SIGNALS = ["imbalance_l1", "ofi_l1_5s", "trade_imbalance_5s"]


@dataclass(frozen=True, slots=True)
class Period:
    instruments: list[str]
    start: date
    end: date


def _reports_dir(run_id: str) -> Path:
    path = Path(research_settings().artifacts_root) / "reports" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload, indent=2, default=lambda o: float(o) if isinstance(o, np.generic) else str(o)
        )
    )


def dataset_path(dataset_id: str) -> Path:
    return Path(research_settings().artifacts_root) / "datasets" / f"{dataset_id}.parquet"


def build(period: Period) -> Dataset:
    ds = build_dataset(
        TickReader(lake_settings().root),
        period.instruments,
        period.start,
        period.end,
        feature_settings(),
        label_settings(),
        feed_settings().publish_depth,
    )
    path = dataset_path(ds.dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.frame.to_parquet(path)
    meta = {"feature_set": ds.feature_set, "features": ds.feature_names, "rows": len(ds.frame)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    log.info(
        "dataset.built", dataset_id=ds.dataset_id, rows=len(ds.frame), feature_set=ds.feature_set
    )
    return ds


def load_dataset(dataset_id: str) -> Dataset:
    path = dataset_path(dataset_id)
    meta = json.loads(path.with_suffix(".json").read_text())
    return Dataset(pd.read_parquet(path), meta["features"], meta["feature_set"], dataset_id)


def train(
    dataset: Dataset, horizon_s: float, families: list[str]
) -> tuple[ModelBundle, dict[str, Any]]:
    ctx = ResearchContext(
        model_settings(), regime_settings(), research_settings(), horizon_s, families
    )
    bundle, report = run_protocol(dataset, ctx)
    bundle.feature_settings = feature_settings().model_dump(mode="json")
    root = Path(research_settings().artifacts_root)
    folder, digest = save_bundle(bundle, root)
    report["bundle"] = {
        "id": bundle.bundle_id,
        "path": str(folder),
        "sha256": digest,
        "git_sha": git_sha(),
    }
    out = _reports_dir(bundle.bundle_id)
    _write(out / "training.json", report)
    _save_holdout_predictions(bundle, dataset, out)
    asyncio.run(_register(bundle, report, digest, folder))
    return bundle, report


def _save_holdout_predictions(bundle: ModelBundle, dataset: Dataset, out: Path) -> None:
    """Chosen arm's calibrated probabilities on the holdout, for the backtest."""
    from niftypulse.regime.features import add_regime_features
    from niftypulse.validation.purged_cv import holdout_split

    rs, research = regime_settings(), research_settings()
    frame = dataset.frame.dropna(subset=[dataset.label_col(bundle.horizon_s)]).reset_index(
        drop=True
    )
    full = add_regime_features(frame, bundle.hmm, rs)
    _, hold = holdout_split(
        full["ts"].to_numpy(), research.holdout_fraction, bundle.horizon_s + research.embargo_s
    )
    chosen = next(iter(bundle.arms))
    ts = full["ts"].to_numpy()
    context = np.flatnonzero(ts >= ts[hold].min() - int(model_settings().seq_len * 2 * _NS))
    proba = bundle.arms[chosen].predict_proba(ModelInput(full.iloc[context], bundle.features))
    keep = np.isin(context, hold)
    preds = full.iloc[context[keep]][
        ["instrument", "ts", "regime", dataset.return_col(bundle.horizon_s)]
    ].copy()
    preds[["p_down", "p_flat", "p_up"]] = proba[keep]
    preds["arm"] = chosen
    preds.to_parquet(out / "holdout_predictions.parquet")


async def _register(bundle: ModelBundle, report: dict[str, Any], digest: str, folder: Path) -> None:
    db = Database(db_settings())
    try:
        async with db.session() as s:
            chosen = report["selection"]["chosen"]
            await ModelRepository(s).register(
                ModelVersion(
                    id=bundle.bundle_id,
                    family="+".join(bundle.arms),
                    target="direction",
                    horizon_s=int(bundle.horizon_s),
                    feature_set=bundle.feature_set,
                    git_sha=report["bundle"]["git_sha"],
                    mlflow_run_id=report["dataset_id"],
                    artifact_uri=str(folder),
                    artifact_sha256=digest,
                    train_start=datetime.fromisoformat(report["period"]["start"]).replace(
                        tzinfo=UTC
                    ),
                    train_end=datetime.fromisoformat(report["period"]["end"]).replace(tzinfo=UTC),
                    metrics={
                        "chosen": chosen,
                        "cv": report["cv"][chosen],
                        "holdout": report["holdout"][chosen],
                    },
                )
            )
    finally:
        await db.close()


def backtest(bundle_id: str, period: Period) -> dict[str, Any]:
    out = _reports_dir(bundle_id)
    preds = pd.read_parquet(out / "holdout_predictions.parquet")
    start = pd.to_datetime(preds["ts"].min(), unit="ns").date()
    end = pd.to_datetime(preds["ts"].max(), unit="ns").date()
    reader = TickReader(lake_settings().root)
    depth = feed_settings().publish_depth
    tapes, costs = {}, {}
    for key in period.instruments:
        events = replay(reader, [key], start, end, depth)
        tapes[key] = TapeSimulator(build_tape(events, label_settings().sample_interval_ms))
        costs[key] = cost_model_for(Venue(key.split("|", 1)[0]), cost_settings())
    training = json.loads((out / "training.json").read_text())
    bundle = load_bundle(Path(training["bundle"]["path"]), training["bundle"]["sha256"])
    conditional = bundle.conditional_means
    ret_col = next(c for c in preds.columns if c.startswith("ret_bps_"))
    horizon = float(ret_col.removeprefix("ret_bps_").removesuffix("s"))
    result = run_signal_backtest(
        preds, tapes, costs, conditional, policy_settings(), rebalance_s=horizon
    )
    step_s = label_settings().sample_interval_ms / 1000
    summary = result.summary(steps_per_year=365 * 86_400 / step_s)
    report = {
        "bundle_id": bundle_id,
        "period": {"start": str(start), "end": str(end)},
        "summary": summary,
    }
    _write(out / "backtest.json", report)
    result.equity_bps.to_frame("net_bps").assign(gross_bps=result.gross_equity_bps).to_parquet(
        out / "equity.parquet"
    )
    asyncio.run(_record_backtest(bundle_id, start, end, summary))
    return report


async def _record_backtest(bundle_id: str, start: date, end: date, summary: dict[str, Any]) -> None:
    db = Database(db_settings())
    try:
        async with db.session() as s:
            await BacktestRepository(s).add(
                BacktestRun(
                    id=f"bt-{int(datetime.now(UTC).timestamp())}",
                    model_version=bundle_id,
                    policy_version="convex-sizer",
                    git_sha=git_sha(),
                    data_start=datetime.combine(start, datetime.min.time(), UTC),
                    data_end=datetime.combine(end, datetime.min.time(), UTC),
                    config=policy_settings().model_dump(mode="json"),
                    results=json.loads(json.dumps(summary, default=float)),
                )
            )
    finally:
        await db.close()


def lead_lag(dataset: Dataset) -> dict[str, Any]:
    """Transfer-entropy lead-lag between every pair of instruments in the dataset."""
    rs = research_settings()
    rows = lead_lag_matrix(
        dataset.frame,
        rs.lead_lag_steps,
        rs.lead_lag_surrogates,
        rs.lead_lag_min_shift_steps,
        policy_settings().seed,
    )
    strongest: dict[str, dict[str, Any]] = {}
    for r in rows:
        pair = f"{r.source} -> {r.target}"
        if pair not in strongest or r.excess_bits > strongest[pair]["excess_bits"]:
            strongest[pair] = r.as_dict()
    report = {
        "dataset_id": dataset.dataset_id,
        "step_s": label_settings().sample_interval_ms / 1000,
        "strongest_lag_per_pair": strongest,
        "all": [r.as_dict() for r in rows],
    }
    _write(_reports_dir(dataset.dataset_id) / "lead_lag.json", report)
    return report


def execution_study(
    bundle: ModelBundle, dataset: Dataset, instrument: str, period: Period
) -> dict[str, Any]:
    """Fill-probability model and PPO execution agent on one instrument's tape."""
    reader = TickReader(lake_settings().root)
    depth = feed_settings().publish_depth
    events = list(replay(reader, [instrument], period.start, period.end, depth))
    tape = build_tape(events, label_settings().sample_interval_ms)
    sim = TapeSimulator(tape)
    feats = (
        dataset.frame[dataset.frame["instrument"] == instrument]
        .sort_values("ts")
        .reset_index(drop=True)
    )
    fill_horizon = policy_settings().exec_horizon_s / 6
    fills = label_fills(events, label_settings().sample_interval_ms / 1000, fill_horizon)
    pos = np.searchsorted(feats["ts"].to_numpy(), fills["ts"].to_numpy(), side="right") - 1
    fills = fills[pos >= 0].reset_index(drop=True)
    pos = pos[pos >= 0]
    for col in FILL_FEATURES:
        fills[col] = feats[col].to_numpy()[pos]
    ti = tape_index(sim.ts, fills["ts"].to_numpy())
    fills["queue_ahead"] = np.where(fills["side"] == "buy", sim.bid_qty[ti], sim.ask_qty[ti])
    n = len(fills)
    a, b, c = int(n * 0.5), int(n * 0.65), int(n * 0.8)
    model = FillProbabilityModel(model_settings(), [*FILL_FEATURES, "queue_ahead"])
    model.fit(fills.iloc[:a], fills.iloc[a:b], fills.iloc[b:c])
    test = fills.iloc[c:]
    p = model.predict(test)
    y = test["filled"].to_numpy()
    from sklearn.metrics import brier_score_loss, roc_auc_score

    fill_report = {
        "horizon_s": fill_horizon,
        "samples": n,
        "base_fill_rate": float(fills["filled"].mean()),
        "test_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "test_brier": float(brier_score_loss(y, p)),
        "test_brier_base_rate": float(brier_score_loss(y, np.full(len(y), y.mean()))),
        "test_ece": expected_calibration_error(np.column_stack([1 - p, p]), y.astype(np.int64)),
        "median_cancel_share": float(np.nanmedian(fills["cancel_share"])),
    }

    chosen = next(iter(bundle.arms))
    from niftypulse.regime.features import add_regime_features

    feats_r = add_regime_features(feats, bundle.hmm, regime_settings())
    proba = bundle.arms[chosen].predict_proba(ModelInput(feats_r, bundle.features))
    score = proba[:, 2] - proba[:, 0]
    tpos = np.clip(
        np.searchsorted(feats["ts"].to_numpy(), sim.ts, side="right") - 1, 0, len(feats) - 1
    )
    signals = np.column_stack([feats[s].to_numpy()[tpos] for s in EXEC_SIGNALS] + [score[tpos]])
    ps = policy_settings()
    costs = cost_model_for(Venue(instrument.split("|", 1)[0]), cost_settings())
    env = ExecutionEnv(sim, signals, costs, ps.exec_parent_lots, ps.exec_horizon_s)
    last = len(sim) - ps.exec_horizon_s - 2
    split = int(last * 0.7)
    rng = np.random.default_rng(ps.seed)
    train_starts = np.arange(0, split, 5, dtype=np.int64)
    test_starts = np.arange(split + ps.exec_horizon_s, last, ps.exec_horizon_s, dtype=np.int64)
    agent = PPOAgent(ps)
    agent.train(env, train_starts, rng)
    report = {
        "instrument": instrument,
        "fill_model": fill_report,
        "execution": evaluate(env, test_starts, agent),
        "ppo_learning_curve": agent.history,
    }
    _write(_reports_dir(bundle.bundle_id) / "execution.json", report)
    return report
