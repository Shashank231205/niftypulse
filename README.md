# NiftyPulse

Tick-level forecasting, regime detection and execution research, run as traceable distributed services.

NiftyPulse records level-2 order books and trades, replays them through one feature engine shared by research and live code, and runs a leakage-checked research protocol: purged walk-forward CV, six model families, calibration, regime models, a cost-aware backtest and a reinforcement-learning execution study. The same code then runs live as services on a NATS JetStream bus, with every prediction traceable to the model artifact, the code commit and the market event behind it.

It was built for Indian markets (NSE index futures and options through Upstox). Until the broker account is active, every result below comes from Binance spot and USD-M perpetuals, which have the same structure: a deep order book, a futures-spot pair, and a leader instrument. The venue sits behind an adapter, so an NSE feed slots in without changes elsewhere.

## Results

Data: 6 hours of live Binance data recorded on 12 Oct 2026 (10:34 to 16:37 UTC). BTC, ETH and SOL, spot and perpetual, 20-level books every 100 ms plus every trade. That is 1.28 M events for BTC perp alone, sampled to a 1-second grid: **130,341 labelled rows**.

The task is to predict the 5-second mid move: down, flat or up. A move only counts when it beats half the spread, so a correct call could pay for crossing it. The last 20% of the time (26,069 rows) is a holdout that model selection never sees.

All numbers below come from `artifacts/reports/<run>/training.json` and the other JSON reports the CLI writes.

### Direction forecasts (holdout)

| Model | Log-loss skill | IC [95% CI] | Macro AUC | ECE |
|---|---|---|---|---|
| Logistic regression | 0.063 | 0.359 [0.324, 0.392] | 0.689 | 0.078 |
| LightGBM | 0.072 | 0.359 [0.322, 0.394] | 0.690 | 0.035 |
| XGBoost | **0.073** | 0.359 [0.322, 0.394] | **0.691** | 0.035 |
| LSTM | 0.070 | 0.329 [0.288, 0.366] | 0.677 | 0.046 |
| TCN | **0.073** | 0.340 [0.305, 0.372] | 0.682 | 0.054 |
| Transformer | 0.054 | 0.314 [0.277, 0.347] | 0.669 | 0.034 |
| Distilled MLP (2,467 params) | **0.079** | 0.354 [0.316, 0.388] | 0.687 | 0.041 |

- **Log-loss skill** is 1 − model log-loss ÷ baseline log-loss. The baseline is each instrument's own class mix from training. A single pooled baseline would credit a model just for recognising the instrument, because BTC has far more "flat" labels than SOL.
- **IC** is the rank correlation between p(up) − p(down) and the realised return. The 95% interval comes from a block bootstrap.
- **ECE** is expected calibration error.

What this shows:

- The signal is real and stable. The IC is about 0.36 on the holdout, and every family beats its baseline in all five CV folds.
- Most of the edge is in the features, not the model. Logistic regression is within about 0.01 skill of the best model.
- The CV pick did not survive the holdout. The transformer ranked first in CV (skill 0.097 vs 0.093 to 0.095 for the next four, all within one fold standard deviation) and came last on the holdout. The report keeps that visible instead of re-picking after the fact.
- Distillation helped. The 2,467-parameter MLP student trained on the transformer's soft labels beats its teacher on the holdout (0.079 vs 0.054) at 1/27 of the latency. After 50% pruning it keeps 0.067.

### Leakage checks

| Check | Result |
|---|---|
| Shifted-label retrain (labels rotated ≥ 25% of the series per instrument) | skill 0.005, limit 0.01: **pass** |
| Feature scan (no feature with \|IC\| > 0.5 against the forward return) | none flagged: **pass** |
| Purge and embargo overlap, all 5 folds | **pass** |

### Volatility (30-second realised vol, QLIKE, lower is better)

| Naive (training mean) | HAR (Corsi) | LightGBM |
|---|---|---|
| 1.069 | 0.847 | **0.632** |

### Lead-lag (transfer entropy)

Transfer entropy measures how much one instrument's last move tells you about another's next move, beyond what the target's own last move already tells you. It is asymmetric, so it shows who leads. Each value is compared with 200 circular-shift surrogates, and the Benjamini-Hochberg false-discovery rate is controlled across all 150 pair-lag tests.

| Pair | Lag | Excess TE (millibits) | Reverse direction |
|---|---|---|---|
| ETH perp → ETH spot | 1 s | **37.4** | 11.3 at 2 s |
| BTC perp → BTC spot | 1 s | **32.8** | 10.0 at 2 s |
| SOL spot → SOL perp | 1 s | 14.0 | 13.6 |

The perpetuals lead spot by about a second, at roughly 3× the information flow in the other direction. SOL is the exception: there the flow is symmetric. That supports the cross-asset features (`leader_return_*`, `basis_bps`). Full table: `reports/<dataset>/lead_lag.json`.

### Backtest and execution

**The forecast is real, but it cannot be traded directionally with market orders at retail fees.** That is the main finding of the backtest:

| Instrument | Largest expected 5 s move (holdout) | 99th pct realised 5 s move | Taker fee |
|---|---|---|---|
| BTC / ETH / SOL perp | 1.07 / 1.30 / 1.33 bps | 3.4 / 4.0 / 4.6 bps | 5 bps |
| BTC / ETH / SOL spot | 1.04 / 1.32 / 1.45 bps | 3.2 / 3.9 / 4.6 bps | 10 bps |

The convex sizer charges each instrument's live crossing cost (half-spread plus taker fee), so on the 1.2-hour holdout it correctly makes **no trades (net 0.0 bps)**. An earlier version charged a flat 1 bps per trade instead. It made 77 trades, roughly broke even before costs (−0.06 bps), and **lost 16.0 bps after costs**. A cost mis-specification turns a correct forecast into a losing strategy.

So the useful edge at this horizon is in *how* to trade, not *whether* to. The execution study asks this: buy 10 lots of ETH perp within 60 seconds. What does each policy cost against the arrival mid, fees included? It runs on 127 out-of-sample episodes after the PPO agent trains on the first 70% of the day.

| Policy | Mean shortfall (bps) | ± s.e. | Passive fill share |
|---|---|---|---|
| Cross the spread at once | 5.19 | 0.09 | 0% |
| TWAP (one lot every 6 s) | 4.80 | 0.23 | 0% |
| PPO agent (RL) | 3.70 | 0.42 | 20% |
| Rest at best bid, cross at deadline | **2.13** | 0.20 | 97% |

Resting passively saves about 3 bps on every lot against crossing. The PPO agent beats both naive baselines, but not the simple passive rule. Its training curve was still falling (3.75 → 2.76 bps in-sample) when the episode budget ran out, so it needs more episodes or a better start before it can win.

The fill-probability model (will a resting order fill within 10 s?) ranks fills with an AUC of 0.67 on 50,126 samples, and 86% of queue depletion comes from cancellations, not trades. Its Brier score (0.222) is not yet better than predicting the 72% base rate (0.217), so it is reported but not used for decisions.

### C++ hot path

The order book, the 27 per-instrument features and the MLP student are ported to header-only C++20. A replay harness checks them against the Python reference on the same recorded events:

| | Python | C++ |
|---|---|---|
| Feature update p50, laptop on mains (Oct 8 sample, 48 k events) | 86.9 µs | **1.8 µs** |
| Feature update p50, laptop on battery (full day, 1.28 M events) | 422 µs | 9.5 µs |
| Order book update (battery run) | | 0.49 µs |
| MLP student inference p50 (battery run) | ONNX Runtime 70 µs | 5.7 µs |
| Parity (all 251,386 feature rows × 27 features) | | max abs error 3.5e-12: **pass** |

Speed-up on feature updates: **45 to 48×** in both runs. The absolute times depend on the laptop's power state; the ratio is the number to compare.

## Architecture

```
 venue feeds                NATS JetStream (md.*, feat.*, pred.*, decision.*, alert.*)
 ───────────                ─────────────────────────────────────────────────────────
 Binance spot ┐
 Binance perp ├─► recorder ──► md.book.<inst> / md.trade.<inst> ──┬─► archiver ──► Parquet lake ──► research CLI
 Upstox (NSE) ┘   (adapters)                                      │                                (dataset, train,
                                                                  ├─► featurizer ──► feat.*        backtest, ...)
                                                                  │        │                            │
                                                                  │        ▼                            ▼
                                                                  │    predictor ◄── model bundle ◄── registry
                                                                  │   (regime, bandit,  (sha256)      (Postgres)
                                                                  │    calibrated)
                                                                  │        │ pred.*
                                                                  └─► strategy (convex sizing, paper fills only)
                                                                           │ decision.*
                                         monitor (drift PSI, live IC, clock skew, staleness) ──► alert.*
                                         gateway (FastAPI: JWT + roles, REST + WebSocket)
 Every hop: OpenTelemetry trace context in message headers ──► Jaeger;  metrics ──► Prometheus
```

- **One feature engine.** Research replay and the live featurizer run the same `FeatureEngine` on the same event order, so offline and online values cannot drift apart. The feature-set id is a hash of its configuration, and a model bundle refuses rows from a different feature set.
- **Delivery.** Publishes are de-duplicated by event id, consumers are durable pull consumers with explicit acks, failures retry with backoff and then go to a dead-letter stream. The archiver acks only after Parquet is on disk.
- **Lineage.** Every prediction row stores the bundle id, feature set, git commit, source tick id and trace id. Bundles are content-addressed by sha256 and verified before they are unpickled.
- **Security.** Each service has its own NATS user with least-privilege subjects. Postgres has separate owner (migrations) and app (data) roles. Gateway API keys are stored only as scrypt hashes and exchanged for short-lived JWTs with viewer, operator and admin roles. Every privileged action is written to an audit log. There is no code path that sends a real order.
- **Patterns.** Ports and adapters for venues, Strategy and Factory for models and cost models, Repository for SQL, a shared service lifecycle with supervised background tasks and graceful shutdown.

## Research protocol

`niftypulse research train` runs, in order:

1. Split off the final 20% of time as a holdout, behind a purge gap.
2. Purged, embargoed walk-forward CV (5 folds) for every family: logistic regression, LightGBM, XGBoost, LSTM, TCN, transformer. Within each fold, the HMM regime model is fitted on training rows only, and the training block is split by time into fit, early-stopping and calibration slices.
3. Leakage checks: shifted labels, a suspicious-feature scan and fold overlap.
4. Selection on mean CV log-loss skill only. Then every family is scored once on the holdout.
5. Calibration: a temperature plus per-class bias, fitted on held-out data.
6. Regime features, all causal: a Kalman local-linear trend, Bayesian online change-point detection, and HMM forward-filter probabilities (not Viterbi, which uses the future).
7. Volatility: HAR baseline against LightGBM, scored by QLIKE.
8. Compression: distil the best sequence model into an MLP, prune 50%, quantise to INT8 ONNX, and export weights for C++. Accuracy, size and latency are measured at each step.
9. Bundle the top three arms with their calibrators, the HMM, priors and drift reference bins, and register it in Postgres.

Live, a Thompson-sampling bandit chooses among the bundled arms per regime. It learns from each prediction's log-score gain over the class prior once the label is known.

## Layout

```
src/niftypulse/
  core/          settings, NATS bus, telemetry, database
  domain/        events and instruments
  marketdata/    venue adapters (Binance, Upstox), order book, Parquet tick lake
  features/      microstructure features, feature engine, labels, dataset builder
  models/        model registry, linear, GBM, sequence nets, calibration, volatility,
                 fill probability, distillation
  regime/        Kalman, BOCPD, HMM, causal regime features
  policy/        Thompson bandit, convex sizer, PPO execution agent
  backtest/      venue cost models, tape simulator, signal backtest
  validation/    purged CV, leakage checks, metrics with bootstrap CIs
  research/      training protocol, workflows, model bundles, lead-lag, C++ harness
  services/      recorder, archiver, featurizer, predictor, strategy, monitor, gateway
  storage/       schema, repositories, database bootstrap
cpp/             header-only C++20 order book, features and MLP; replay benchmark; pybind11 module
migrations/      Alembic
ops/             NATS, Prometheus and Jaeger config templates, devstack.ps1
tests/           unit tests, including C++/Python parity
```

## Running it

The stack runs natively: NATS, PostgreSQL, Jaeger and Prometheus as portable binaries, no containers. Every path, port and secret comes from `.env`. Copy `.env.example` and fill it in.

```powershell
uv sync --all-extras --dev
./ops/devstack.ps1 up infra                 # NATS, Postgres, Jaeger, Prometheus
uv run niftypulse db-bootstrap; uv run alembic upgrade head
uv run niftypulse bootstrap                 # JetStream streams
./ops/devstack.ps1 up recorder archiver     # start recording to the lake

uv run niftypulse research dataset   --start 2026-10-12 --end 2026-10-12
uv run niftypulse research lead-lag  --dataset <id>
uv run niftypulse research train     --dataset <id> --horizon 5
uv run niftypulse research backtest  --bundle <bundle> --start 2026-10-12 --end 2026-10-12
uv run niftypulse research execution --bundle <bundle> --dataset <id> --instrument "BINANCE_USDM|ETHUSDT" --start 2026-10-12 --end 2026-10-12
uv run niftypulse research native-bench --binary build/replay_bench.exe --instrument "BINANCE_USDM|BTCUSDT" --start 2026-10-12 --end 2026-10-12

uv run niftypulse promote <bundle> --actor <you>
./ops/devstack.ps1 up featurizer predictor strategy monitor gateway
```

Build the C++ replay benchmark with CMake (`cmake -S cpp -B build && cmake --build build`), or with the bundled Zig toolchain on Windows: `python -m ziglang c++ -std=c++20 -O3 -Icpp/include cpp/src/replay_bench.cpp -o build/replay_bench.exe`.

CI runs ruff, mypy, pytest, gitleaks, pip-audit, bandit, and the C++ build plus parity test on Linux and Windows.

## Status and limits

- **NSE feed.** The Upstox login and the daily instrument-universe resolver (front-month futures, nearest-expiry options within N strikes) are done. The live market-data WebSocket adapter for NSE is next, once the broker account is reactivated. The NSE cost model (brokerage, STT, exchange charges, SEBI fee, GST, stamp duty) is already in `backtest/costs.py`.
- **Sample size.** Six hours from one day is enough to separate signal from noise at a 5-second horizon (holdout IC intervals exclude zero by a wide margin). It is not enough to claim the result holds across regimes or days.
- **Fill model.** Fills come from level-2 data, which has no order ids, so queue position and cancellations are approximated (see `models/fill_probability.py`). Real queue dynamics will differ.
