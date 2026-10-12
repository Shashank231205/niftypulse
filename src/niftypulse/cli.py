"""Operator entry point: `niftypulse <command>`."""

import asyncio
import webbrowser
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING

import truststore
import typer

from niftypulse.core import telemetry
from niftypulse.core.bus import EventBus
from niftypulse.core.config import (
    broker_settings,
    bus_settings,
    market_settings,
    nse_universe_settings,
)
from niftypulse.marketdata.upstox_instruments import InstrumentMaster
from niftypulse.marketdata.upstox_session import AuthError, UpstoxSession

if TYPE_CHECKING:
    from niftypulse.research.workflows import Period

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _process_setup() -> None:
    """Verify TLS against the OS trust store, so corporate or antivirus HTTPS inspection
    roots that Windows trusts are honoured without ever disabling verification."""
    truststore.inject_into_ssl()


def _services() -> dict[str, Callable[[], None]]:
    from niftypulse.services import archiver, featurizer, monitor, predictor, recorder, strategy
    from niftypulse.services.gateway import app as gateway

    return {
        "recorder": recorder.main,
        "archiver": archiver.main,
        "featurizer": featurizer.main,
        "predictor": predictor.main,
        "strategy": strategy.main,
        "monitor": monitor.main,
        "gateway": gateway.main,
    }


@app.command()
def run(service: str) -> None:
    """Run one long-lived service (recorder, archiver, featurizer, predictor, ...)."""
    services = _services()
    if service not in services:
        typer.secho(f"unknown service {service!r}; one of {sorted(services)}", fg="red", err=True)
        raise typer.Exit(2)
    services[service]()


@app.command()
def bootstrap() -> None:
    """Create or update JetStream streams. Run once per environment, with admin credentials."""

    async def _run() -> None:
        bus = EventBus(bus_settings(), client_name="niftypulse-bootstrap")
        await bus.connect()
        try:
            await bus.ensure_streams()
        finally:
            await bus.close()

    telemetry.setup("bootstrap")
    try:
        asyncio.run(_run())
    finally:
        telemetry.shutdown()


research = typer.Typer(no_args_is_help=True, help="Datasets, training, backtests.")
app.add_typer(research, name="research")


def _period(start: str, end: str, instruments: list[str] | None) -> "Period":
    from datetime import date

    from niftypulse.research.workflows import Period
    from niftypulse.services.featurizer import live_instruments

    return Period(
        instruments or live_instruments(), date.fromisoformat(start), date.fromisoformat(end)
    )


@research.command("dataset")
def research_dataset(
    start: str = typer.Option(..., help="First UTC date, YYYY-MM-DD"),
    end: str = typer.Option(..., help="Last UTC date, YYYY-MM-DD"),
    instrument: list[str] = typer.Option(None, help="Instrument keys; default: live universe"),
) -> None:
    """Replay the tick lake through the feature engine and save a labelled dataset."""
    from niftypulse.research.workflows import build

    ds = build(_period(start, end, instrument))
    typer.echo(f"dataset {ds.dataset_id}: {len(ds.frame)} rows, feature set {ds.feature_set}")


@research.command("train")
def research_train(
    dataset: str = typer.Option(...),
    horizon: float = typer.Option(5.0, help="Label horizon in seconds"),
    family: list[str] = typer.Option(
        ["logistic", "lightgbm", "xgboost", "lstm", "tcn", "transformer"], help="Model families"
    ),
) -> None:
    """Run the full validation protocol and register the resulting bundle."""
    import json

    from niftypulse.research.workflows import load_dataset, train

    bundle, report = train(load_dataset(dataset), horizon, family)
    typer.echo(json.dumps({"bundle": bundle.bundle_id, "selection": report["selection"]}, indent=2))


@research.command("lead-lag")
def research_lead_lag(dataset: str = typer.Option(...)) -> None:
    """Transfer-entropy lead-lag between instruments, with surrogate significance."""
    import json

    from niftypulse.research.workflows import lead_lag, load_dataset

    out = lead_lag(load_dataset(dataset))
    typer.echo(json.dumps(out["strongest_lag_per_pair"], indent=2))


@research.command("backtest")
def research_backtest(
    bundle: str = typer.Option(...),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
) -> None:
    """Backtest the bundle's holdout predictions with costs."""
    import json

    from niftypulse.research.workflows import backtest

    typer.echo(
        json.dumps(backtest(bundle, _period(start, end, None))["summary"], indent=2, default=float)
    )


@research.command("execution")
def research_execution(
    bundle: str = typer.Option(...),
    dataset: str = typer.Option(...),
    instrument: str = typer.Option(...),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
) -> None:
    """Fill-probability model and PPO execution agent versus baselines."""
    import json
    from pathlib import Path

    from niftypulse.core.config import research_settings
    from niftypulse.research.bundle import load_bundle
    from niftypulse.research.workflows import execution_study, load_dataset

    report_dir = Path(research_settings().artifacts_root) / "reports" / bundle
    meta = json.loads((report_dir / "training.json").read_text())["bundle"]
    loaded = load_bundle(Path(meta["path"]), meta["sha256"])
    out = execution_study(
        loaded, load_dataset(dataset), instrument, _period(start, end, [instrument])
    )
    typer.echo(json.dumps(out["execution"], indent=2))


@research.command("native-bench")
def research_native_bench(
    binary: str = typer.Option(..., help="Path to the compiled replay_bench executable"),
    instrument: str = typer.Option(...),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    mlp: str = typer.Option(None, help="Optional .npmlp student weights for inference timing"),
) -> None:
    """Parity and latency of the C++ feature engine against the Python reference."""
    import json
    from datetime import date
    from pathlib import Path

    from niftypulse.core.config import feature_settings, feed_settings, lake_settings
    from niftypulse.core.config import research_settings as rs
    from niftypulse.marketdata.tick_store import TickReader
    from niftypulse.research.native import run as run_native

    report = run_native(
        Path(binary),
        TickReader(lake_settings().root),
        instrument,
        date.fromisoformat(start),
        date.fromisoformat(end),
        feature_settings(),
        feed_settings().publish_depth,
        Path(rs().artifacts_root) / "native",
        Path(mlp) if mlp else None,
    )
    (Path(rs().artifacts_root) / "native" / "report.json").write_text(json.dumps(report, indent=2))
    typer.echo(json.dumps(report, indent=2))


@app.command()
def promote(
    model_id: str, actor: str = typer.Option(..., help="Who is promoting (audited)")
) -> None:
    """Make a registered bundle the production model (audited)."""
    from niftypulse.core.config import db_settings
    from niftypulse.core.db import Database
    from niftypulse.storage.repositories import AuditRepository, ModelRepository

    async def _run() -> None:
        db = Database(db_settings())
        try:
            async with db.session() as s:
                repo = ModelRepository(s)
                if await repo.get(model_id) is None:
                    raise typer.BadParameter(f"unknown model {model_id}")
                await repo.promote(model_id, actor)
                await AuditRepository(s).record(
                    actor, "model.promote", model_id, "ok", None, None, {"via": "cli"}
                )
        finally:
            await db.close()

    asyncio.run(_run())
    typer.secho(f"{model_id} is now in production", fg="green")


@app.command("create-api-key")
def create_api_key(
    principal: str = typer.Option(..., help="Principal id, e.g. alice"),
    role: str = typer.Option("viewer", help="viewer | operator | admin"),
) -> None:
    """Create or rotate an API key. The key is printed once and stored only as a hash."""
    from niftypulse.core.config import db_settings
    from niftypulse.core.db import Database
    from niftypulse.services.gateway.auth import ROLES, new_api_key
    from niftypulse.storage.repositories import AuditRepository, PrincipalRepository

    if role not in ROLES or "." in principal:
        raise typer.BadParameter(
            "role must be viewer/operator/admin; principal must not contain '.'"
        )
    key, hashed = new_api_key(principal)

    async def _run() -> None:
        db = Database(db_settings())
        try:
            async with db.session() as s:
                await PrincipalRepository(s).upsert(principal, role, hashed)
                await AuditRepository(s).record(
                    "cli", "principal.key_rotated", principal, "ok", None, None, {"role": role}
                )
        finally:
            await db.close()

    asyncio.run(_run())
    typer.echo(f"API key for {principal} ({role}). Store it now; it is not shown again:")
    typer.echo(key)


@app.command("db-bootstrap")
def db_bootstrap() -> None:
    """Create the database, owner role and app role (superuser credentials, idempotent)."""
    from niftypulse.core.config import migration_settings, postgres_admin_settings
    from niftypulse.storage.bootstrap import bootstrap as run_bootstrap

    asyncio.run(run_bootstrap(postgres_admin_settings(), migration_settings()))
    typer.secho("Database and roles ready. Next: uv run alembic upgrade head", fg="green")


@app.command()
def login() -> None:
    """Daily Upstox login. Opens the browser, then asks for the URL it lands on."""
    session = UpstoxSession(broker_settings(), market_settings().tz)
    url, state = session.authorization_url()
    typer.echo("Opening the Upstox login page. If it does not open, visit:\n" + url)
    webbrowser.open(url)
    typer.echo(
        "\nAfter login the browser shows an error page. That is expected."
        "\nCopy the full address from the address bar and paste it below."
    )
    redirect = typer.prompt("Redirect URL")
    try:
        token = session.exchange(UpstoxSession.code_from_redirect(redirect, state))
    except AuthError as exc:
        typer.secho(f"Login failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(
        f"Logged in as {token.user_id}. Token valid until {token.expires_at:%d %b %H:%M %Z}.",
        fg=typer.colors.GREEN,
    )


@app.command("nse-universe")
def nse_universe() -> None:
    """Resolve and print today's NSE subscription universe (needs a valid Upstox login)."""

    async def _run() -> None:
        tz = market_settings().tz
        token = UpstoxSession(broker_settings(), tz).load()
        master = InstrumentMaster(
            broker_settings(), nse_universe_settings(), token.access_token, tz
        )
        try:
            u = await master.resolve()
        finally:
            await master.close()
        counts = Counter(i.kind.value for i in u.instruments)
        typer.echo(f"Trading day {u.trading_day}: {len(u.instruments)} instruments {dict(counts)}")
        for i in u.instruments:
            typer.echo(f"  {i.kind.value:<9} {i.symbol:<28} {i.key}")

    try:
        asyncio.run(_run())
    except AuthError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
