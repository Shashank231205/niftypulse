"""Operator entry point: `niftypulse <command>`."""

import asyncio
import webbrowser
from collections import Counter
from collections.abc import Callable

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

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _process_setup() -> None:
    """Verify TLS against the OS trust store, so corporate or antivirus HTTPS inspection
    roots that Windows trusts are honoured without ever disabling verification."""
    truststore.inject_into_ssl()


def _services() -> dict[str, Callable[[], None]]:
    from niftypulse.services import archiver, recorder

    return {"recorder": recorder.main, "archiver": archiver.main}


@app.command()
def run(service: str) -> None:
    """Run one long-lived service (recorder, archiver, ...)."""
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
