"""Operator entry point: `niftypulse <command>`."""

import asyncio
import webbrowser
from collections import Counter

import typer

from niftypulse.core import telemetry
from niftypulse.core.bus import EventBus
from niftypulse.core.config import broker_settings, bus_settings, universe_settings
from niftypulse.marketdata.instrument_master import InstrumentMaster
from niftypulse.marketdata.upstox_session import AuthError, UpstoxSession

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def login() -> None:
    """Daily Upstox login. Opens the browser, then asks for the URL it lands on."""
    session = UpstoxSession(broker_settings())
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
        f"Logged in as {token.user_id}. Token valid until {token.expires_at:%d %b %H:%M} IST.",
        fg=typer.colors.GREEN,
    )


@app.command()
def universe() -> None:
    """Resolve and print today's subscription universe."""

    async def _run() -> None:
        token = UpstoxSession(broker_settings()).load()
        master = InstrumentMaster(broker_settings(), universe_settings(), token.access_token)
        try:
            u = await master.resolve()
        finally:
            await master.close()
        counts = Counter(i.kind.value for i in u.instruments)
        typer.echo(f"Trading day {u.trading_day}: {len(u.instruments)} instruments {dict(counts)}")
        for i in u.instruments:
            if i.kind.value != "call" and i.kind.value != "put":
                typer.echo(f"  {i.kind.value:<7} {i.symbol:<28} {i.key}")
        options = [i for i in u.instruments if i.kind.value in ("call", "put")]
        if options:
            strikes = sorted({i.strike or 0.0 for i in options})
            typer.echo(
                f"  options {options[0].underlying} expiry {options[0].expiry}, "
                f"strikes {strikes[0]:.0f}..{strikes[-1]:.0f} ({len(strikes)} strikes)"
            )

    try:
        asyncio.run(_run())
    except AuthError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


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


if __name__ == "__main__":
    app()
