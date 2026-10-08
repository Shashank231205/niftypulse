"""Operator entry point: `niftypulse <command>`."""

import asyncio

import typer

from niftypulse.core import telemetry
from niftypulse.core.bus import EventBus
from niftypulse.core.config import bus_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)


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
