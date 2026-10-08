"""Common lifecycle for every long-running service.

A service subclass implements `start` and `stop`; `run` handles telemetry setup,
signal-driven graceful shutdown and the order of teardown, so every process behaves
the same way under Ctrl+C, a supervisor stop or a crash.
"""

import asyncio
import contextlib
import signal
from abc import ABC, abstractmethod

from niftypulse.core import telemetry

log = telemetry.logger(__name__)


class Service(ABC):
    name: str

    def __init__(self) -> None:
        self._shutdown = asyncio.Event()

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    def request_shutdown(self) -> None:
        self._shutdown.set()

    async def _serve(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            # Windows event loops do not support add_signal_handler; KeyboardInterrupt
            # still reaches run() below.
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.request_shutdown)
        await self.start()
        log.info("service.started", service=self.name)
        try:
            await self._shutdown.wait()
        finally:
            log.info("service.stopping", service=self.name)
            await self.stop()
            log.info("service.stopped", service=self.name)

    def run(self) -> None:
        telemetry.setup(self.name)
        try:
            asyncio.run(self._serve())
        except KeyboardInterrupt:
            pass
        finally:
            telemetry.shutdown()
