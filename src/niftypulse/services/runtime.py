"""Common lifecycle for every long-running service.

A service subclass implements `start` and `stop`; `run` handles telemetry setup,
signal-driven graceful shutdown and the order of teardown, so every process behaves
the same way under Ctrl+C, a supervisor stop or a crash. Background work started with
`spawn` is supervised: if it dies, the service stops and exits non-zero instead of
running on half-broken.
"""

import asyncio
import contextlib
import signal
import sys
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from typing import Any

from niftypulse.core import telemetry

log = telemetry.logger(__name__)


class Service(ABC):
    name: str

    def __init__(self) -> None:
        self._shutdown = asyncio.Event()
        self._failure: BaseException | None = None
        self._tasks: list[asyncio.Task[None]] = []

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    def request_shutdown(self) -> None:
        self._shutdown.set()

    def fail(self, exc: BaseException) -> None:
        if self._failure is None:
            self._failure = exc
            log.error("service.failed", service=self.name, error=repr(exc))
        self._shutdown.set()

    def spawn(self, coro: Coroutine[Any, Any, None], name: str) -> None:
        task = asyncio.create_task(coro, name=name)
        self._tasks.append(task)

        def _done(t: asyncio.Task[None]) -> None:
            if not t.cancelled() and t.exception() is not None:
                self.fail(t.exception())  # type: ignore[arg-type]

        task.add_done_callback(_done)

    async def _cancel_spawned(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _serve(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            # Windows event loops do not support add_signal_handler; KeyboardInterrupt
            # still reaches run() below.
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.request_shutdown)
        try:
            await self.start()
            log.info("service.started", service=self.name)
            await self._shutdown.wait()
        except Exception as exc:
            self.fail(exc)
        finally:
            log.info("service.stopping", service=self.name)
            await self._cancel_spawned()
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
        if self._failure is not None:
            sys.exit(1)
