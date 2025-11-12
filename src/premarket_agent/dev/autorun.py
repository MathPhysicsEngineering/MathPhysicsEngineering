from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
from typing import Iterable, List

try:
    from watchfiles import awatch
except Exception:  # pragma: no cover
    awatch = None  # type: ignore[assignment]


async def _spawn(cmd: List[str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


async def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            proc.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
        except Exception:
            pass


async def run_autoreload(command: str, paths: Iterable[str], debounce: float = 0.5) -> int:
    if awatch is None:
        print("watchfiles is not installed. Please install dev extras: pip install -e .[dev]")
        return 2

    cmd = command if isinstance(command, list) else command.split()
    proc = await _spawn(cmd)  # type: ignore[arg-type]

    async def _printer():
        if proc.stdout is None:
            return
        loop = asyncio.get_running_loop()
        while True:
            if proc.poll() is not None:
                break
            line = await asyncio.to_thread(proc.stdout.readline)
            if not line:
                break
            sys.stdout.write(line.decode(errors="ignore"))
            await asyncio.sleep(0)

    printer_task = asyncio.create_task(_printer())

    async for _changes in awatch(*paths, debounce=debounce):
        await _terminate(proc)
        proc = await _spawn(cmd)  # type: ignore[arg-type]
        if printer_task.done():
            printer_task = asyncio.create_task(_printer())

    await _terminate(proc)
    printer_task.cancel()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-reload the premarket GUI on code changes.")
    parser.add_argument(
        "--cmd",
        default="python -m premarket_agent.ui",
        help="Command to run and restart on changes",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=["src/premarket_agent"],
        help="Paths to watch for changes",
    )
    parser.add_argument("--debounce", type=float, default=0.5, help="Debounce seconds")
    args = parser.parse_args()
    try:
        asyncio.run(run_autoreload(args.cmd, args.paths, debounce=args.debounce))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()


