"""Background watchdog that detects event loop stalls.

Wakes up at a fixed cadence and records whenever the actual wake-up
arrives later than expected — a proxy for sync I/O or other blocking
work that's pinning the loop. Useful both as a production watchdog
(logs stalls over a threshold) and as a test instrument (the yielded
stats expose the largest stall observed).

Usage:
    from inspect_ai._util.event_loop_monitor import event_loop_monitor

    async with event_loop_monitor():
        await do_work()

    # Custom cadence/threshold (seconds); inspect the stats afterwards:
    async with event_loop_monitor(interval=0.005, threshold=0.25) as stats:
        await do_work()
    assert stats.max_lateness_ms < 50

`interval` controls how often the watchdog checks the loop; `threshold`
is the lateness (in seconds) above which a warning is logged. Warnings
go to the module logger (`inspect_ai._util.event_loop_monitor`) at
WARNING level — ensure logging is configured to surface them.

Backend-agnostic (anyio), so it works under both asyncio and trio.
"""

import gc
import logging
import os
import sys
import threading
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import final

import anyio

from inspect_ai.util._anyio import inner_exception

logger = logging.getLogger(__name__)

LOOP_ATTRIBUTION_ENV = "INSPECT_LOOP_ATTRIBUTION"


def loop_attribution_enabled() -> bool:
    return os.getenv(LOOP_ATTRIBUTION_ENV) == "1"


@final
class _LoopStallAttributor:
    def __init__(self, interval: float, threshold: float) -> None:
        self._interval = interval
        self._threshold = threshold
        self._owner_thread_id = threading.get_ident()
        self._stop = threading.Event()
        # Protects the heartbeat shared by the event-loop and watchdog threads.
        self._heartbeat_lock = threading.Lock()
        self._heartbeat = time.monotonic()
        self._gc = _GCTracker()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def start(self) -> None:
        self._gc.start()
        self._thread.start()

    def pulse(self) -> None:
        with self._heartbeat_lock:
            self._heartbeat = time.monotonic()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self._gc.stop()

    def _watch(self) -> None:
        reported_heartbeat: float | None = None
        previous_gc_count = gc.get_count()
        previous_gc_gen2 = gc.get_stats()[2]
        while not self._stop.wait(self._interval):
            with self._heartbeat_lock:
                heartbeat = self._heartbeat
            elapsed = time.monotonic() - heartbeat
            if elapsed <= self._threshold or heartbeat == reported_heartbeat:
                continue
            frames = sys._current_frames()  # pyright: ignore[reportPrivateUsage]  # no public all-thread frame API
            frame = frames.get(self._owner_thread_id)
            if frame is None:
                continue
            now = time.monotonic()
            gc_overlap = self._gc.overlap(heartbeat, now)
            gc_count = gc.get_count()
            gc_gen2 = gc.get_stats()[2]
            print(
                "INSPECT_LOOP_ATTRIBUTION "
                + f"blocked_for={elapsed:.3f}s threshold={self._threshold:.3f}s "
                + f"stack={_format_stack(frame)}",
                flush=True,
            )
            for thread_line in _format_thread_stacks(frames, self._owner_thread_id):
                print(thread_line, flush=True)
            print(
                _format_gc_line(
                    gc_overlap,
                    gc_count,
                    _subtract_tuples(gc_count, previous_gc_count),
                    gc_gen2,
                    _subtract_gc_stats(gc_gen2, previous_gc_gen2),
                ),
                flush=True,
            )
            previous_gc_count = gc_count
            previous_gc_gen2 = gc_gen2
            reported_heartbeat = heartbeat


@dataclass(frozen=True)
class _GCOverlap:
    open: bool
    generation: int | None
    elapsed: float


@final
class _GCTracker:
    def __init__(self) -> None:
        # GC callbacks and the watchdog can run on different threads.
        self._lock = threading.Lock()
        self._open: tuple[int, float] | None = None
        self._completed: list[tuple[int, float, float]] = []

    def start(self) -> None:
        gc.callbacks.append(self._callback)

    def stop(self) -> None:
        gc.callbacks.remove(self._callback)

    def overlap(self, start: float, end: float) -> _GCOverlap:
        with self._lock:
            candidates = [event for event in self._completed if event[2] >= start]
            if self._open is not None:
                generation, opened_at = self._open
                candidates.append((generation, opened_at, end))
        if not candidates:
            return _GCOverlap(False, None, 0.0)
        generation, opened_at, closed_at = max(
            candidates, key=lambda event: (event[0], event[2] - event[1])
        )
        return _GCOverlap(True, generation, closed_at - opened_at)

    def _callback(self, phase: str, info: dict[str, int]) -> None:
        now = time.monotonic()
        generation = info["generation"]
        with self._lock:
            if phase == "start":
                self._open = (generation, now)
            elif phase == "stop" and self._open is not None:
                opened_generation, opened_at = self._open
                self._completed.append((opened_generation, opened_at, now))
                self._completed = self._completed[-16:]
                self._open = None


def _format_thread_stacks(
    frames: dict[int, FrameType], owner_thread_id: int
) -> list[str]:
    threads = {
        thread.ident: thread
        for thread in threading.enumerate()
        if thread.ident is not None
    }
    return [
        "INSPECT_LOOP_ATTRIBUTION_THREAD "
        + f"thread_id={thread_id} name={thread.name if thread else 'unknown'} "
        + f"daemon={thread.daemon if thread else 'unknown'} "
        + f"loop_owner={thread_id == owner_thread_id} "
        + f"stack={_format_stack(frame, limit=8)}"
        for thread_id, frame in frames.items()
        for thread in [threads.get(thread_id)]
    ]


def _format_gc_line(
    overlap: _GCOverlap,
    count: tuple[int, int, int],
    count_delta: tuple[int, int, int],
    gen2_stats: dict[str, int],
    gen2_stats_delta: dict[str, int],
) -> str:
    generation = overlap.generation if overlap.generation is not None else "-"
    return (
        "INSPECT_LOOP_ATTRIBUTION_GC "
        f"gc_open={overlap.open} gc_gen={generation} gc_elapsed={overlap.elapsed:.3f}s "
        f"gc_count={count} gc_count_delta={count_delta} "
        f"gc_gen2={gen2_stats} gc_gen2_delta={gen2_stats_delta}"
    )


def _subtract_tuples(
    current: tuple[int, int, int], previous: tuple[int, int, int]
) -> tuple[int, int, int]:
    return (
        current[0] - previous[0],
        current[1] - previous[1],
        current[2] - previous[2],
    )


def _subtract_gc_stats(
    current: dict[str, int], previous: dict[str, int]
) -> dict[str, int]:
    return {name: value - previous[name] for name, value in current.items()}


def _format_stack(frame: FrameType, limit: int = 12) -> str:
    return " <- ".join(
        _format_frame(frame_summary)
        for frame_summary in traceback.extract_stack(frame)[-limit:]
    )


def _format_frame(frame_summary: traceback.FrameSummary) -> str:
    path = Path(frame_summary.filename).with_suffix("")
    for package in ("inspect_ai", "tests"):
        if package in path.parts:
            index = path.parts.index(package)
            module = ".".join(path.parts[index:])
            return f"{module}:{frame_summary.name}:{frame_summary.lineno}"
    return f"{path.stem}:{frame_summary.name}:{frame_summary.lineno}"


@dataclass
class EventLoopMonitorStats:
    """Observations collected by a running event-loop monitor."""

    max_lateness: float = 0.0
    """Largest observed tick lateness, in seconds."""

    stalls: int = 0
    """Number of ticks whose lateness exceeded the threshold."""

    @property
    def max_lateness_ms(self) -> float:
        return self.max_lateness * 1000


async def _monitor_loop(
    interval: float,
    threshold: float,
    stop: anyio.Event,
    stats: EventLoopMonitorStats,
    attribution: _LoopStallAttributor | None,
) -> None:
    next_tick = time.monotonic()
    while not stop.is_set():
        if attribution is not None:
            attribution.pulse()
        next_tick += interval
        delay = next_tick - time.monotonic()
        if delay > 0:
            await anyio.sleep(delay)
        # Measure lateness before honoring `stop`: a block that ends right
        # before teardown leaves us parked in the sleep above with `stop`
        # already set, and that final wake-up is exactly the stall we want
        # to record. Returning early here would discard it.
        lateness = time.monotonic() - next_tick
        if lateness > stats.max_lateness:
            stats.max_lateness = lateness
        if lateness > threshold:
            stats.stalls += 1
            logger.warning(
                "event loop blocked for ~%.0fms (threshold=%.0fms)",
                lateness * 1000,
                threshold * 1000,
            )
            # Reset baseline so a single stall doesn't generate
            # a cascade of catch-up warnings.
            next_tick = time.monotonic()
        if stop.is_set():
            return


@asynccontextmanager
async def event_loop_monitor(
    interval: float = 0.1,
    threshold: float = 1.0,
) -> AsyncIterator[EventLoopMonitorStats]:
    """Scoped event-loop monitor.

    Yields a stats object that is updated live as the monitored block
    runs and is final once the block exits.
    """
    stats = EventLoopMonitorStats()
    stop = anyio.Event()
    attribution_enabled = loop_attribution_enabled()
    attribution = (
        _LoopStallAttributor(interval, threshold) if attribution_enabled else None
    )
    if attribution is not None:
        attribution.start()
    try:
        async with anyio.create_task_group() as tg:
            _ = tg.start_soon(
                _monitor_loop, interval, threshold, stop, stats, attribution
            )
            try:
                yield stats
            finally:
                stop.set()
    except Exception as ex:
        # anyio task groups wrap body exceptions in an ExceptionGroup;
        # unwrap so callers' `except SpecificError` keeps working
        raise inner_exception(ex) from None
    finally:
        if attribution is not None:
            attribution.stop()
