"""Opt-in process-wide garbage-collector policy for operational low-latency runs."""

import gc
import os
import threading
import time
from typing import Callable, Final, Literal, NamedTuple

from .error import PrerequisiteError

GcMode = Literal["default", "freeze", "low_latency"]

GEN2_NEVER: Final = 2**31 - 1
_MODE_ENV: Final = "INSPECT_GC_MODE"
_GEN2_THRESHOLD_ENV: Final = "INSPECT_GC_GEN2_THRESHOLD"
_HIGH_WATER_ENV: Final = "INSPECT_GC_GUARD_HIGH_WATER"
_DEFAULT_HIGH_WATER: Final = 0.75
_GUARD_INTERVAL_SECONDS: Final = 30.0
_GUARD_MIN_COLLECT_INTERVAL_SECONDS: Final = 120.0
_CGROUP_LIMIT_PATHS: Final = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)

_configured = False
_guard_thread: threading.Thread | None = None


class GcConfiguration(NamedTuple):
    """The collector policy applied to this process."""

    mode: GcMode
    thresholds_before: tuple[int, int, int]
    thresholds_after: tuple[int, int, int]
    frozen_objects: int
    guard_started: bool


class MemoryGuard(NamedTuple):
    """State needed to enforce the memory high-water policy."""

    memory_limit_bytes: int
    high_water: float
    last_collect: float


class GuardDependencies(NamedTuple):
    """Injectable operations for one memory-guard tick."""

    process_rss_bytes: Callable[[], int]
    collect_generation_two: Callable[[], int]
    emit: Callable[[str], None]
    monotonic: Callable[[], float]


def gc_mode() -> GcMode:
    """Read the requested operational GC policy."""
    value = os.environ.get(_MODE_ENV, "").strip().lower()
    match value:
        case "" | "default":
            return "default"
        case "freeze":
            return "freeze"
        case "low_latency":
            return "low_latency"
        case _:
            raise PrerequisiteError(
                f"Invalid {_MODE_ENV}: {value!r}. Valid values are default, freeze, low_latency."
            )


def _high_water() -> float:
    value = os.environ.get(_HIGH_WATER_ENV, "").strip()
    if not value:
        return _DEFAULT_HIGH_WATER
    try:
        high_water = float(value)
    except ValueError:
        raise PrerequisiteError(
            f"Invalid {_HIGH_WATER_ENV}: {value!r} is not a number."
        ) from None
    if not 0.0 < high_water < 1.0:
        raise PrerequisiteError(
            f"Invalid {_HIGH_WATER_ENV}: {value!r} must be between 0 and 1."
        )
    return high_water


def _gen2_threshold() -> int:
    value = os.environ.get(_GEN2_THRESHOLD_ENV, "").strip()
    if not value:
        return GEN2_NEVER
    try:
        threshold = int(value)
    except ValueError:
        raise PrerequisiteError(
            f"Invalid {_GEN2_THRESHOLD_ENV}: {value!r} is not an integer."
        ) from None
    if not 0 < threshold <= GEN2_NEVER:
        raise PrerequisiteError(
            f"Invalid {_GEN2_THRESHOLD_ENV}: {value!r} must be between 1 and {GEN2_NEVER}."
        )
    return threshold


def process_rss_bytes() -> int:
    """Return this process's resident memory in bytes, or zero when unavailable."""
    try:
        with open("/proc/self/statm") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (IndexError, OSError, ValueError):
        return 0


def memory_limit_bytes() -> int | None:
    """Return the finite cgroup memory limit, if one is configured."""
    for path in _CGROUP_LIMIT_PATHS:
        try:
            with open(path) as limit_file:
                value = limit_file.read().strip()
        except OSError:
            continue
        if value == "max":
            continue
        try:
            limit = int(value)
        except ValueError:
            continue
        if 0 < limit < 2**62:
            return limit
    return None


def guard_tick(guard: MemoryGuard, dependencies: GuardDependencies) -> MemoryGuard:
    """Perform one testable high-water guard decision."""
    now = dependencies.monotonic()
    rss = dependencies.process_rss_bytes()
    if rss / guard.memory_limit_bytes < guard.high_water:
        return guard
    if now - guard.last_collect < _GUARD_MIN_COLLECT_INTERVAL_SECONDS:
        return guard

    collected = dependencies.collect_generation_two()
    elapsed = dependencies.monotonic() - now
    rss_after = dependencies.process_rss_bytes()
    dependencies.emit(
        "INSPECT_GC_GUARD forced full collection: "
        + f"rss_mb={rss // 2**20}/{guard.memory_limit_bytes // 2**20} "
        + f"({100 * rss / guard.memory_limit_bytes:.1f}%) collected={collected} "
        + f"elapsed={elapsed:.3f}s rss_after_mb={rss_after // 2**20}"
    )
    return guard._replace(last_collect=now)


def _emit(message: str) -> None:
    print(message, flush=True)


def _memory_guard_loop(guard: MemoryGuard) -> None:
    dependencies = GuardDependencies(
        process_rss_bytes=process_rss_bytes,
        collect_generation_two=lambda: gc.collect(2),
        emit=_emit,
        monotonic=time.monotonic,
    )
    while True:
        time.sleep(_GUARD_INTERVAL_SECONDS)
        guard = guard_tick(guard, dependencies)


def start_memory_guard(guard: MemoryGuard) -> bool:
    """Start the daemon that protects memory when full collections are suppressed."""
    global _guard_thread
    if _guard_thread is not None:
        return False
    _guard_thread = threading.Thread(
        target=_memory_guard_loop,
        args=(guard,),
        name="inspect-gc-guard",
        daemon=True,
    )
    _guard_thread.start()
    return True


def configure_gc() -> GcConfiguration | None:
    """Apply the requested opt-in GC policy once for this process."""
    global _configured
    mode = gc_mode()
    if _configured or mode == "default":
        return None

    thresholds_before = gc.get_threshold()
    threshold = thresholds_before[2]
    guard_started = False
    rss_description = "unavailable"
    limit_description = "unavailable"

    if mode == "low_latency":
        threshold = _gen2_threshold()
        high_water = _high_water()
        rss = process_rss_bytes()
        memory_limit = memory_limit_bytes()
        if rss <= 0 or memory_limit is None:
            raise PrerequisiteError(
                "INSPECT_GC_MODE=low_latency requires a positive RSS and finite cgroup memory limit."
            )
        rss_description = str(rss // 2**20)
        limit_description = str(memory_limit // 2**20)
        gc.set_threshold(thresholds_before[0], thresholds_before[1], threshold)
        gc.freeze()
        guard_started = start_memory_guard(
            MemoryGuard(memory_limit, high_water, last_collect=0.0)
        )
    else:
        gc.freeze()

    thresholds_after = gc.get_threshold()
    frozen_objects = gc.get_freeze_count()
    _configured = True
    print(
        f"INSPECT_GC mode={mode} thresholds={thresholds_before}->{thresholds_after} "
        + f"frozen={frozen_objects} rss_mb={rss_description} "
        + f"limit_mb={limit_description} guard={'on' if guard_started else 'off'}",
        flush=True,
    )
    return GcConfiguration(
        mode=mode,
        thresholds_before=thresholds_before,
        thresholds_after=thresholds_after,
        frozen_objects=frozen_objects,
        guard_started=guard_started,
    )
