import gc
import threading
from collections.abc import Callable
from typing import ClassVar

import pytest

from inspect_ai import Task, eval
from inspect_ai._util import gc_tuning
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match


@pytest.fixture(autouse=True)
def restore_gc_state(monkeypatch: pytest.MonkeyPatch):
    thresholds = gc.get_threshold()
    for name in (
        "INSPECT_GC_MODE",
        "INSPECT_GC_GEN2_THRESHOLD",
        "INSPECT_GC_GUARD_HIGH_WATER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gc_tuning, "_configured", False)
    monkeypatch.setattr(gc_tuning, "_guard_thread", None)
    yield
    gc.unfreeze()
    gc.set_threshold(*thresholds)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "default"),
        ("default", "default"),
        ("freeze", "freeze"),
        ("low_latency", "low_latency"),
        ("  LOW_LATENCY  ", "low_latency"),
    ],
)
def test_gc_mode_parses_requested_policy(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: str
) -> None:
    if value is not None:
        monkeypatch.setenv("INSPECT_GC_MODE", value)

    assert gc_tuning.gc_mode() == expected


def test_gc_mode_rejects_unknown_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECT_GC_MODE", "aggressive")

    with pytest.raises(PrerequisiteError, match="INSPECT_GC_MODE"):
        _ = gc_tuning.gc_mode()


def test_default_mode_ignores_invalid_low_latency_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    thresholds = gc.get_threshold()
    frozen_objects = gc.get_freeze_count()
    monkeypatch.setenv("INSPECT_GC_GEN2_THRESHOLD", "not-an-integer")
    monkeypatch.setenv("INSPECT_GC_GUARD_HIGH_WATER", "not-a-float")

    assert gc_tuning.configure_gc() is None

    assert gc.get_threshold() == thresholds
    assert gc.get_freeze_count() == frozen_objects
    assert capsys.readouterr().out == ""


def test_freeze_mode_freezes_without_changing_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSPECT_GC_MODE", "freeze")
    thresholds = gc.get_threshold()

    configuration = gc_tuning.configure_gc()

    assert configuration is not None
    assert configuration.mode == "freeze"
    assert configuration.thresholds_before == thresholds
    assert configuration.thresholds_after == thresholds
    assert configuration.frozen_objects > 0
    assert configuration.guard_started is False


def test_low_latency_requires_usable_rss_and_cgroup_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSPECT_GC_MODE", "low_latency")
    monkeypatch.setattr(gc_tuning, "process_rss_bytes", lambda: 0)
    monkeypatch.setattr(gc_tuning, "memory_limit_bytes", lambda: None)

    with pytest.raises(PrerequisiteError, match="RSS and finite cgroup memory limit"):
        _ = gc_tuning.configure_gc()


def memory_guard_starts(guard: gc_tuning.MemoryGuard) -> bool:
    return guard.memory_limit_bytes > 0


def fixed_rss() -> int:
    return 512 * 2**20


def fixed_memory_limit() -> int:
    return 96 * 2**30


def test_low_latency_freezes_suppresses_gen2_and_starts_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSPECT_GC_MODE", "low_latency")
    monkeypatch.setattr(gc_tuning, "process_rss_bytes", fixed_rss)
    monkeypatch.setattr(gc_tuning, "memory_limit_bytes", fixed_memory_limit)
    monkeypatch.setattr(gc_tuning, "start_memory_guard", memory_guard_starts)
    thresholds = gc.get_threshold()

    configuration = gc_tuning.configure_gc()

    assert configuration is not None
    assert configuration.mode == "low_latency"
    assert configuration.thresholds_before == thresholds
    assert configuration.thresholds_after == (
        thresholds[0],
        thresholds[1],
        gc_tuning.GEN2_NEVER,
    )
    assert configuration.frozen_objects > 0
    assert configuration.guard_started is True


@pytest.mark.parametrize("value", ["0", "-1", str(2**31), "lots"])
def test_low_latency_rejects_invalid_gen2_threshold(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("INSPECT_GC_MODE", "low_latency")
    monkeypatch.setenv("INSPECT_GC_GEN2_THRESHOLD", value)

    with pytest.raises(PrerequisiteError, match="INSPECT_GC_GEN2_THRESHOLD"):
        _ = gc_tuning.configure_gc()


def test_low_latency_emits_one_startup_line_once_per_process(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INSPECT_GC_MODE", "low_latency")
    monkeypatch.setattr(gc_tuning, "process_rss_bytes", fixed_rss)
    monkeypatch.setattr(gc_tuning, "memory_limit_bytes", fixed_memory_limit)
    monkeypatch.setattr(gc_tuning, "start_memory_guard", memory_guard_starts)

    assert gc_tuning.configure_gc() is not None
    assert gc_tuning.configure_gc() is None

    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert "INSPECT_GC mode=low_latency" in output
    assert f"{gc_tuning.GEN2_NEVER}" in output
    assert "guard=on" in output
    assert "rss_mb=512" in output
    assert "limit_mb=98304" in output


def test_guard_tick_collects_above_high_water() -> None:
    emitted: list[str] = []
    collections: list[int] = []

    def process_rss() -> int:
        return 80 * 2**30

    def collect_generation_two() -> int:
        collections.append(2)
        return 17

    def monotonic() -> float:
        return 1000.0

    dependencies = gc_tuning.GuardDependencies(
        process_rss_bytes=process_rss,
        collect_generation_two=collect_generation_two,
        emit=emitted.append,
        monotonic=monotonic,
    )
    guard = gc_tuning.MemoryGuard(
        memory_limit_bytes=96 * 2**30,
        high_water=0.75,
        last_collect=0.0,
    )

    updated_guard = gc_tuning.guard_tick(guard, dependencies)

    assert collections == [2]
    assert updated_guard.last_collect == 1000.0
    assert len(emitted) == 1
    assert emitted[0].startswith("INSPECT_GC_GUARD forced full collection:")


def test_guard_tick_is_idle_below_high_water() -> None:
    emitted: list[str] = []

    def process_rss() -> int:
        return 40 * 2**30

    def unexpected_collection() -> int:
        pytest.fail("collector must remain idle")

    def monotonic() -> float:
        return 1000.0

    dependencies = gc_tuning.GuardDependencies(
        process_rss_bytes=process_rss,
        collect_generation_two=unexpected_collection,
        emit=emitted.append,
        monotonic=monotonic,
    )
    guard = gc_tuning.MemoryGuard(
        memory_limit_bytes=96 * 2**30,
        high_water=0.75,
        last_collect=0.0,
    )

    assert gc_tuning.guard_tick(guard, dependencies) == guard
    assert emitted == []


def test_guard_tick_respects_minimum_collection_interval() -> None:
    def process_rss() -> int:
        return 80 * 2**30

    def unexpected_collection() -> int:
        pytest.fail("collector must remain idle")

    def unexpected_emit(message: str) -> None:
        pytest.fail(f"unexpected output: {message}")

    def monotonic() -> float:
        return 100.0

    dependencies = gc_tuning.GuardDependencies(
        process_rss_bytes=process_rss,
        collect_generation_two=unexpected_collection,
        emit=unexpected_emit,
        monotonic=monotonic,
    )
    guard = gc_tuning.MemoryGuard(
        memory_limit_bytes=96 * 2**30,
        high_water=0.75,
        last_collect=50.0,
    )

    assert gc_tuning.guard_tick(guard, dependencies) == guard


def test_start_memory_guard_creates_daemon_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeThread:
        instances: ClassVar[list["FakeThread"]] = []

        def __init__(
            self,
            *,
            target: Callable[[gc_tuning.MemoryGuard], None],
            args: tuple[gc_tuning.MemoryGuard],
            name: str,
            daemon: bool,
        ) -> None:
            self.target: Callable[[gc_tuning.MemoryGuard], None] = target
            self.args: tuple[gc_tuning.MemoryGuard] = args
            self.name: str = name
            self.daemon: bool = daemon
            self.started: bool = False
            self.instances.append(self)

        def start(self) -> None:
            self.started = True

    monkeypatch.setattr(threading, "Thread", FakeThread)

    guard = gc_tuning.MemoryGuard(96 * 2**30, 0.75, last_collect=0.0)
    assert gc_tuning.start_memory_guard(guard) is True

    assert len(FakeThread.instances) == 1
    assert FakeThread.instances[0].daemon is True
    assert FakeThread.instances[0].target.__name__ == "_memory_guard_loop"
    assert FakeThread.instances[0].args == (guard,)
    assert FakeThread.instances[0].started is True


def test_eval_configures_gc_once_per_process(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INSPECT_GC_MODE", "freeze")
    calls: list[str] = []
    real_configure_gc = gc_tuning.configure_gc

    def counting_configure_gc() -> gc_tuning.GcConfiguration | None:
        calls.append("called")
        return real_configure_gc()

    monkeypatch.setattr("inspect_ai._eval.eval.configure_gc", counting_configure_gc)
    task = Task(dataset=[Sample(input="Say hello", target="hello")], scorer=match())
    _ = eval(task, model="mockllm/model")
    _ = eval(task, model="mockllm/model")

    assert calls == ["called", "called"]
    assert capsys.readouterr().out.count("INSPECT_GC mode=") == 1
