import threading
import time

import anyio
import pytest

from inspect_ai._util.event_loop_monitor import event_loop_monitor


async def test_event_loop_monitor_is_silent_for_idle_loop_when_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INSPECT_LOOP_ATTRIBUTION", "1")

    async with event_loop_monitor(interval=0.001, threshold=0.01):
        await anyio.sleep(0.03)

    assert capsys.readouterr().out == ""


async def test_event_loop_monitor_attributes_blocked_loop_when_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INSPECT_LOOP_ATTRIBUTION", "1")
    worker_started = threading.Event()
    worker_stop = threading.Event()

    def wait_for_stop() -> None:
        worker_started.set()
        _ = worker_stop.wait()

    worker = threading.Thread(
        name="attribution-test-worker", target=wait_for_stop, daemon=True
    )
    worker.start()
    assert worker_started.wait(timeout=1)

    try:
        async with event_loop_monitor(interval=0.001, threshold=0.01):
            await anyio.sleep(0.02)
            time.sleep(0.05)
            await anyio.sleep(0.04)
    finally:
        worker_stop.set()
        worker.join(timeout=1)

    stdout = capsys.readouterr().out
    assert "INSPECT_LOOP_ATTRIBUTION blocked_for=" in stdout
    assert (
        "test_event_loop_monitor:test_event_loop_monitor_attributes_blocked_loop_when_enabled:"
        in stdout
    )
    assert "INSPECT_LOOP_ATTRIBUTION_THREAD " in stdout
    assert "name=attribution-test-worker daemon=True loop_owner=False" in stdout
    assert "INSPECT_LOOP_ATTRIBUTION_GC gc_open=" in stdout
