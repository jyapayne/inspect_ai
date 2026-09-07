"""`_sample_resources_cm` teardown must not violate anyio cancel-scope order.

A `SampleResource` like `mcp_connection()` opens a task group (a cancel scope)
when entered. The teardown path originally wrapped `exit_stack.aclose()` in a
NEW `CancelScope(shield=True)` entered inside the `finally` — i.e. *after* the
resources' own scopes. anyio requires scopes to exit in reverse entry order, so
closing the resource's task group from inside that younger scope raised
"Attempted to exit a cancel scope that isn't the current tasks's current cancel
scope" and errored the sample after the agent had finished (observed on hawk
staging: epoch failed at cleanup with the full run complete).

These tests run the CM with a task-group-bearing resource — the exact shape of
`mcp_connection` over a sandbox MCP server — through both the normal exit and
the cancelled-sample exit.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import anyio

from inspect_ai._eval.task.run import _sample_resources_cm

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from inspect_ai.solver._task_state import TaskState


def _task_group_resource(events: list[str]):
    """A SampleResource whose context holds an open task group, like mcp_connection."""

    @contextlib.asynccontextmanager
    async def resource(_state: "TaskState") -> "AsyncIterator[None]":
        async with anyio.create_task_group() as tg:
            tg.start_soon(anyio.sleep_forever)
            try:
                yield
            finally:
                # Resources with must-run teardown shield it themselves,
                # as the sandbox MCP kill-server path does.
                with anyio.move_on_after(2, shield=True):
                    await anyio.lowlevel.checkpoint()
                    events.append("teardown")
                tg.cancel_scope.cancel()

    return resource


async def test_sample_resources_teardown_normal_exit() -> None:
    events: list[str] = []
    async with _sample_resources_cm([_task_group_resource(events)], None):  # type: ignore[arg-type]  # resource() ignores state; a real TaskState isn't needed
        await anyio.sleep(0.01)
    assert events == ["teardown"]


async def test_sample_resources_teardown_on_sample_cancel() -> None:
    events: list[str] = []
    entered = anyio.Event()

    async def sample_body() -> None:
        async with _sample_resources_cm([_task_group_resource(events)], None):  # type: ignore[arg-type]  # resource() ignores state; a real TaskState isn't needed
            entered.set()
            await anyio.sleep_forever()

    async with anyio.create_task_group() as tg:
        tg.start_soon(sample_body)
        await entered.wait()
        tg.cancel_scope.cancel()

    assert events == ["teardown"]
