"""Regression tests for MCPServerLocal session-cache isolation.

Before the fix, ``MCPServerLocal._task_sessions`` was a class-level dict
keyed on ``anyio.get_current_task().id`` (which is ``id(asyncio.current_task())``
— a memory address). Python recycles those addresses once tasks are GC'd,
so a later sample in an ``eval_set`` run could inherit the previous
sample's ``MCPServerLocalSession`` — including its cached tool list —
producing "Tool not found" errors when the tool sets differed.

These tests don't spawn real subprocesses; they exercise the session
bookkeeping directly so the isolation guarantees are easy to verify.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock

import anyio
import pytest

from inspect_ai.log._samples import _sample_active
from inspect_ai.tool._mcp._local import MCPServerLocal, MCPServerLocalSession


def _make_server(name: str = "s") -> MCPServerLocal:
    # _client is only called when a real session connects; these tests never
    # reach that path, so a placeholder is sufficient.
    def _unused_client() -> Any:
        raise AssertionError("client should not be invoked in this test")

    return MCPServerLocal(client=_unused_client, name=name, events=False)


async def test_task_sessions_are_instance_level() -> None:
    """Two MCPServerLocal instances share no session state.

    Holds even on the same anyio task (i.e. same task id).
    """
    a = _make_server("a")
    b = _make_server("b")

    sa = a._task_session()
    sb = b._task_session()

    # Different instances → different sessions, regardless of shared task id.
    assert sa is not sb
    # Each instance owns its own table; class has none.
    assert "_task_sessions" not in MCPServerLocal.__dict__
    assert sa in a._task_sessions.values()
    assert sb in b._task_sessions.values()

    # And within one instance, repeated calls on the same task reuse the session.
    assert a._task_session() is sa


async def test_task_sessions_independent_across_instances_with_same_name() -> None:
    """Two instances sharing the same ``name`` don't collide.

    Sharing a name is a common pattern when the same Task builder runs for
    every sample.
    """
    a = _make_server("same-name")
    b = _make_server("same-name")

    sa = a._task_session()
    sb = b._task_session()

    assert sa is not sb


async def test_cached_tool_list_cleared_on_close() -> None:
    """A session whose refcount drops to zero must clear its cached tool list.

    Otherwise a reused session object would serve stale tools to the next
    caller without contacting the server.
    """
    session = MCPServerLocalSession(
        client=lambda: (_ for _ in ()).throw(AssertionError("unused")),
        name="s",
        events=False,
    )

    # Simulate an active session with a populated cache.
    sentinel_tools: list[Any] = [object()]
    session._cached_tool_list = sentinel_tools
    session._refcount = 1
    fake_exit_stack = AsyncMock()
    fake_exit_stack.aclose = AsyncMock(return_value=None)
    session._exit_stack = fake_exit_stack
    session._session = (
        AsyncMock()
    )  # any non-None placeholder satisfying ClientSession | None

    await session.__aexit__(None, None, None)

    assert session._refcount == 0
    assert session._session is None
    assert session._exit_stack is None
    assert session._cached_tool_list is None, (
        "cache must be dropped on close so a reused session re-fetches tools"
    )


async def test_cached_tool_list_preserved_when_still_referenced() -> None:
    """Closing one holder must NOT invalidate the cache while others remain.

    If the session is still held by another caller (refcount > 0), the
    remaining holder still expects its tools.
    """
    session = MCPServerLocalSession(
        client=lambda: (_ for _ in ()).throw(AssertionError("unused")),
        name="s",
        events=False,
    )
    sentinel_tools: list[Any] = [object()]
    session._cached_tool_list = sentinel_tools
    session._refcount = 2  # two holders

    await session.__aexit__(None, None, None)

    assert session._refcount == 1
    assert session._cached_tool_list is sentinel_tools


class _FakeActiveSample:
    """Stand-in for ``ActiveSample``: only identity and ``completed`` matter."""

    def __init__(self) -> None:
        self.completed: float | None = None

    def complete(self) -> None:
        self.completed = 1.0


@contextlib.contextmanager
def _active(sample: object | None) -> Iterator[None]:
    """Bind ``sample_active()`` for the duration of the block."""
    token = _sample_active.set(sample)  # type: ignore[arg-type]  # duck-typed stand-in, not a real ActiveSample  # pyright: ignore[reportArgumentType]  # duck-typed stand-in, not a real ActiveSample
    try:
        yield
    finally:
        _sample_active.reset(token)


async def test_child_tasks_of_a_sample_share_one_session() -> None:
    """The churn fix: tasks under one sample must not each launch a server.

    The agent bridge serves every tool call from its own task; keyed on task
    id those all missed the table and started another MCP server process.
    """
    server = _make_server()
    sample = _FakeActiveSample()
    child_sessions: list[object] = []

    async def child() -> None:
        child_sessions.append(server._task_session())

    with _active(sample):
        parent_session = server._task_session()
        async with anyio.create_task_group() as tg:
            for _ in range(5):
                tg.start_soon(child)

    assert child_sessions, "children never ran"
    assert all(session is parent_session for session in child_sessions)
    assert len(server._task_sessions) == 1


async def test_concurrent_samples_get_distinct_sessions() -> None:
    """Isolation guarantee: two live attempts never share a session."""
    server = _make_server()
    first, second = _FakeActiveSample(), _FakeActiveSample()

    with _active(first):
        first_session = server._task_session()
    with _active(second):
        second_session = server._task_session()

    assert first_session is not second_session
    assert len(server._task_sessions) == 2


async def test_same_sample_id_across_epochs_does_not_collide() -> None:
    """Distinct attempt objects, even with identical sample id/epoch data."""
    server = _make_server()
    epoch_one, epoch_two = _FakeActiveSample(), _FakeActiveSample()

    with _active(epoch_one):
        session_one = server._task_session()
    with _active(epoch_two):
        session_two = server._task_session()

    assert session_one is not session_two


async def test_completed_sample_does_not_reuse_its_session() -> None:
    """A task outliving its sample must not resurrect a dead sandbox's server.

    It falls back to task scope, so it gets a fresh session object (which will
    fail loudly against the torn-down sandbox) rather than silently reusing the
    completed sample's.
    """
    server = _make_server()
    sample = _FakeActiveSample()

    with _active(sample):
        live_session = server._task_session()
        sample.complete()
        after_completion = server._task_session()

    assert after_completion is not live_session


async def test_scope_falls_back_to_task_outside_a_sample() -> None:
    """No active sample (bare mcp_connection in tooling/tests) keeps task scope."""
    server = _make_server()
    with _active(None):
        first = server._task_session()
        second = server._task_session()
        assert first is second

        other_task_session: list[object] = []

        async def child() -> None:
            other_task_session.append(server._task_session())

        async with anyio.create_task_group() as tg:
            tg.start_soon(child)

    assert other_task_session[0] is not first


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
