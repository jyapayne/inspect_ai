"""Tool closures must not each start their own MCP server.

`MCPToolSourceLocal.tools()` hands out `Tool` closures bound to the
`MCPServerLocalSession` that produced them. The sandbox agent bridge
materializes those closures ONCE (`bridge.py::_register_bridged_tools` builds
`{name: tool}` from `spec.tools`) and then invokes them per tool call, from its
own request tasks, long after the producing connection has exited.

When that happens `_client_session()` sees `self._session is None` and builds a
private client — starting a whole MCP server process per tool call, without
ever consulting the session table. Measured in-sandbox at 40 concurrent
samples: launches tracked tool calls (5 per sandbox for 4 calls) with ZERO
server exits, because each per-call client stack unwinds only when the call
returns.

These tests drive real stdio servers and count process starts, in the bridge's
shape: resolve tools inside a connection, leave it, then call.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from inspect_ai.tool import ToolDef, mcp_server_stdio
from inspect_ai.tool._mcp.connection import mcp_connection

if TYPE_CHECKING:
    from inspect_ai.tool._tool import Tool

MCP_TEST_SERVER = str(Path(__file__).parent / "mcp_test_server.py")

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="mcp package requires Python 3.10+"
)


class _FakeActiveSample:
    """Minimal stand-in for `ActiveSample` as a session scope.

    Carries the attributes `SampleContextFilter` reads (`task`, `sample.id`,
    `epoch`) as well as `completed`: any log emitted while this is installed as
    `sample_active()` passes through that filter, so a fake with only
    `completed` blows up with AttributeError as soon as another test in the
    session has attached a handler carrying it.
    """

    def __init__(self, sample_id: str = "s1") -> None:
        self.completed: float | None = None
        self.task = "test-task"
        self.epoch = 1
        self.sample = SimpleNamespace(id=sample_id)


def _sample_var(value: object):
    from contextvars import ContextVar

    var: ContextVar[object] = ContextVar("_sample_active_test", default=None)
    var.set(value)
    return var


def _counting_server(counter: Path):
    prelude = (
        f"open({str(counter)!r}, 'a').write('start\\n');"
        f"exec(open({MCP_TEST_SERVER!r}).read())"
    )
    return mcp_server_stdio(command=sys.executable, args=["-c", prelude])


def _launches(counter: Path) -> int:
    if not counter.exists():
        return 0
    return len([line for line in counter.read_text().splitlines() if line.strip()])


async def _echo(tool: Tool, message: str) -> None:
    result = await tool(message=message)
    assert result[0].text == message  # pyright: ignore[reportIndexIssue,reportAttributeAccessIssue]


async def test_without_a_sample_resource_closures_still_launch_per_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: with nothing holding a session, churn returns.

    Documents that the sample-lifetime hold is load-bearing, not belt-and-braces.
    Closure re-resolution (`_client_session` consulting the owner) only helps
    when SOMETHING is keeping a session open for the scope; on its own it still
    falls back to a private client per call. `Task.sample_resources` is what
    supplies that hold in production -- see
    test_sample_resource_shape_gives_one_launch for the composed result.
    """
    counter = tmp_path / "launches.txt"
    server = _counting_server(counter)
    monkeypatch.setattr(
        "inspect_ai.log._samples._sample_active",
        _sample_var(_FakeActiveSample()),
        raising=True,
    )

    # Bridge registration: materialize the tool closures inside a connection...
    async with mcp_connection(server):
        tools = await server.tools()
        echo = next(t for t in tools if ToolDef(t).name == "echo")
    # ...connection is gone; the agent now calls the SAME closures per request.
    for i in range(4):
        await _echo(echo, f"call-{i}")

    assert _launches(counter) == 5, (
        "without a sample-lifetime hold each call starts its own server "
        f"(4 calls + the resolve); got {_launches(counter)}"
    )


async def test_closures_still_work_after_sample_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call after the sample ends must not silently reuse a dead session.

    It may start its own server (the old behaviour) — what it must not do is
    raise or hang, since scoring can legitimately trail sample completion.
    """
    counter = tmp_path / "launches.txt"
    server = _counting_server(counter)
    sample = _FakeActiveSample()
    monkeypatch.setattr(
        "inspect_ai.log._samples._sample_active",
        _sample_var(sample),
        raising=True,
    )

    async with mcp_connection(server):
        tools = await server.tools()
        echo = next(t for t in tools if ToolDef(t).name == "echo")

    sample.completed = 1.0
    await _echo(echo, "after-completion")


async def test_closures_adopt_a_held_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a connection held for the sample, closures must adopt it.

    This is the other half of the fix: `_client_session()` re-resolving through
    the owner only helps if SOMETHING is holding a session open for the scope.
    Here the outer connection plays the role a sample-lifetime resource would.
    """
    counter = tmp_path / "launches.txt"
    server = _counting_server(counter)
    monkeypatch.setattr(
        "inspect_ai.log._samples._sample_active",
        _sample_var(_FakeActiveSample()),
        raising=True,
    )

    async with mcp_connection(server):  # held for the whole "sample"
        # inner connection resolves the closures, then exits (the setup phase)
        async with mcp_connection(server):
            tools = await server.tools()
            echo = next(t for t in tools if ToolDef(t).name == "echo")
        for i in range(4):
            await _echo(echo, f"call-{i}")

    assert _launches(counter) == 1, (
        f"closures must adopt the held session; got {_launches(counter)} launches"
    )


async def test_sample_resource_shape_gives_one_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full production composition, as `Task.sample_resources` runs it.

    A sample-lifetime connection wraps everything (what inspect's
    `_sample_resources_cm` does), the setup chain opens and closes its own
    nested connection, the bridge resolves tool closures there, and the agent
    then invokes those closures from separate tasks. This is the shape that
    measured 5 launches for 4 calls in-sandbox.
    """
    import anyio

    counter = tmp_path / "launches.txt"
    server = _counting_server(counter)
    monkeypatch.setattr(
        "inspect_ai.log._samples._sample_active",
        _sample_var(_FakeActiveSample()),
        raising=True,
    )

    async with mcp_connection(server):  # Task.sample_resources
        async with mcp_connection(server):  # setup chain
            tools = await server.tools()
            echo = next(t for t in tools if ToolDef(t).name == "echo")

        # agent phase: each bridged call runs in its own task
        async def call(i: int) -> None:
            await _echo(echo, f"call-{i}")

        for i in range(4):
            async with anyio.create_task_group() as tg:
                tg.start_soon(call, i)

    assert _launches(counter) == 1, (
        f"the composed fix must yield one server per sample; got {_launches(counter)}"
    )
