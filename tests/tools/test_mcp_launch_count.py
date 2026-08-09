"""How many MCP server processes one sample's tool traffic actually starts.

The unit tests in ``test_mcp_session_isolation.py`` assert session-table
bookkeeping with a stand-in sample; these launch **real** stdio servers and
count the processes, which is the property that matters: a sandboxed MCP
server costs an exec channel, an interpreter start and a handshake per launch,
and paying that once per calling task (rather than once per sample) is what
starved the event loop on large evals.

The count is recorded by the server process itself, so it cannot be satisfied
by mocking.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import anyio
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


def _counting_server(counter: Path):
    """A stdio MCP server that appends one line per process start."""
    prelude = (
        f"open({str(counter)!r}, 'a').write('start\\n');"
        f"exec(open({MCP_TEST_SERVER!r}).read())"
    )
    return mcp_server_stdio(command=sys.executable, args=["-c", prelude])


def _launches(counter: Path) -> int:
    if not counter.exists():
        return 0
    return len([line for line in counter.read_text().splitlines() if line.strip()])


async def _echo(tools: list[Tool]) -> None:
    echo = next(t for t in tools if ToolDef(t).name == "echo")
    result = await echo(message="hi")
    assert result[0].text == "hi"  # pyright: ignore[reportIndexIssue,reportAttributeAccessIssue]


async def test_one_server_launch_per_sample_across_child_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The churn fix, measured: N calling tasks must not mean N servers.

    Mirrors the shape that regressed in production — a held connection in the
    sample's task, with tool calls served from separate tasks (the sandbox
    agent bridge spawns one per request).
    """
    counter = tmp_path / "launches.txt"
    server = _counting_server(counter)
    monkeypatch.setattr(
        "inspect_ai.log._samples._sample_active",
        _sample_var(_FakeActiveSample()),
        raising=True,
    )

    async with mcp_connection(server):
        tools = await server.tools()
        await _echo(tools)

        async def child() -> None:
            await _echo(await server.tools())

        async with anyio.create_task_group() as tg:
            for _ in range(4):
                tg.start_soon(child)

    assert _launches(counter) == 1, (
        f"expected one server process for the whole sample, got {_launches(counter)}"
    )


async def test_distinct_samples_launch_distinct_servers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolation, measured: two attempts must not share a server process."""
    counter = tmp_path / "launches.txt"
    server = _counting_server(counter)

    for _ in range(2):
        monkeypatch.setattr(
            "inspect_ai.log._samples._sample_active",
            _sample_var(_FakeActiveSample()),
            raising=True,
        )
        async with mcp_connection(server):
            await _echo(await server.tools())

    assert _launches(counter) == 2, (
        f"each sample attempt must get its own server process, got {_launches(counter)}"
    )


def _sample_var(value: object):
    """A ContextVar pre-set to ``value`` (module-level patch target)."""
    from contextvars import ContextVar

    var: ContextVar[object] = ContextVar("_sample_active_test", default=None)
    var.set(value)
    return var
