from contextlib import asynccontextmanager
from io import BytesIO
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

from inspect_ai.tool._sandbox_tools_utils.sandbox import (
    SandboxInjectionError,
    _inject_container_tools_code,
)
from inspect_ai.util._sandbox.environment import SandboxEnvironment


@pytest.mark.anyio
async def test_injection_rejects_binary_with_wrong_fork_revision() -> None:
    @asynccontextmanager
    async def opened_executable(*_: object):
        yield "inspect-sandbox-tools-amd64-v23-tl1", BytesIO(b"artifact")

    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(success=True, stderr=""),
                SimpleNamespace(success=True, stderr=""),
                SimpleNamespace(success=True, stderr=""),
                SimpleNamespace(
                    success=True,
                    stderr="",
                    stdout='{"jsonrpc":"2.0","result":"1.2.1+tl.999","id":1}',
                ),
            ]
        )
    )

    with (
        patch(
            "inspect_ai.tool._sandbox_tools_utils.sandbox.detect_sandbox_os",
            AsyncMock(return_value={"architecture": "amd64", "libc": "glibc"}),
        ),
        patch(
            "inspect_ai.tool._sandbox_tools_utils.sandbox._open_executable_for_arch",
            opened_executable,
        ),
        patch(
            "inspect_ai.tool._sandbox_tools_utils.sandbox._extract_tools_tree",
            AsyncMock(),
        ),
        pytest.raises(SandboxInjectionError, match="fork revision"),
    ):
        await _inject_container_tools_code(cast(SandboxEnvironment, sandbox))
