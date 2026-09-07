import re
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator, BinaryIO

import pytest
import semver
from test_helpers.sandbox import CannedSandbox

from inspect_ai.tool._sandbox_tools_utils import sandbox as sandbox_tools
from inspect_ai.tool._sandbox_tools_utils.sandbox import SandboxInjectionError
from inspect_ai.util._sandbox._cli import SANDBOX_CLI, SANDBOX_TOOLS_DIR
from inspect_ai.util._sandbox._framework_directory import _VERIFIED_MARKER
from inspect_ai.util._sandbox._privileged import SHELL_PATH
from inspect_ai.util._sandbox.environment import SandboxEnvironment
from inspect_ai.util._sandbox.recon import Architecture, SupportedContainerOSInfo
from inspect_ai.util._subprocess import ExecResult

REPO_ROOT = Path(__file__).parents[3]
SANDBOX_TOOLS_PYPROJECT = REPO_ROOT / "src/inspect_sandbox_tools/pyproject.toml"
FORK_REVISION = (
    REPO_ROOT
    / "src/inspect_ai/tool/_sandbox_tools_utils/sandbox_tools_fork_revision.txt"
)


OK = ExecResult(success=True, returncode=0, stdout="", stderr="")
"""Result of an ordinary (non-helper) command."""

VERIFIED = ExecResult(
    success=True, returncode=0, stdout="", stderr=f"{_VERIFIED_MARKER}\n"
)
"""Helper result: the tools directory verified and the wrapped command succeeded."""

WRONG_FORK_VERSION = ExecResult(
    success=True,
    returncode=0,
    stdout='{"jsonrpc":"2.0","result":"1.2.1+tl.999","id":1}',
    stderr="",
)
"""Launcher's answer to the version query: a build from some other fork revision."""


def is_framework_dir_call(cmd: list[str]) -> bool:
    return cmd[:2] == [SHELL_PATH, "-c"] and SANDBOX_TOOLS_DIR.rsplit("/", 1)[1] in cmd


def is_version_query(cmd: list[str]) -> bool:
    return cmd == [SANDBOX_CLI, "exec"]


DEFAULT_USER = ExecResult(
    success=True,
    returncode=0,
    stdout="Uid: 1111\t1111\t1111\t1111\nGid: 1111\t1111\t1111\t1111\nGroups: 1111 \nHOME: /home/nonroot\nHOME_SET: 1\n",
    stderr="",
)
"""Result of the default-user identity probe."""

ROOT_CAPS = ExecResult(
    success=True,
    returncode=0,
    stdout="Uid: 0 0 0 0\nCapEff: 000001ffffffffff\nsetgroups: allow\n",
    stderr="",
)
"""Result of the root capability probe when root can switch users."""


def is_identity_probe(cmd: list[str]) -> bool:
    return cmd[:2] == [SHELL_PATH, "-c"] and "Groups:" in cmd[2]


def is_caps_probe(cmd: list[str]) -> bool:
    return cmd[:2] == [SHELL_PATH, "-c"] and "CapEff:" in cmd[2]


def wrong_fork_binary(cmd: list[str], user: str | None) -> ExecResult[str]:
    """Every helper call verifies and every command succeeds; only the build is foreign."""
    if is_framework_dir_call(cmd):
        return VERIFIED
    if is_identity_probe(cmd):
        return DEFAULT_USER
    if is_caps_probe(cmd):
        return ROOT_CAPS
    return WRONG_FORK_VERSION if is_version_query(cmd) else OK


@pytest.mark.anyio
async def test_injection_rejects_binary_with_wrong_fork_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert sandbox_tools._get_sandbox_tools_fork_revision() != 999

    async def fake_detect_sandbox_os(
        _sandbox: SandboxEnvironment,
    ) -> SupportedContainerOSInfo:
        return {"architecture": "amd64", "libc": "glibc"}

    @asynccontextmanager
    async def fake_open_executable_for_arch(
        _arch: Architecture, _musl: bool
    ) -> AsyncIterator[tuple[str, BinaryIO]]:
        yield "inspect-sandbox-tools-amd64-v23-tl1", BytesIO(b"artifact")

    async def fake_extract_tools_tree(
        _sandbox: SandboxEnvironment, _name: str, _gz_bytes: bytes, _user: str | None
    ) -> None:
        pass

    monkeypatch.setattr(sandbox_tools, "detect_sandbox_os", fake_detect_sandbox_os)
    monkeypatch.setattr(
        sandbox_tools, "_open_executable_for_arch", fake_open_executable_for_arch
    )
    monkeypatch.setattr(sandbox_tools, "_extract_tools_tree", fake_extract_tools_tree)

    sandbox = CannedSandbox(wrong_fork_binary)
    with pytest.raises(SandboxInjectionError, match="fork revision"):
        await sandbox_tools._inject_container_tools_code(sandbox)

    # The launcher was started and asked for its version before the mismatch was raised.
    assert ([SANDBOX_CLI, "start-server"], "root") in sandbox.exec_calls
    assert sandbox.exec_calls[-1] == ([SANDBOX_CLI, "exec"], "root")


def test_tools_package_version_carries_fork_revision() -> None:
    pyproject = SANDBOX_TOOLS_PYPROJECT.read_text(encoding="utf-8")
    version_match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert version_match is not None

    version = semver.Version.parse(version_match.group(1))
    assert version.build == f"tl.{int(FORK_REVISION.read_text(encoding='utf-8'))}"
