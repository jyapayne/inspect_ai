"""Shared pytest fixtures and utilities for inspect_sandbox_tools tests.

This module provides common test infrastructure for integration tests that
communicate with the sandbox tools server via JSON-RPC.
"""

import json
import logging
import os
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
from inspect_sandbox_tools._util.constants import SOCKET_PATH

logger = logging.getLogger(__name__)

# Test configuration constants
DEFAULT_RPC_TIMEOUT = 10

# (forwarded provider HTTP status, the Anthropic error type a bridged client must
# observe). One table for both the in-process proxy tests and the built-artifact
# wire test, so the two cannot drift: a status that the proxy reports as the wrong
# type here is a status a streaming Claude Code client misreads (409 conflicts and
# 504 deadlines were surfacing as `api_error`, i.e. as server failures). 422 pins
# the unlisted-4xx fallback; 503 is the control that a real server error still
# reports `api_error`.
ANTHROPIC_WIRE_ERROR_TYPES: list[tuple[int, str]] = [
    (409, "conflict_error"),
    (504, "timeout_error"),
    (402, "billing_error"),
    (422, "invalid_request_error"),
    (503, "api_error"),
]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--sandbox-tools-artifact",
        action="store",
        default=None,
        metavar="PATH",
        help=(
            "A built inspect-sandbox-tools bundle: the gzipped tar that "
            "build_executable.py writes under src/inspect_ai/binaries/, or a "
            "directory it was extracted into. Enables the tests that run the built "
            "executable itself; without it those tests skip."
        ),
    )


@pytest.fixture(scope="session")
def built_sandbox_tools(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The launcher of a built sandbox-tools bundle, or skip.

    The in-process tests exercise the proxy's source; only a test that runs the
    built executable can show what a consumer's injected binary does. A skip here
    means that test did not run, so a report that leans on it must say whether
    it ran.
    """
    artifact = request.config.getoption("--sandbox-tools-artifact")
    if artifact is None:
        pytest.skip(
            "no built artifact: pass --sandbox-tools-artifact PATH to run the "
            "built executable"
        )
    path = Path(artifact)
    if path.is_dir():
        bundle_dir = path
    else:
        bundle_dir = tmp_path_factory.mktemp("sandbox-tools-bundle")
        with tarfile.open(path, "r:gz") as bundle:
            bundle.extractall(bundle_dir, filter="data")
    launcher = bundle_dir / "inspect-sandbox-tools"
    if not launcher.is_file():
        raise FileNotFoundError(
            f"{artifact} holds no inspect-sandbox-tools launcher at {launcher}"
        )
    return launcher


def cleanup_socket() -> None:
    """Remove any existing socket file."""
    SOCKET_PATH.unlink(missing_ok=True)


_SERVER_PROCESS_PATTERN = "inspect_sandbox_tools._cli.main server"


def _count_server_processes() -> int:
    """Count running server processes matching our pattern."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", _SERVER_PROCESS_PATTERN],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            return len(pids)
        return 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return -1


def cleanup_server_processes() -> None:
    """Kill any running server processes."""
    before_count = _count_server_processes()

    try:
        subprocess.run(
            ["pkill", "-f", _SERVER_PROCESS_PATTERN],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if before_count > 0:
        for _ in range(10):
            time.sleep(0.05)
            if _count_server_processes() == 0:
                break
        else:
            logger.warning(
                "[cleanup] Server processes still running after pkill (%d before kill)",
                before_count,
            )


def _exec_rpc_request_impl(
    request: dict[str, Any], timeout: int = DEFAULT_RPC_TIMEOUT
) -> dict[str, Any]:
    """Execute an RPC request via the CLI and return the parsed response.

    Args:
        request: The JSON-RPC request dictionary.
        timeout: Timeout in seconds for the subprocess call.

    Returns:
        The parsed JSON-RPC response dictionary.

    Raises:
        pytest.fail: If the CLI command fails or returns invalid JSON.
    """
    request_json = json.dumps(request)

    result = subprocess.run(
        [sys.executable, "-m", "inspect_sandbox_tools._cli.main", "exec"],
        input=request_json,
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=os.getcwd(),
        check=False,
    )

    if result.returncode != 0:
        pytest.fail(f"CLI command failed: {result.stderr}")

    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON response: {result.stdout}\nError: {e}")


# Type alias for the RPC client function
RpcClient = Callable[[dict[str, Any], int], dict[str, Any]]


@pytest.fixture
def rpc_client() -> RpcClient:
    """Fixture providing an RPC client for executing JSON-RPC requests.

    Returns:
        A callable that takes a request dict and optional timeout, returning the response.

    Example:
        def test_something(rpc_client):
            response = rpc_client({"jsonrpc": "2.0", "method": "...", "params": {}, "id": 1}, 10)
            assert "result" in response
    """
    return _exec_rpc_request_impl


@pytest.fixture(autouse=False)
def sandbox_server_cleanup() -> Generator[None, None, None]:
    """Fixture to set up and tear down the sandbox server environment.

    This fixture:
    - Cleans up any existing socket files before the test
    - Kills any running server processes before the test
    - Cleans up after the test completes

    Usage:
        @pytest.mark.usefixtures("sandbox_server_cleanup")
        class TestMyFeature:
            ...

        # Or for individual tests:
        def test_something(sandbox_server_cleanup, rpc_client):
            ...
    """
    cleanup_socket()
    cleanup_server_processes()
    yield
    cleanup_socket()
    cleanup_server_processes()
