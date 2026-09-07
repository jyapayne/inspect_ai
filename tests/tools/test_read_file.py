"""Tests for read_file tool."""

from pathlib import Path

import pytest
from test_helpers.tool_call_utils import get_tool_call, get_tool_response
from test_helpers.utils import skip_if_no_docker

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, read_file
from inspect_ai.util import SandboxEnvironmentType

TOOL_SANDBOX: SandboxEnvironmentType = (
    "docker",
    str(Path(__file__).with_name("test_sandbox_compose.yaml")),
)

CWD_TEST_FILES = {
    "provider-default.txt": "provider-default-read-file",
    "/tmp/cwd-bound/inside.txt": "configured-cwd-read-file",
    "/tmp/outside/outside.txt": "absolute-path-read-file",
}


def test_read_file_constructible() -> None:
    """Tool is constructible without a sandbox."""
    tool = read_file()
    assert tool is not None


def _read_file_task(
    files: dict[str, str] | None = None,
    *,
    tool: Tool | None = None,
    sandbox: SandboxEnvironmentType = "docker",
) -> Task:
    sample = Sample(
        input="Please use the tool",
        target="n/a",
        files=files or {"/tmp/test.txt": "line1\nline2\nline3\nline4\nline5"},
    )
    return Task(
        dataset=[sample],
        solver=[use_tools(tool if tool is not None else read_file()), generate()],
        scorer=includes(),
        message_limit=3,
        sandbox=sandbox,
    )


def _run_read_file(
    tool_arguments: dict,
    files: dict[str, str] | None = None,
    *,
    tool: Tool | None = None,
    sandbox: SandboxEnvironmentType = "docker",
) -> str:
    task = _read_file_task(files, tool=tool, sandbox=sandbox)
    result = eval(
        task,
        model=get_model(
            "mockllm/model",
            custom_outputs=[
                ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name="read_file",
                    tool_arguments=tool_arguments,
                ),
            ],
        ),
    )[0]
    assert result.samples
    messages = result.samples[0].messages
    tool_call = get_tool_call(messages, "read_file")
    assert tool_call is not None
    response = get_tool_response(messages, tool_call)
    assert response is not None
    return str(response.content)


@skip_if_no_docker
@pytest.mark.slow
def test_read_file_basic() -> None:
    content = _run_read_file({"file_path": "/tmp/test.txt"})
    assert "line1" in content
    assert "line5" in content


@skip_if_no_docker
@pytest.mark.slow
def test_read_file_with_offset() -> None:
    content = _run_read_file({"file_path": "/tmp/test.txt", "offset": 2})
    assert "line1" not in content
    assert "line2" not in content
    assert "line3" in content
    assert "line5" in content


@skip_if_no_docker
@pytest.mark.slow
def test_read_file_with_limit() -> None:
    content = _run_read_file({"file_path": "/tmp/test.txt", "limit": 2})
    assert "line1" in content
    assert "line2" in content
    assert "line3" not in content


@skip_if_no_docker
@pytest.mark.slow
def test_read_file_with_offset_and_limit() -> None:
    content = _run_read_file({"file_path": "/tmp/test.txt", "offset": 1, "limit": 2})
    assert "line1" not in content
    assert "line2" in content
    assert "line3" in content
    assert "line4" not in content


@skip_if_no_docker
@pytest.mark.slow
def test_read_file_not_found() -> None:
    task = _read_file_task()
    result = eval(
        task,
        model=get_model(
            "mockllm/model",
            custom_outputs=[
                ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name="read_file",
                    tool_arguments={"file_path": "/tmp/nonexistent.txt"},
                ),
            ],
        ),
    )[0]
    assert result.samples
    messages = result.samples[0].messages
    tool_call = get_tool_call(messages, "read_file")
    assert tool_call is not None
    response = get_tool_response(messages, tool_call)
    assert response is not None
    assert response.error is not None
    assert "not found" in response.error.message.lower()


@skip_if_no_docker
@pytest.mark.slow
def test_read_file_cwd_uses_configured_directory_and_keeps_absolute_paths() -> None:
    configured_tool = read_file(cwd="/tmp/cwd-bound")

    relative_content = _run_read_file(
        {"file_path": "inside.txt"},
        CWD_TEST_FILES,
        tool=configured_tool,
        sandbox=TOOL_SANDBOX,
    )
    absolute_content = _run_read_file(
        {"file_path": "/tmp/outside/outside.txt"},
        CWD_TEST_FILES,
        tool=configured_tool,
        sandbox=TOOL_SANDBOX,
    )

    assert "configured-cwd-read-file" in relative_content
    assert "absolute-path-read-file" in absolute_content


@skip_if_no_docker
@pytest.mark.slow
def test_read_file_omitted_cwd_keeps_provider_default_directory() -> None:
    content = _run_read_file(
        {"file_path": "provider-default.txt"},
        CWD_TEST_FILES,
        sandbox=TOOL_SANDBOX,
    )

    assert "provider-default-read-file" in content
