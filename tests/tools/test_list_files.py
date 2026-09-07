"""Tests for list_files tool."""

from pathlib import Path

import pytest
from test_helpers.tool_call_utils import get_tool_call, get_tool_response
from test_helpers.utils import skip_if_no_docker

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, list_files
from inspect_ai.util import SandboxEnvironmentType

TEST_FILES = {
    "/tmp/testdir/a.txt": "a",
    "/tmp/testdir/b.txt": "b",
    "/tmp/testdir/sub/c.txt": "c",
}

TOOL_SANDBOX: SandboxEnvironmentType = (
    "docker",
    str(Path(__file__).with_name("test_sandbox_compose.yaml")),
)

CWD_TEST_FILES = {
    "provider-default.txt": "provider-default-list-files",
    "/tmp/cwd-bound/inside.txt": "configured-cwd-list-files",
    "/tmp/outside/outside.txt": "absolute-path-list-files",
}


def test_list_files_constructible() -> None:
    """Tool is constructible without a sandbox."""
    tool = list_files()
    assert tool is not None


def _run_list_files(
    tool_arguments: dict,
    *,
    tool: Tool | None = None,
    files: dict[str, str] | None = None,
    sandbox: SandboxEnvironmentType = "docker",
) -> str:
    task = Task(
        dataset=[
            Sample(
                input="Please use the tool",
                target="n/a",
                files=files or TEST_FILES,
            )
        ],
        solver=[use_tools(tool if tool is not None else list_files()), generate()],
        scorer=includes(),
        message_limit=3,
        sandbox=sandbox,
    )
    result = eval(
        task,
        model=get_model(
            "mockllm/model",
            custom_outputs=[
                ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name="list_files",
                    tool_arguments=tool_arguments,
                ),
            ],
        ),
    )[0]
    assert result.samples
    messages = result.samples[0].messages
    tool_call = get_tool_call(messages, "list_files")
    assert tool_call is not None
    response = get_tool_response(messages, tool_call)
    assert response is not None
    return str(response.content)


@skip_if_no_docker
@pytest.mark.slow
def test_list_files_basic() -> None:
    content = _run_list_files({"path": "/tmp/testdir"})
    assert "a.txt" in content
    assert "b.txt" in content
    assert "c.txt" in content


@skip_if_no_docker
@pytest.mark.slow
def test_list_files_with_depth() -> None:
    content = _run_list_files({"path": "/tmp/testdir", "depth": 1})
    assert "a.txt" in content
    assert "b.txt" in content
    assert "sub" in content
    # c.txt is at depth 2, should not appear with depth=1
    assert "c.txt" not in content


@skip_if_no_docker
@pytest.mark.slow
def test_list_files_dash_path_safe() -> None:
    """Path starting with - must not be interpreted as a find predicate."""
    task = Task(
        dataset=[
            Sample(
                input="Please use the tool",
                target="n/a",
                files=TEST_FILES,
            )
        ],
        solver=[use_tools(list_files()), generate()],
        scorer=includes(),
        message_limit=3,
        sandbox="docker",
    )
    result = eval(
        task,
        model=get_model(
            "mockllm/model",
            custom_outputs=[
                ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name="list_files",
                    tool_arguments={"path": "-delete"},
                ),
            ],
        ),
    )[0]
    assert result.samples
    messages = result.samples[0].messages
    tool_call = get_tool_call(messages, "list_files")
    assert tool_call is not None
    response = get_tool_response(messages, tool_call)
    assert response is not None
    # Should get an error (path not found), not execute -delete
    assert response.error is not None


@skip_if_no_docker
@pytest.mark.slow
def test_list_files_cwd_uses_configured_directory_and_keeps_absolute_paths() -> None:
    configured_tool = list_files(cwd="/tmp/cwd-bound")

    relative_content = _run_list_files(
        {},
        tool=configured_tool,
        files=CWD_TEST_FILES,
        sandbox=TOOL_SANDBOX,
    )
    absolute_content = _run_list_files(
        {"path": "/tmp/outside"},
        tool=configured_tool,
        files=CWD_TEST_FILES,
        sandbox=TOOL_SANDBOX,
    )

    assert "inside.txt" in relative_content
    assert "outside.txt" in absolute_content


@skip_if_no_docker
@pytest.mark.slow
def test_list_files_omitted_cwd_keeps_provider_default_directory() -> None:
    content = _run_list_files(
        {},
        files=CWD_TEST_FILES,
        sandbox=TOOL_SANDBOX,
    )

    assert "provider-default.txt" in content
