import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from test_helpers.utils import skip_if_no_docker

from inspect_ai import Task, eval
from inspect_ai.agent import sandbox_agent_bridge
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import Solver, solver
from inspect_ai.util import sandbox



class _CollectingSink:
    """ModelEventSink that retains bridged event pairs for assertions."""

    def __init__(self) -> None:
        self.pending: list[object] = []
        self.complete: list[object] = []

    def on_pending(self, event: object) -> None:
        self.pending.append(event)

    def on_complete(self, event: object) -> None:
        self.complete.append(event)

@skip_if_no_docker
@pytest.mark.slow
def test_sandbox_bridge_cannot_read_host_media(tmp_path: Path) -> None:
    secret = tmp_path / "host-only-secret.txt"
    secret_bytes = b"SANDBOX_MUST_NOT_RECEIVE_THIS"
    secret.write_bytes(secret_bytes)
    provider_requests: list[dict[str, Any]] = []

    target_model = get_model(
        "openai/gpt-5",
        api_key="local-fake-key",
        base_url="http://127.0.0.1:9/v1",
        memoize=False,
        responses_api=True,
        config=GenerateConfig(reasoning_summary="none"),
    )

    async def capture_create(**kwargs: Any) -> None:
        provider_requests.append(kwargs)
        raise RuntimeError("Stop after capturing provider request.")

    cast(Any, target_model.api).client.responses.create = capture_create

    @solver
    def bridge_solver() -> Solver:
        async def solve(state, generate):
            async with sandbox_agent_bridge(
                model_aliases={"inspect": target_model}
            ) as bridge:
                request = {
                    "model": "inspect",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_file",
                                    "file_data": str(secret),
                                    "filename": "secret.txt",
                                }
                            ],
                        }
                    ],
                    "tools": [],
                }
                script = (
                    "import urllib.error, urllib.request\n"
                    f"body = {json.dumps(request)!r}.encode()\n"
                    "request = urllib.request.Request(\n"
                    f"    'http://127.0.0.1:{bridge.port}/v1/responses',\n"
                    "    data=body,\n"
                    "    headers={'Content-Type': 'application/json'},\n"
                    "    method='POST',\n"
                    ")\n"
                    "try:\n"
                    "    print(urllib.request.urlopen(request).read().decode())\n"
                    "except urllib.error.HTTPError as ex:\n"
                    "    print(ex.read().decode())\n"
                )
                await sandbox().write_file("bridge_media_test.py", script)
                result = await sandbox().exec(
                    ["python3", "bridge_media_test.py"], timeout=60
                )
                state.metadata["bridge_stdout"] = result.stdout
                state.metadata["bridge_stderr"] = result.stderr
            return state

        return solve

    logs = eval(
        Task(
            dataset=[Sample(id="sample", input="test")],
            solver=bridge_solver(),
            sandbox="docker",
        ),
        model="mockllm/model",
        display="none",
        log_dir=str(tmp_path / "logs"),
    )

    try:
        assert logs[0].status == "success"
        assert logs[0].samples is not None
        metadata = logs[0].samples[0].metadata
        output = metadata["bridge_stdout"] + metadata["bridge_stderr"]
        assert secret_bytes.decode() not in output
        assert provider_requests == []
        serialized_requests = json.dumps(provider_requests)
        assert str(secret) not in serialized_requests
        assert secret_bytes.decode() not in serialized_requests
    finally:
        asyncio.run(target_model.api.aclose())


@skip_if_no_docker
@pytest.mark.slow
def test_sandbox_bridge_scopes_allowlisted_headers_to_model_events(
    tmp_path: Path,
) -> None:
    """Scripted concurrent HTTP requests preserve only selected headers per event."""
    from inspect_ai.model import (
        BRIDGE_FILTER_SYNTHETIC,
        BRIDGE_REQUEST_HEADERS,
        ModelOutput,
    )

    sink = _CollectingSink()
    target_model = get_model("mockllm/model")

    async def header_filter(
        model: object,
        input: object,
        tools: object,
        tool_choice: object,
        config: GenerateConfig,
    ) -> ModelOutput | None:
        del model, input, tools, tool_choice
        if config.extra_headers and config.extra_headers.get("x-session-id") == "filter":
            return ModelOutput.from_content("mockllm/model", "filter output")
        return None

    @solver
    def bridge_solver() -> Solver:
        async def solve(state, generate):
            del generate
            async with sandbox_agent_bridge(
                model_aliases={"inspect": target_model},
                filter=header_filter,
                model_event_sink=sink,
                model_event_metadata_headers=(
                    "x-opencode-session",
                    "x-session-id",
                    "x-parent-session-id",
                ),
            ) as bridge:
                request = {
                    "model": "inspect",
                    "messages": [{"role": "user", "content": "hello"}],
                }
                script = (
                    "import json\n"
                    "from concurrent.futures import ThreadPoolExecutor\n"
                    "from urllib.request import Request, urlopen\n"
                    f"url = 'http://127.0.0.1:{bridge.port}/v1/chat/completions'\n"
                    f"body = {json.dumps(request)!r}.encode()\n"
                    "def post(headers):\n"
                    "    request = Request(url, data=body, headers=headers, method='POST')\n"
                    "    with urlopen(request) as response:\n"
                    "        return json.loads(response.read())['choices'][0]['message']['content']\n"
                    "headers = [\n"
                    "    {\n"
                    "        'Content-Type': 'application/json',\n"
                    "        'X-OpenCode-Session': 'provider-session',\n"
                    "        'X-Parent-Session-Id': 'provider-parent',\n"
                    "        'Authorization': 'Bearer must-not-record',\n"
                    "        'X-Unselected': 'must-not-record',\n"
                    "    },\n"
                    "    {\n"
                    "        'Content-Type': 'application/json',\n"
                    "        'X-Session-Id': 'filter',\n"
                    "        'X-Parent-Session-Id': 'filter-parent',\n"
                    "        'Cookie': 'must-not-record',\n"
                    "        'X-Unselected': 'must-not-record',\n"
                    "    },\n"
                    "]\n"
                    "with ThreadPoolExecutor(max_workers=2) as executor:\n"
                    "    print(json.dumps(list(executor.map(post, headers))))\n"
                )
                await sandbox().write_file("bridge_metadata_test.py", script)
                result = await sandbox().exec(
                    ["python3", "bridge_metadata_test.py"], timeout=60
                )
                state.metadata["bridge_stdout"] = result.stdout
                state.metadata["bridge_stderr"] = result.stderr
            return state

        return solve

    logs = eval(
        Task(
            dataset=[Sample(id="sample", input="test")],
            solver=bridge_solver(),
            sandbox="docker",
        ),
        model="mockllm/model",
        display="none",
        log_dir=str(tmp_path / "logs"),
    )

    assert logs[0].status == "success"
    assert logs[0].samples is not None
    assert "filter output" in logs[0].samples[0].metadata["bridge_stdout"]
    assert len(sink.pending) == 2
    assert len(sink.complete) == 2
    event_metadata = [getattr(event, "metadata", None) or {} for event in sink.complete]
    headers_by_session = {
        metadata[BRIDGE_REQUEST_HEADERS].get(
            "x-opencode-session", metadata[BRIDGE_REQUEST_HEADERS].get("x-session-id")
        ): metadata
        for metadata in event_metadata
    }
    assert headers_by_session["provider-session"][BRIDGE_REQUEST_HEADERS] == {
        "x-opencode-session": "provider-session",
        "x-parent-session-id": "provider-parent",
    }
    assert headers_by_session["filter"][BRIDGE_REQUEST_HEADERS] == {
        "x-session-id": "filter",
        "x-parent-session-id": "filter-parent",
    }
    assert headers_by_session["filter"][BRIDGE_FILTER_SYNTHETIC] is True
