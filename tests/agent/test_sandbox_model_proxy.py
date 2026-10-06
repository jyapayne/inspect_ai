"""Tests for `sandbox_model_proxy`, the agent bridge's proxy with caller-supplied handlers."""

import contextlib
import json
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import JsonValue
from test_helpers.utils import skip_if_no_docker

from inspect_ai import Task, eval, task
from inspect_ai.agent import (
    BridgedToolsSpec,
    ModelProxy,
    ModelProxyError,
    sandbox_agent_bridge,
    sandbox_model_proxy,
)
from inspect_ai.agent._bridge._errors import PROVIDER_ERROR_KEY
from inspect_ai.agent._bridge.sandbox import bridge as bridge_module
from inspect_ai.agent._bridge.sandbox import proxy as proxy_module
from inspect_ai.agent._bridge.sandbox.types import SandboxAgentBridge
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import tool
from inspect_ai.util import sandbox
from inspect_ai.util._limit import LimitExceededError
from inspect_ai.util._sandbox.environment import SandboxEnvironment
from inspect_ai.util._sandbox.service import SandboxServiceMethod
from inspect_ai.util._subprocess import ExecResult

GENERATE_METHODS = {
    "generate_completions",
    "generate_responses",
    "generate_anthropic",
    "generate_google",
}

ADD_DESCRIPTION = "Add two numbers."

FIXED_MESSAGE: dict[str, JsonValue] = {
    "id": "msg_fixed_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-fixed",
    "content": [{"type": "text", "text": "The proxy returned this unchanged."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 11, "output_tokens": 7},
}


@tool
def calculator_add(call_log: list[dict[str, int]]):
    async def execute(x: int, y: int) -> str:
        """Add two numbers.

        Args:
            x: First number to add.
            y: Second number to add.
        """
        call_log.append({"x": x, "y": y})
        return str(x + y)

    return execute


def _unused_sandbox() -> MagicMock:
    return MagicMock(spec=SandboxEnvironment)


async def _ok(json_data: dict[str, JsonValue], **_: JsonValue) -> JsonValue:
    return FIXED_MESSAGE


# ---------------------------------------------------------------------------
# Method table validation (refused before the sandbox is touched)
# ---------------------------------------------------------------------------


async def test_unknown_method_is_refused() -> None:
    sandbox_env = _unused_sandbox()
    with pytest.raises(ValueError, match="generate_bedrock"):
        async with sandbox_model_proxy(
            sandbox_env, methods={"generate_anthropic": _ok, "generate_bedrock": _ok}
        ):
            pass
    assert sandbox_env.method_calls == []


async def test_non_callable_method_is_refused() -> None:
    sandbox_env = _unused_sandbox()
    methods: dict[str, Any] = {"generate_anthropic": "not a handler"}
    with pytest.raises(TypeError, match="generate_anthropic"):
        async with sandbox_model_proxy(sandbox_env, methods=methods):
            pass
    assert sandbox_env.method_calls == []


async def test_tool_methods_with_bridged_tools_are_refused() -> None:
    async def list_tools(server: str) -> JsonValue:
        return []

    async def call_tool(server: str, tool: str, arguments: JsonValue) -> JsonValue:
        return ""

    with pytest.raises(ValueError, match="not both"):
        async with sandbox_model_proxy(
            _unused_sandbox(),
            methods={"list_tools": list_tools, "call_tool": call_tool},
            bridged_tools=[BridgedToolsSpec(name="calc", tools=[calculator_add([])])],
        ):
            pass


async def test_one_tool_method_without_the_other_is_refused() -> None:
    async def list_tools(server: str) -> JsonValue:
        return []

    with pytest.raises(ValueError, match="counterpart"):
        async with sandbox_model_proxy(
            _unused_sandbox(), methods={"list_tools": list_tools}
        ):
            pass


# ---------------------------------------------------------------------------
# The service method table the proxy is given (proxy runner replaced)
# ---------------------------------------------------------------------------


class _ServedTable:
    """Captures what `sandbox_model_proxy` / `sandbox_agent_bridge` hand the proxy runner."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
        self.methods: dict[str, SandboxServiceMethod] = {}
        self.kwargs: dict[str, Any] = {}

        async def inject(**kwargs: Any) -> SandboxEnvironment:
            return _unused_sandbox()

        @contextlib.asynccontextmanager
        async def runner(
            sandbox_env: SandboxEnvironment,
            methods: dict[str, SandboxServiceMethod],
            bridge: SandboxAgentBridge,
            **kwargs: Any,
        ) -> AsyncIterator[None]:
            self.methods = methods
            self.kwargs = kwargs
            yield

        monkeypatch.setattr(module, "sandbox_with_injected_tools", inject)
        monkeypatch.setattr(module, "_model_proxy_service", runner)


async def test_sandbox_agent_bridge_serves_inspect_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served = _ServedTable(monkeypatch, bridge_module)
    async with sandbox_agent_bridge(model="mockllm/model"):
        assert set(served.methods) == GENERATE_METHODS | {"list_tools", "call_tool"}
        assert served.kwargs["polling_interval"] == 2
        result = await served.methods["generate_anthropic"](
            json_data={
                "model": "claude-any",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    # answered by Inspect's model layer (mockllm), not passed through
    assert isinstance(result, dict)
    assert result["type"] == "message"
    content = result["content"]
    assert isinstance(content, list) and isinstance(content[0], dict)
    assert content[0]["text"] == "Default output from mockllm/model"


async def test_seam_serves_every_generate_method_and_its_bridged_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served = _ServedTable(monkeypatch, proxy_module)
    async with sandbox_model_proxy(
        _unused_sandbox(),
        methods={"generate_anthropic": _ok},
        port=14141,
        bridged_tools=[BridgedToolsSpec(name="calc", tools=[calculator_add([])])],
        polling_interval=0.5,
    ) as model_proxy:
        assert set(served.methods) == GENERATE_METHODS | {"list_tools", "call_tool"}
        assert served.kwargs["polling_interval"] == 0.5
        assert model_proxy == ModelProxy(
            port=14141, mcp_server_configs=model_proxy.mcp_server_configs
        )
        assert [c.url for c in model_proxy.mcp_server_configs] == [
            "http://localhost:14141/mcp/calc"
        ]
        listed = await served.methods["list_tools"](server="calc")
        assert isinstance(listed, list) and isinstance(listed[0], dict)
        assert (listed[0]["name"], listed[0]["description"]) == (
            "calculator_add",
            ADD_DESCRIPTION,
        )


async def test_seam_serves_caller_tool_methods_without_bridged_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def list_tools(server: str) -> JsonValue:
        return [{"name": f"{server}-tool"}]

    async def call_tool(server: str, tool: str, arguments: JsonValue) -> JsonValue:
        return f"{server}/{tool}"

    served = _ServedTable(monkeypatch, proxy_module)
    async with sandbox_model_proxy(
        _unused_sandbox(), methods={"list_tools": list_tools, "call_tool": call_tool}
    ) as model_proxy:
        assert model_proxy.mcp_server_configs == []
        assert await served.methods["list_tools"](server="s") == [{"name": "s-tool"}]
        assert (
            await served.methods["call_tool"](server="s", tool="t", arguments={})
            == "s/t"
        )


# ---------------------------------------------------------------------------
# Explicit raw HTTP mode: no Model, provider transformations, or implicit grants
# ---------------------------------------------------------------------------


def _raw_methods() -> dict[str, proxy_module.ModelProxyMethod]:
    async def acknowledge(**params: JsonValue) -> JsonValue:
        return {"version": 1}

    return {name: acknowledge for name in proxy_module._RAW_HTTP_METHODS}


@pytest.mark.parametrize("invalid", ["missing", "generate", "tools", "bridged"])
async def test_raw_method_contract_is_refused_before_injection(invalid: str) -> None:
    environment = _unused_sandbox()
    methods = _raw_methods()
    bridged = None
    if invalid == "missing":
        del methods["raw_http_close"]
    elif invalid == "generate":
        methods["generate_anthropic"] = _ok
    elif invalid == "tools":
        methods["list_tools"] = _ok
        methods["call_tool"] = _ok
    else:
        bridged = [BridgedToolsSpec(name="calc", tools=[calculator_add([])])]
    with pytest.raises(ValueError):
        async with sandbox_model_proxy(
            environment, methods=methods, bridged_tools=bridged, raw_http=True
        ):
            pytest.fail("Invalid raw method contract was accepted")
    assert environment.method_calls == []


@pytest.mark.parametrize(
    "stdout,success",
    [("", False), ("invalid", True), ("{}", True),
     ('{"raw_http_version":true}', True), ('{"raw_http_version":2}', True)],
)
async def test_raw_capability_requires_the_installed_binary(
    monkeypatch: pytest.MonkeyPatch, stdout: str, success: bool
) -> None:
    environment = _unused_sandbox()
    environment.exec = AsyncMock(
        return_value=ExecResult(
            success=success, returncode=0 if success else 2, stdout=stdout, stderr=""
        )
    )
    monkeypatch.setattr(
        proxy_module, "sandbox_with_injected_tools", AsyncMock(return_value=environment)
    )
    runner = MagicMock()
    monkeypatch.setattr(proxy_module, "_model_proxy_service", runner)
    with pytest.raises(RuntimeError, match="bundle does not support"):
        async with sandbox_model_proxy(environment, methods=_raw_methods(), raw_http=True):
            pytest.fail("Unsupported raw binary was accepted")
    runner.assert_not_called()


@pytest.mark.parametrize("ready", ["accepted", "missing", "rejected", "wrong_version"])
async def test_raw_public_proxy_requires_handshake_without_model_state(
    monkeypatch: pytest.MonkeyPatch, ready: str
) -> None:
    environment = _unused_sandbox()
    environment._tools_user = "root"
    environment.exec = AsyncMock(
        return_value=ExecResult(
            success=True, returncode=0, stdout='{"raw_http_version":1}', stderr=""
        )
    )
    methods = _raw_methods()
    methods["raw_http_ready"] = AsyncMock(
        return_value={"version": 2 if ready == "rejected" else 1}
    )
    response: JsonValue = {
        "version": 1, "status": 529, "response_id": "opaque",
        "headers": [["x-provider-field", "first"], ["x-provider-field", "second"]],
    }
    methods["raw_http_start"] = AsyncMock(return_value=response)

    @contextlib.asynccontextmanager
    async def runner(sandbox_env, served, bridge, **kwargs):
        assert sandbox_env is environment
        assert bridge is None
        assert kwargs["raw_http"] is True
        assert set(served) == set(proxy_module._RAW_HTTP_METHODS)
        if ready != "missing":
            await served["raw_http_ready"](version=2 if ready == "wrong_version" else 1)
        # No ModelProxyError/status conversion is introduced in the raw path.
        assert await served["raw_http_start"](
            version=1, method="POST", path="/v1/messages", headers=[], body_b64="AA=="
        ) is response
        yield

    monkeypatch.setattr(
        proxy_module, "sandbox_with_injected_tools", AsyncMock(return_value=environment)
    )
    monkeypatch.setattr(proxy_module, "_model_proxy_service", runner)
    monkeypatch.setattr(proxy_module, "_RAW_HTTP_START_TIMEOUT", 0.01)
    monkeypatch.setattr(
        proxy_module, "AgentState", MagicMock(side_effect=AssertionError("agent state"))
    )
    monkeypatch.setattr(
        proxy_module, "SandboxAgentBridge", MagicMock(side_effect=AssertionError("bridge"))
    )
    if ready == "accepted":
        async with sandbox_model_proxy(
            environment, methods=methods, port=14141, raw_http=True
        ) as proxy:
            assert proxy == ModelProxy(port=14141)
        environment.exec.assert_awaited_once_with(
            [proxy_module.SANDBOX_CLI, "model_proxy", "--capabilities"],
            user="root", timeout=30,
        )
    else:
        with pytest.raises(RuntimeError, match="protocol"):
            async with sandbox_model_proxy(environment, methods=methods, raw_http=True):
                pytest.fail("Raw proxy yielded without its protocol handshake")


# ---------------------------------------------------------------------------
# Generate handlers: errors are forwarded, the proxy is never failed
# ---------------------------------------------------------------------------


class _StatusError(Exception):
    status_code = 503


@pytest.mark.parametrize(
    "raised,payload",
    [
        (
            ModelProxyError(529, "overloaded", body={"code": "busy"}),
            {"status": 529, "message": "overloaded", "body": {"code": "busy"}},
        ),
        (ModelProxyError(429, "slow down"), {"status": 429, "message": "slow down"}),
        (
            _StatusError("gateway unavailable"),
            {"status": 503, "message": "gateway unavailable"},
        ),
        (RuntimeError("handler bug"), {"status": None, "message": "handler bug"}),
    ],
)
async def test_handler_exception_becomes_provider_error(
    monkeypatch: pytest.MonkeyPatch, raised: Exception, payload: dict[str, JsonValue]
) -> None:
    async def failing(json_data: dict[str, JsonValue], **_: JsonValue) -> JsonValue:
        raise raised

    served = _ServedTable(monkeypatch, proxy_module)
    async with sandbox_model_proxy(
        _unused_sandbox(), methods={"generate_responses": failing}
    ):
        result = await served.methods["generate_responses"](json_data={"model": "m"})
    assert result == {PROVIDER_ERROR_KEY: payload}


async def test_handler_limit_exceeded_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def over_limit(json_data: dict[str, JsonValue], **_: JsonValue) -> JsonValue:
        raise LimitExceededError("token", value=10, limit=5)

    served = _ServedTable(monkeypatch, proxy_module)
    async with sandbox_model_proxy(
        _unused_sandbox(), methods={"generate_anthropic": over_limit}
    ):
        with pytest.raises(LimitExceededError):
            await served.methods["generate_anthropic"](json_data={"model": "m"})


async def test_missing_generate_method_answers_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served = _ServedTable(monkeypatch, proxy_module)
    async with sandbox_model_proxy(
        _unused_sandbox(), methods={"generate_anthropic": _ok}
    ):
        result = await served.methods["generate_completions"](
            json_data={"model": "m"}, headers={"x-api-key": "placeholder"}
        )
    assert result == {
        PROVIDER_ERROR_KEY: {
            "status": 404,
            "message": "This model proxy does not serve the OpenAI Chat Completions API.",
        }
    }


async def test_handler_receives_proxy_params_and_result_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, JsonValue]] = []
    error_result: JsonValue = {PROVIDER_ERROR_KEY: {"status": 418, "message": "teapot"}}

    async def recording(**params: JsonValue) -> JsonValue:
        received.append(params)
        return error_result if params.get("headers") else FIXED_MESSAGE

    served = _ServedTable(monkeypatch, proxy_module)
    async with sandbox_model_proxy(
        _unused_sandbox(), methods={"generate_anthropic": recording}
    ):
        method = served.methods["generate_anthropic"]
        assert await method(json_data={"model": "m"}) == FIXED_MESSAGE
        assert (
            await method(
                json_data={"model": "m"},
                headers={"anthropic-beta": "b"},
                metadata_headers=None,
            )
            == error_result
        )
    assert received == [
        {"json_data": {"model": "m"}},
        {
            "json_data": {"model": "m"},
            "headers": {"anthropic-beta": "b"},
            "metadata_headers": None,
        },
    ]


# ---------------------------------------------------------------------------
# Execution grants minted from raw provider responses
# ---------------------------------------------------------------------------

_ADD_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
}

# (generate method, request declaring the bridged tool, response proposing a call)
_DIALECTS: list[tuple[str, dict[str, JsonValue], dict[str, JsonValue]]] = [
    (
        "generate_anthropic",
        {
            "model": "m",
            "tools": [
                {
                    "name": "mcp__calc__calculator_add",
                    "description": ADD_DESCRIPTION,
                    "input_schema": _ADD_SCHEMA,
                }
            ],
        },
        {
            "content": [
                {"type": "text", "text": "adding"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "mcp__calc__calculator_add",
                    "input": {"x": 2, "y": 3},
                },
            ]
        },
    ),
    (
        "generate_completions",
        {
            "model": "m",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calc_add",
                        "description": ADD_DESCRIPTION,
                        "parameters": _ADD_SCHEMA,
                    },
                }
            ],
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "calc_add",
                                    "arguments": '{"x": 2, "y": 3}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
    ),
    (
        "generate_responses",
        {
            "model": "m",
            "tools": [
                {
                    "type": "function",
                    "name": "calc_add",
                    "description": ADD_DESCRIPTION,
                    "parameters": _ADD_SCHEMA,
                }
            ],
        },
        {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "calc_add",
                    "arguments": '{"x": 2, "y": 3}',
                }
            ]
        },
    ),
    (
        "generate_google",
        {
            "model": "m",
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "calc_add",
                            "description": ADD_DESCRIPTION,
                            "parameters": _ADD_SCHEMA,
                        }
                    ]
                }
            ],
        },
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "calc_add",
                                    "args": {"x": 2, "y": 3},
                                }
                            }
                        ],
                    }
                }
            ]
        },
    ),
]


@pytest.mark.parametrize("method,request_body,response", _DIALECTS)
async def test_raw_response_grants_one_execution_of_the_proposed_call(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    request_body: dict[str, JsonValue],
    response: dict[str, JsonValue],
) -> None:
    call_log: list[dict[str, int]] = []

    async def raw(json_data: dict[str, JsonValue], **_: JsonValue) -> JsonValue:
        return response

    served = _ServedTable(monkeypatch, proxy_module)
    async with sandbox_model_proxy(
        _unused_sandbox(),
        methods={method: raw},
        bridged_tools=[BridgedToolsSpec(name="calc", tools=[calculator_add(call_log)])],
    ):
        call_tool = served.methods["call_tool"]
        arguments = {"x": 2, "y": 3}

        # nothing proposed yet
        with pytest.raises(PermissionError, match="not proposed"):
            await call_tool(server="calc", tool="calculator_add", arguments=arguments)

        # the raw response is returned unchanged and grants the proposed call once
        assert await served.methods[method](json_data=request_body) == response
        with pytest.raises(PermissionError):
            await call_tool(
                server="calc", tool="calculator_add", arguments={"x": 2, "y": 4}
            )
        assert (
            await call_tool(server="calc", tool="calculator_add", arguments=arguments)
            == "5"
        )
        with pytest.raises(PermissionError):
            await call_tool(server="calc", tool="calculator_add", arguments=arguments)

    assert call_log == [{"x": 2, "y": 3}]


async def test_call_to_an_undeclared_name_grants_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, request_body, response = _DIALECTS[0]
    undeclared = {**request_body, "tools": []}

    async def raw(json_data: dict[str, JsonValue], **_: JsonValue) -> JsonValue:
        return response

    served = _ServedTable(monkeypatch, proxy_module)
    async with sandbox_model_proxy(
        _unused_sandbox(),
        methods={method: raw},
        bridged_tools=[BridgedToolsSpec(name="calc", tools=[calculator_add([])])],
    ):
        assert await served.methods[method](json_data=undeclared) == response
        with pytest.raises(PermissionError):
            await served.methods["call_tool"](
                server="calc", tool="calculator_add", arguments={"x": 2, "y": 3}
            )


async def test_provider_error_result_grants_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, request_body, response = _DIALECTS[0]
    error_with_calls: dict[str, JsonValue] = {
        **response,
        PROVIDER_ERROR_KEY: {"status": 500, "message": "x"},
    }

    async def raw(json_data: dict[str, JsonValue], **_: JsonValue) -> JsonValue:
        return error_with_calls

    served = _ServedTable(monkeypatch, proxy_module)
    async with sandbox_model_proxy(
        _unused_sandbox(),
        methods={method: raw},
        bridged_tools=[BridgedToolsSpec(name="calc", tools=[calculator_add([])])],
    ):
        await served.methods[method](json_data=request_body)
        with pytest.raises(PermissionError):
            await served.methods["call_tool"](
                server="calc", tool="calculator_add", arguments={"x": 2, "y": 3}
            )


# ---------------------------------------------------------------------------
# End to end in a Docker sandbox
# ---------------------------------------------------------------------------

PORT = 13131


async def _curl(
    path: str, body: dict[str, JsonValue] | None = None, *, method: str = "POST"
) -> tuple[int, str]:
    """POST `body` to the proxy from inside the sandbox; (HTTP status, response body)."""
    cmd = [
        "curl",
        "-sS",
        "--retry",
        "30",
        "--retry-connrefused",
        "--retry-delay",
        "1",
        "-X",
        method,
        "-H",
        "Content-Type: application/json",
        "-w",
        "\n%{http_code}",
        f"http://localhost:{PORT}{path}",
    ]
    if body is not None:
        cmd[-1:-1] = ["-d", json.dumps(body)]
    result = await sandbox().exec(cmd, timeout=120)
    assert result.success, result.stderr
    response, _, status = result.stdout.rpartition("\n")
    return int(status), response


async def _mcp(
    method: str, params: dict[str, JsonValue] | None = None
) -> dict[str, Any]:
    request: dict[str, JsonValue] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        request["params"] = params
    status, response = await _curl("/mcp/calc", request)
    assert status == 200, response
    parsed: dict[str, Any] = json.loads(response)
    return parsed


def _anthropic_request(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }


def _parse_with_anthropic_sdk(sse: str) -> Any:
    """Read `sse` as the Anthropic SDK would read a streamed Messages response."""
    import anthropic
    import httpx2  # the HTTP client the anthropic SDK is built on

    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse.encode("utf-8"),
        )

    client = anthropic.Anthropic(
        api_key="placeholder",
        base_url=f"http://localhost:{PORT}",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    )
    with client.messages.stream(
        model="claude-fixed",
        max_tokens=64,
        messages=[{"role": "user", "content": "hi"}],
    ) as stream:
        return stream.get_final_message()


@task
def proxy_task(test_solver: Any) -> Task:
    return Task(
        dataset=[Sample(input="Test", target="Test")],
        solver=[test_solver],
        scorer=includes(),
        sandbox="docker",
    )


@skip_if_no_docker
@pytest.mark.slow
def test_sandbox_model_proxy_end_to_end() -> None:
    call_log: list[dict[str, int]] = []
    handled: list[str] = []
    checks: dict[str, Any] = {}

    async def generate_anthropic(
        json_data: dict[str, JsonValue], **_: JsonValue
    ) -> JsonValue:
        model = json_data["model"]
        assert isinstance(model, str)
        handled.append(model)
        match model:
            case "overloaded":
                raise ModelProxyError(529, "overloaded right now")
            case "bug":
                raise RuntimeError("the handler has a bug")
            case "tool":
                return {
                    **FIXED_MESSAGE,
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "mcp__calc__calculator_add",
                            "input": {"x": 5, "y": 3},
                        }
                    ],
                    "stop_reason": "tool_use",
                }
            case _:
                return FIXED_MESSAGE

    @solver
    def proxy_solver() -> Any:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            async with sandbox_model_proxy(
                sandbox(),
                methods={"generate_anthropic": generate_anthropic},
                port=PORT,
                bridged_tools=[
                    BridgedToolsSpec(name="calc", tools=[calculator_add(call_log)])
                ],
            ) as model_proxy:
                checks["mcp_urls"] = [c.url for c in model_proxy.mcp_server_configs]

                # a non-streaming request returns the handler's dict unchanged
                checks["fixed"] = await _curl(
                    "/v1/messages", _anthropic_request("fixed")
                )

                # a streaming request gets server-sent events built by the proxy
                checks["stream"] = await _curl(
                    "/v1/messages", _anthropic_request("fixed", stream=True)
                )

                # a raising handler yields a provider-shaped error; the proxy stays up
                checks["overloaded"] = await _curl(
                    "/v1/messages", _anthropic_request("overloaded")
                )
                checks["bug"] = await _curl("/v1/messages", _anthropic_request("bug"))
                checks["missing"] = await _curl(
                    "/v1/chat/completions",
                    {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                )
                checks["after_errors"] = await _curl(
                    "/v1/messages", _anthropic_request("fixed")
                )

                # bridged tools: listed, denied until proposed, run once per proposal
                checks["tools_list"] = await _mcp("tools/list")
                call: dict[str, JsonValue] = {
                    "name": "calculator_add",
                    "arguments": {"x": 5, "y": 3},
                }
                checks["call_unproposed"] = await _mcp("tools/call", call)
                checks["propose"] = await _curl(
                    "/v1/messages",
                    _anthropic_request(
                        "tool",
                        tools=[
                            {
                                "name": "mcp__calc__calculator_add",
                                "description": ADD_DESCRIPTION,
                                "input_schema": _ADD_SCHEMA,
                            }
                        ],
                    ),
                )
                checks["call_proposed"] = await _mcp("tools/call", call)
                checks["call_again"] = await _mcp("tools/call", call)
            return state

        return solve

    log = eval(proxy_task(proxy_solver()), model=get_model("mockllm/model"))[0]
    assert log.status == "success", log.error

    assert checks["mcp_urls"] == [f"http://localhost:{PORT}/mcp/calc"]

    assert checks["fixed"] == (200, json.dumps(FIXED_MESSAGE))

    status, sse = checks["stream"]
    assert status == 200
    assert "event: message_start" in sse and "event: message_stop" in sse
    message = _parse_with_anthropic_sdk(sse)
    # message_start carries the result's id and model in proxy builds that wait
    # for the result before starting the stream; upstream's build starts it
    # earlier, with a generated id and the request's model
    assert (message.id, message.model) in {
        (FIXED_MESSAGE["id"], FIXED_MESSAGE["model"]),
        (message.id, "fixed"),
    }
    assert message.id.startswith("msg_")
    assert message.stop_reason == "end_turn"
    assert [block.text for block in message.content] == [
        "The proxy returned this unchanged."
    ]
    assert message.usage.output_tokens == 7

    assert checks["overloaded"] == (
        529,
        json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "overloaded right now",
                },
            }
        ),
    )
    status, body = checks["bug"]
    assert status == 400
    assert json.loads(body)["error"]["message"] == "the handler has a bug"
    status, body = checks["missing"]
    assert status == 404
    assert json.loads(body)["error"]["message"] == (
        "This model proxy does not serve the OpenAI Chat Completions API."
    )
    assert checks["after_errors"] == (200, json.dumps(FIXED_MESSAGE))
    # the chat completions request never reached a handler
    assert handled == ["fixed", "fixed", "overloaded", "bug", "fixed", "tool"]

    assert [t["name"] for t in checks["tools_list"]["result"]["tools"]] == [
        "calculator_add"
    ]
    assert "was not proposed" in checks["call_unproposed"]["error"]["message"]
    assert checks["propose"][0] == 200
    assert json.loads(checks["propose"][1])["content"][0]["type"] == "tool_use"
    assert checks["call_proposed"]["result"]["content"] == [
        {"type": "text", "text": "8"}
    ]
    assert "was not proposed" in checks["call_again"]["error"]["message"]
    assert call_log == [{"x": 5, "y": 3}]
