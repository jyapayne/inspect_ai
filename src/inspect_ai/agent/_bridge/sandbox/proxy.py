import contextlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from logging import getLogger
from typing import AsyncIterator, cast

import anyio
from pydantic import JsonValue

from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.tool._mcp._tools_bridge import BridgedToolsSpec
from inspect_ai.tool._sandbox_tools_utils.sandbox import sandbox_with_injected_tools
from inspect_ai.tool._tool_call import ToolCall
from inspect_ai.tool._tool_info import ToolInfo
from inspect_ai.util._limit import LimitExceededError
from inspect_ai.util._sandbox._cli import SANDBOX_CLI
from inspect_ai.util._sandbox.environment import SandboxEnvironment
from inspect_ai.util._sandbox.service import SandboxServiceMethod

from ..._agent import AgentState
from .._errors import PROVIDER_ERROR_KEY, provider_error_payload
from .bridge import _model_proxy_service, _register_bridged_tool_specs
from .service import call_tool, list_tools
from .types import SandboxAgentBridge

logger = getLogger(__name__)

ModelProxyMethod = Callable[..., Awaitable[JsonValue]]
"""A `sandbox_model_proxy` handler, called with the keyword arguments the proxy sends."""

_GENERATE_METHODS = {
    "generate_completions": "the OpenAI Chat Completions API",
    "generate_responses": "the OpenAI Responses API",
    "generate_anthropic": "the Anthropic Messages API",
    "generate_google": "the Google Gemini API",
}
_TOOL_METHODS = ("list_tools", "call_tool")
_RAW_HTTP_METHODS = (
    "raw_http_ready",
    "raw_http_start",
    "raw_http_read",
    "raw_http_close",
)
_RAW_HTTP_START_TIMEOUT = 30


@dataclass(frozen=True)
class ModelProxy:
    """A model proxy running inside a sandbox (see `sandbox_model_proxy`)."""

    port: int
    """Port the proxy listens on at `localhost` inside the sandbox."""

    mcp_server_configs: list[MCPServerConfigHTTP] = field(default_factory=list)
    """MCP server configs for the bridged tools, one per `BridgedToolsSpec`."""


class ModelProxyError(Exception):
    """Raised by a `sandbox_model_proxy` handler to answer with a provider error.

    The proxy responds with `status_code` and the API's own error body built
    from `message`. Proxy builds that forward it merge `body` into the error
    object of the two OpenAI APIs (clients read fields such as `code` from it);
    the Anthropic and Gemini routes do not use it.
    """

    def __init__(
        self, status_code: int, message: str, body: dict[str, JsonValue] | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.body = body


@contextlib.asynccontextmanager
async def sandbox_model_proxy(
    sandbox: SandboxEnvironment,
    *,
    methods: Mapping[str, ModelProxyMethod],
    port: int = 13131,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    polling_interval: float | None = None,
    raw_http: bool = False,
) -> AsyncIterator[ModelProxy]:
    """Run the agent bridge's in-sandbox model proxy with caller-supplied handlers.

    Injects Inspect's sandbox tools into `sandbox` and starts their `model_proxy`
    process, which serves the OpenAI Chat Completions (`/v1/chat/completions`),
    OpenAI Responses (`/v1/responses`), Anthropic Messages (`/v1/messages`) and
    Google Gemini (`/v1beta/models/*`, `/models/*`) APIs on `localhost:<port>`
    inside the sandbox. Each request is passed to the matching handler in
    `methods` on the host, over the sandbox service file queue, and the handler's
    result is returned to the client. This is the transport `sandbox_agent_bridge`
    uses, without Inspect's model layer: a handler may forward the request
    anywhere, for example unchanged to a model gateway. Point an agent in the
    sandbox at it with `OPENAI_BASE_URL=http://localhost:<port>/v1`,
    `ANTHROPIC_BASE_URL=http://localhost:<port>` or
    `GOOGLE_GEMINI_BASE_URL=http://localhost:<port>` and a placeholder API key.

    `bridged_tools` are served as MCP servers at
    `http://localhost:<port>/mcp/<name>` (`ModelProxy.mcp_server_configs`) under
    `sandbox_agent_bridge`'s rule: a bridged tool runs only for a call the model
    proposed, once per proposal, unless its spec sets `require_proposal=False`.
    Proposals are read from each successful generate result: a tool call in it
    (an Anthropic `tool_use` block, a Chat Completions `tool_calls` entry, a
    Responses `function_call` item, a Gemini `functionCall` part) whose name the
    request declared as a function tool, with a description matching a bridged
    tool's, grants that tool one execution with exactly those arguments. A tool
    declared any other way (a Responses namespace or `tool_search` result)
    grants nothing, so a call to it is denied.

    Handler contract, read from the proxy's source
    (`inspect_sandbox_tools/_agent_bridge/proxy.py`; line numbers are those of
    the file as built into sandbox tools v32-tl1):

    - Arguments: the host calls `handler(**params)` with the keyword arguments
      the proxy filed: always `json_data`, the client's JSON request body.
      Proxy builds that forward client headers also pass `headers` (every
      client request header, names lower-cased, including the client's API key
      header) and `metadata_headers` (the headers named in the proxy's
      `BRIDGE_MODEL_EVENT_METADATA_HEADERS`, else `None`) to the Chat
      Completions, Responses and Anthropic methods, and `metadata_headers` alone
      to `generate_google` (proxy.py:825-830, 1575-1580, 1812-1818, 2148-2154,
      2198-2202). Handlers should accept those keyword arguments.
    - Request body: as the client sent it, except that the Chat Completions
      route sets `parallel_tool_calls` to `false` (proxy.py:1573) and the
      Gemini routes set `model` from the URL path (proxy.py:2196). `stream` is
      left as the client sent it; the handler returns one complete response
      either way. The proxy itself answers 400 to a request without `model`
      (or `messages`, `input`), and 415/400 to a POST body that is not a JSON
      object, without calling the handler (proxy.py:391-418, 819-822,
      1562-1565, 1793-1796); bodies are limited to 50 MiB (proxy.py:36).
    - Result: the API's non-streaming response object (a Chat Completion, a
      Response, an Anthropic Message, a Gemini `GenerateContentResponse`) as a
      JSON dict. A non-streaming client receives it serialized with
      `json.dumps` (proxy.py:252, 1551, 1783, 2164, 2217).
    - Streaming: for a request with `stream: true`, or a Gemini
      `:streamGenerateContent` path, the proxy builds server-sent events from
      the complete result. Chat Completions: per choice a role chunk,
      48-character content chunks, function and tool call chunks and a finish
      chunk, a usage chunk if `stream_options.include_usage`, then
      `data: [DONE]` (proxy.py:1597-1771). Responses: `response.created`,
      `response.in_progress`, each output item as `response.output_item.added`,
      its type's delta and done events and `response.output_item.done`, then
      `response.completed` (proxy.py:847-1539). Anthropic: a `ping` every 5
      seconds while the handler runs, then `message_start` (with the result's
      `id` and `model`, both required, and zero usage), each content block as
      start, deltas and stop, `message_delta` (stop reason and the result's
      usage) and `message_stop` (proxy.py:1801-2137); only `text`, `tool_use`,
      `server_tool_use`, `web_search_tool_result`, `thinking`, `compaction` and
      `fallback` blocks are streamed, and a block of any other type (such as
      `redacted_thinking`) is left out (proxy.py:1896-2099). Upstream's build
      instead sends `message_start` before the handler returns, with a
      generated `id` and the request's `model`. Gemini: the whole result as one
      `data:` event (proxy.py:2219-2224).
    - Provider errors: a result `{"__inspect_provider_error__": {"status": int
      | None, "message": str, "body": dict}}` is answered with that status (400
      when it is `None`) and the API's error body built from `message`; `body`
      (optional) is merged into the error object of the two OpenAI APIs only
      (proxy.py:603-700, 834-843). On an Anthropic stream the error is an
      `error` event after a 200 status (proxy.py:1840-1847). A handler may
      return that payload itself. A handler that raises `ModelProxyError` gets
      that payload with its status, message and body; any other exception gets
      it with the status recovered from the exception (its integer
      `status_code` or `code`, else `None`), except `LimitExceededError`, which
      propagates.
    - Failures are fatal to the proxy: an error that reaches the service queue
      as an RPC error makes a generate route exit the proxy process
      (proxy.py:2383-2400, 2447, 1555), which fails this block. This function
      returns a provider error instead for every generate method, including one
      missing from `methods` (answered with status 404).
    - Tools: MCP `tools/list` calls `list_tools(server=...)` and `tools/call`
      calls `call_tool(server=..., tool=..., arguments=...)`; a string result
      becomes one text content block, and an RPC error becomes a JSON-RPC error
      without stopping the proxy (proxy.py:2277-2356).
    - Transport: the proxy listens on `BRIDGE_MODEL_SERVICE_PORT`
      (proxy.py:2368), files requests under
      `/var/tmp/sandbox-services/bridge_model_service/<instance>/`
      (proxy.py:2466-2470) and polls for each response every 0.1 seconds
      (proxy.py:2408); the host polls for requests every `polling_interval`.

    Args:
        sandbox: Sandbox to run the proxy in. It need not belong to an Inspect
            sample.
        methods: Handlers by service method name: any of `generate_completions`,
            `generate_responses`, `generate_anthropic` and `generate_google`,
            and, only without `bridged_tools`, both `list_tools` and
            `call_tool` (served as given, with no execution grants).
        port: Port the proxy listens on inside the sandbox.
        bridged_tools: Host-side Inspect tools to expose to the sandboxed agent
            over MCP, as for `sandbox_agent_bridge`.
        polling_interval: Seconds between the host's polls of the sandbox for
            requests. Defaults to the sandbox's own default (0.2 for Docker, 2
            otherwise); a smaller value is raised to that default.
        raw_http: Opt in to byte-preserving HTTP transport, not model generation.
            Requires exactly the four ``raw_http_ready``, ``raw_http_start``,
            ``raw_http_read`` and ``raw_http_close`` handlers implementing raw
            protocol v1, and a matching sandbox-tools binary. In this mode none
            of the generate-handler transformations documented above apply:
            response status, end-to-end header pairs and entity bytes (including
            real SSE chunks) pass through without provider parsing. Transfer
            framing is rebuilt. The host owns authorization, credentials, model
            policy and tool grants, including any MCP routes. ``bridged_tools``
            and generate/tool handlers are refused rather than silently ignored.
            Checks the selected binary's capability before starting the proxy
            and waits up to 30 seconds for its bound-loopback v1 handshake before
            yielding. No Inspect Model or agent state is constructed.

    Yields:
        The running proxy, stopped when the block exits.

    Raises:
        ValueError: `methods` names an unknown method, only one of `list_tools`
            and `call_tool`, or either of them together with `bridged_tools`;
            or two `bridged_tools` specs share a name.
        TypeError: A value in `methods` is not callable.
        RuntimeError: The proxy process exits while the block runs.
    """
    _validate_methods(methods, bridged_tools, raw_http=raw_http)

    if raw_http:
        sandbox_env = await sandbox_with_injected_tools(sandbox=sandbox)
        await _require_raw_http(sandbox_env)
        ready = anyio.Event()

        async def raw_ready(*, version: JsonValue) -> JsonValue:
            if type(version) is not int or version != 1:
                raise RuntimeError("Unsupported raw HTTP proxy protocol")
            result = await methods["raw_http_ready"](version=version)
            if (
                not isinstance(result, dict)
                or type(result.get("version")) is not int
                or result["version"] != 1
                or "error" in result
            ):
                raise RuntimeError("Host rejected raw HTTP proxy protocol v1")
            ready.set()
            return result

        async with _model_proxy_service(
            sandbox_env,
            {**methods, "raw_http_ready": raw_ready},
            None,
            port=port,
            polling_interval=polling_interval,
            caller="sandbox_model_proxy",
            raw_http=True,
        ):
            try:
                with anyio.fail_after(_RAW_HTTP_START_TIMEOUT):
                    await ready.wait()
            except TimeoutError as ex:
                raise RuntimeError(
                    "Sandbox tools did not acknowledge raw HTTP protocol v1"
                ) from ex
            yield ModelProxy(port=port)
        return

    # execution grant bookkeeping for bridged tools, and the channel through
    # which a failing host tool fails the block (`SandboxAgentBridge.request_fail`)
    bridge = SandboxAgentBridge(
        state=AgentState(messages=[]),
        filter=None,
        retry_refusals=None,
        compaction=None,
        port=port,
        model=None,
    )
    _register_bridged_tool_specs(bridge, bridged_tools or [], port)

    service_methods: dict[str, SandboxServiceMethod] = {
        method: _serve_generate(method, methods.get(method), bridge)
        for method in _GENERATE_METHODS
    }
    if bridged_tools:
        service_methods["list_tools"] = list_tools(bridge)
        service_methods["call_tool"] = call_tool(bridge)
    else:
        service_methods.update(
            {name: methods[name] for name in _TOOL_METHODS if name in methods}
        )

    sandbox_env = await sandbox_with_injected_tools(sandbox=sandbox)
    async with _model_proxy_service(
        sandbox_env,
        service_methods,
        bridge,
        port=port,
        polling_interval=polling_interval,
        caller="sandbox_model_proxy",
    ):
        yield ModelProxy(port=port, mcp_server_configs=list(bridge.mcp_server_configs))


def _validate_methods(
    methods: Mapping[str, ModelProxyMethod],
    bridged_tools: Sequence[BridgedToolsSpec] | None,
    *,
    raw_http: bool = False,
) -> None:
    supported = (
        list(_RAW_HTTP_METHODS) if raw_http else [*_GENERATE_METHODS, *_TOOL_METHODS]
    )
    unknown = sorted(name for name in methods if name not in supported)
    if unknown:
        raise ValueError(
            f"Unknown model proxy method(s): {', '.join(unknown)} "
            f"(supported: {', '.join(supported)})."
        )
    for name, handler in methods.items():
        if not callable(handler):
            raise TypeError(f"Model proxy method '{name}' is not callable.")
    if raw_http:
        missing = sorted(name for name in _RAW_HTTP_METHODS if name not in methods)
        if missing:
            raise ValueError(f"Missing raw HTTP proxy method(s): {', '.join(missing)}.")
        if bridged_tools:
            raise ValueError(
                "Raw HTTP hosts must own tool grants; bridged_tools is not supported."
            )
        return
    tool_methods = [name for name in _TOOL_METHODS if name in methods]
    if tool_methods and bridged_tools:
        raise ValueError(
            "Pass either bridged_tools or the list_tools and call_tool methods, "
            "not both: bridged_tools are served by the proxy's own list_tools "
            "and call_tool."
        )
    if len(tool_methods) == 1:
        raise ValueError(
            f"Model proxy method '{tool_methods[0]}' needs its counterpart: pass "
            "both list_tools and call_tool, or neither."
        )


async def _require_raw_http(sandbox: SandboxEnvironment) -> None:
    """Require a capability of the installed binary, not a source/version label."""
    result = await sandbox.exec(
        [SANDBOX_CLI, "model_proxy", "--capabilities"],
        user=sandbox._tools_user,
        timeout=30,
    )
    try:
        capabilities = json.loads(result.stdout) if result.success else None
    except json.JSONDecodeError:
        capabilities = None
    if (
        not isinstance(capabilities, dict)
        or type(capabilities.get("raw_http_version")) is not int
        or capabilities["raw_http_version"] != 1
    ):
        raise RuntimeError(
            "The installed sandbox-tools bundle does not support raw HTTP v1. "
            "Build and install the complete bundle from the aligned SDK source; "
            "a matching package version alone is insufficient."
        )


def _serve_generate(
    method: str, handler: ModelProxyMethod | None, bridge: SandboxAgentBridge
) -> SandboxServiceMethod:
    """The service method for `method`: `handler` with errors forwarded and grants minted.

    An exception from `handler` (other than `LimitExceededError`, which ends a
    sample) is returned as a provider error so the proxy answers the client and
    stays up; a missing handler answers 404. Tool calls in a successful result
    mint execution grants on `bridge` for its bridged tools.
    """

    async def generate(**params: JsonValue) -> JsonValue:
        if handler is None:
            return _provider_error(
                {
                    "status": 404,
                    "message": "This model proxy does not serve "
                    f"{_GENERATE_METHODS[method]}.",
                }
            )
        try:
            result = await handler(**params)
        except LimitExceededError:
            raise
        except ModelProxyError as ex:
            payload: dict[str, JsonValue] = {
                "status": ex.status_code,
                "message": ex.message,
            }
            if ex.body is not None:
                payload["body"] = ex.body
            return _provider_error(payload)
        except Exception as ex:
            recovered = provider_error_payload(ex)
            if recovered["status"] is None:
                logger.warning(
                    "Model proxy handler '%s' raised an exception with no HTTP "
                    "status (forwarding to the client as an error response): %s",
                    method,
                    ex,
                    exc_info=True,
                )
            return _provider_error(cast(dict[str, JsonValue], recovered))

        if (
            bridge.bridged_tools
            and isinstance(result, dict)
            and PROVIDER_ERROR_KEY not in result
        ):
            calls = _proposed_calls(method, result)
            if calls:
                bridge.register_tool_execution_grants(
                    calls, _declared_tools(method, params.get("json_data"))
                )
        return result

    return generate


def _provider_error(payload: dict[str, JsonValue]) -> JsonValue:
    return {PROVIDER_ERROR_KEY: payload}


def _declared_tools(method: str, request: JsonValue) -> list[ToolInfo]:
    """The function tools `request` declares to the model, as grant resolution reads them.

    Only the name and description are kept: a call is matched to a declaration
    by name, and the declaration to a bridged tool by description
    (`SandboxAgentBridge.register_tool_execution_grants`). Tools the provider
    implements (Anthropic tools with a `type`, OpenAI built-in tools) are not
    function declarations and are skipped.
    """
    if not isinstance(request, dict):
        return []
    declared: list[tuple[JsonValue, JsonValue]] = []
    for tool in _dicts(request.get("tools")):
        match method:
            case "generate_anthropic":
                if tool.get("type", "custom") == "custom":
                    declared.append((tool.get("name"), tool.get("description")))
            case "generate_completions":
                function = tool.get("function")
                if tool.get("type") == "function" and isinstance(function, dict):
                    declared.append((function.get("name"), function.get("description")))
            case "generate_responses":
                if tool.get("type") == "function":
                    declared.append((tool.get("name"), tool.get("description")))
            case "generate_google":
                for key in ("functionDeclarations", "function_declarations"):
                    declared.extend(
                        (declaration.get("name"), declaration.get("description"))
                        for declaration in _dicts(tool.get(key))
                    )
    if method == "generate_completions":
        declared.extend(
            (function.get("name"), function.get("description"))
            for function in _dicts(request.get("functions"))
        )
    return [
        ToolInfo(
            name=name, description=description if isinstance(description, str) else ""
        )
        for name, description in declared
        if isinstance(name, str)
    ]


def _proposed_calls(method: str, response: dict[str, JsonValue]) -> list[ToolCall]:
    """The function calls the model proposed in a raw `method` response.

    A call whose id, name or arguments cannot be read (arguments that are not a
    JSON object) is skipped, so it grants nothing.
    """
    calls: list[ToolCall] = []

    def add(id: JsonValue, function: JsonValue, arguments: JsonValue) -> None:
        if (
            isinstance(id, str)
            and isinstance(function, str)
            and isinstance(arguments, dict)
        ):
            calls.append(ToolCall(id=id, function=function, arguments=arguments))

    match method:
        case "generate_anthropic":
            for block in _dicts(response.get("content")):
                if block.get("type") == "tool_use":
                    add(block.get("id"), block.get("name"), block.get("input"))
        case "generate_completions":
            for choice in _dicts(response.get("choices")):
                message = choice.get("message")
                if not isinstance(message, dict):
                    continue
                for tool_call in _dicts(message.get("tool_calls")):
                    function = tool_call.get("function")
                    if tool_call.get("type", "function") == "function" and isinstance(
                        function, dict
                    ):
                        add(
                            tool_call.get("id"),
                            function.get("name"),
                            _json_object(function.get("arguments")),
                        )
                function_call = message.get("function_call")
                if isinstance(function_call, dict):
                    add(
                        "function_call",
                        function_call.get("name"),
                        _json_object(function_call.get("arguments")),
                    )
        case "generate_responses":
            for item in _dicts(response.get("output")):
                if item.get("type") == "function_call":
                    add(
                        item.get("call_id"),
                        item.get("name"),
                        _json_object(item.get("arguments")),
                    )
        case "generate_google":
            for candidate in _dicts(response.get("candidates")):
                content = candidate.get("content")
                if not isinstance(content, dict):
                    continue
                for part in _dicts(content.get("parts")):
                    function_call = part.get("functionCall", part.get("function_call"))
                    if isinstance(function_call, dict):
                        name = function_call.get("name")
                        add(
                            function_call.get("id", name),
                            name,
                            function_call.get("args"),
                        )
    return calls


def _dicts(value: JsonValue) -> list[dict[str, JsonValue]]:
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _json_object(value: JsonValue) -> JsonValue:
    """`value` parsed as a JSON object if it is one (tool call arguments), else `None`."""
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return cast(JsonValue, parsed) if isinstance(parsed, dict) else None
