import contextlib
from collections.abc import Sequence
from logging import getLogger
from typing import TYPE_CHECKING, AsyncIterator

import anyio
from shortuuid import uuid

from inspect_ai._util.exception import TerminateSampleError
from inspect_ai.model._compaction.types import CompactionStrategy
from inspect_ai.model._model import (
    GenerateFilter,
    Model,
    ModelEventSink,
    ModelResolver,
    ModelResponseFilter,
)
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.tool._mcp._tools_bridge import BridgedToolsSpec
from inspect_ai.tool._sandbox_tools_utils.sandbox import sandbox_with_injected_tools
from inspect_ai.tool._tool_def import ToolDef
from inspect_ai.tool._tools._code_execution import CodeExecutionProviders
from inspect_ai.tool._tools._web_search._web_search import (
    WebSearchProviders,
)
from inspect_ai.util._anyio import inner_exception
from inspect_ai.util._checkpoint.checkpointer import Checkpointer
from inspect_ai.util._sandbox._cli import SANDBOX_CLI
from inspect_ai.util._sandbox.environment import SandboxEnvironment
from inspect_ai.util._sandbox.exec_remote import (
    ExecCompleted,
    ExecRemoteProcess,
    ExecRemoteStreamingOptions,
    ExecStderr,
)
from inspect_ai.util._sandbox.service import SandboxServiceMethod

from ..._agent import AgentState
from ..types import StateFilter
from ..util import resolve_bridge_code_execution, resolve_bridge_web_search
from .service import MODEL_SERVICE, model_service_methods, run_model_service
from .types import SandboxAgentBridge

if TYPE_CHECKING:
    # deferred: importing `inspect_ai.approval` at module scope here cycles
    # through approval -> event -> scorer while `inspect_ai.agent` is still
    # initializing. Same reason `model/_call_tools.py` defers it.
    from inspect_ai.approval._policy import ApprovalPolicy

logger = getLogger(__name__)


@contextlib.asynccontextmanager
async def sandbox_agent_bridge(
    state: AgentState | None = None,
    *,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    model_resolver: ModelResolver | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    compaction: CompactionStrategy | None = None,
    sandbox: str | None = None,
    port: int = 13131,
    poll_timeout_recovery: float | None = None,
    web_search: WebSearchProviders | bool | None = None,
    code_execution: CodeExecutionProviders | bool | None = None,
    client_mcp_servers: bool | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    model_event_sink: ModelEventSink | None = None,
    model_event_metadata_headers: Sequence[str] | None = None,
    forward_generation_config: bool = False,
    approval: list["ApprovalPolicy"] | None = None,
    checkpointer: Checkpointer | None = None,
    response_filter: ModelResponseFilter | None = None,
    state_filter: StateFilter | None = None,
    accumulate_conversations: bool = False,
) -> AsyncIterator[SandboxAgentBridge]:
    """Sandbox agent bridge.

    Provide Inspect integration for agents running inside sandboxes. Runs
    a proxy server in the container that provides REST endpoints for the OpenAI Completions API, OpenAI Responses API, Anthropic API, and Google API. This proxy server
    runs on port 13131 and routes requests to the current Inspect model provider.

    You should set `OPENAI_BASE_URL=http://localhost:13131/v1`, `ANTHROPIC_BASE_URL=http://localhost:13131`, or `GOOGLE_GEMINI_BASE_URL=http://localhost:13131` when executing
    the agent within the container and ensure that your agent targets the
    model name "inspect" when calling OpenAI, Anthropic, or Google. Use "inspect/<full-model-name>" to target other Inspect model providers.

    Args:
        state: Initial state for agent bridge. Used as a basis for yielding
            an updated state based on traffic over the bridge.
        model: Fallback model for requests that don't use "inspect" or an "inspect/"
            prefixed model (defaults to "inspect", can also specify e.g.
            "inspect/openai/gpt-4o" to force another specific model).
        model_aliases: Map of model name aliases. When a request uses a name
            that appears here, the corresponding value (a ``Model`` instance
            or model spec string) is used instead. Checked before the fallback ``model``.
        model_resolver: Dynamic routing policy called with the requested model
            name (provider-qualified on a provider-specific endpoint, e.g.
            ``openai/gpt-5.1``). Checked after ``model_aliases`` and before the ``model``
            fallback; return a ``Model``/spec to route the request there, or
            ``None`` to defer. Routes by policy without enumerating every name.
        filter: Filter for bridge model generation.
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        compaction: Compact the conversation when it it is close to overflowing
            the model's context window. See [Compaction](https://inspect.aisi.org.uk/compaction.html) for details on compaction strategies.
        sandbox: Sandbox to run model proxy server within.
        port: Port to run proxy server on.
        poll_timeout_recovery: Seconds to keep re-polling the proxy server's
            process after a poll of it times out. Defaults to `None`, where a
            proxy poll that times out fails the sample. Each re-issued poll can
            wait the proxy's full 600-second poll timeout, so recovery can run
            past this value by about that much (see
            `ExecRemoteCommonOptions.poll_timeout_recovery`).
        web_search: Configuration for mapping model internal web_search tools to
            Inspect. Withheld by default: a sandboxed agent that names the native
            tool in a request would otherwise reach the web through the model
            provider, bypassing the sandbox's own network policy. Pass `True` to
            map to the internal provider of the target model (supported for
            OpenAI, Anthropic, Gemini, Grok, and Perplexity), or a configuration
            to use an external provider like Tavily or Exa for models that don't
            support internal search.
        code_execution: Configuration for mapping model internal code_execution
            tools to Inspect. Withheld by default (see `web_search`). Pass `True`
            to map to the internal provider of the target model (supported for
            OpenAI, Anthropic, Google, and Grok); if the provider does not support
            native code execution then the bash() tool will be provided
            (note that this requires a sandbox by declared for the task).
        client_mcp_servers: Honor MCP servers declared by the sandboxed agent
            (defaults to `False`). When enabled, the agent may name any server URL
            and the model provider will connect to it. Prefer `bridged_tools` for
            exposing tools you choose.
        bridged_tools: Host-side Inspect tools to expose to the sandboxed agent
            via MCP protocol. Each BridgedToolsSpec creates an MCP server that
            makes the specified tools available to the agent. A bridged tool
            executes only for a call the model proposed in a bridged generation,
            once per proposal, unless its spec sets `require_proposal=False`
            (see `BridgedToolsSpec`). The resolved MCPServerConfigStdio objects
            to pass to CLI agents are available via bridge.mcp_server_configs.
        model_event_sink: Optional sink that takes ownership of `ModelEvent`
            emission for calls routed through the bridge. When set, the bridge
            installs it around `model.generate()` so the sink decides when and
            under which span each event is emitted to the transcript.
        model_event_metadata_headers: Optional non-sensitive client request
            header names to copy into each matching bridged `ModelEvent` under
            `BRIDGE_REQUEST_HEADERS`. Names are normalized to lower case; only
            these names cross the sandbox RPC boundary as event metadata, never
            provider headers, and sensitive names such as authorization and cookies
            are rejected. Defaults to `None`.
        forward_generation_config: Forward client generation parameters (e.g.
            `max_tokens`, `temperature`, reasoning effort) to the model. Defaults
            to `False`, in which case those parameters are dropped and the resolved
            Inspect model config and provider defaults govern generation (structural
            parameters like the system prompt, tools, and response format are always
            forwarded). Set `True` for faithful-proxy behavior where the client's
            generation parameters are authoritative.
        approval: Approval policies for tool calls made by the bridged agent.
            Temporarily replaces any active approval policies for the duration of
            each approval. Eval-level and task-level policies already apply without
            this, but an `approval()` block entered inside the agent body does not
            reach the sandbox service task — pass policies here instead. A rejected
            tool call is never handed to the agent: the model is told it was
            rejected and generation is retried.
        checkpointer: Checkpointer to drive through the bridge. When provided,
            the bridge ticks it after each generation and registers its agent
            state (messages, output, compaction prefix) for checkpoint backup
            and restore, so a checkpointed run survives resume. Defaults to
            `None` (no checkpointing).
        response_filter: Filter that mutates model output after generation.
            Called inside the refusal-retry loop, after ``model.generate()``
            and after the compaction baseline update. Return ``None`` to pass
            through; return a ``ModelOutput`` to replace the response.
        state_filter: Optional predicate that selects client requests whose
            generations update the yielded state's messages and output.
            Rejected requests still receive normal model responses and model
            events, and still tick the checkpointer.
        accumulate_conversations: Keep every conversation observed over the bridge
            rather than tracking a single main one. Defaults to `False`, which surfaces
            the main agent loop and treats other traffic as side calls. Set `True` when
            the sandbox may run several independent conversations: `state.messages`
            then holds each in the order they started.
    """
    # resolve sandbox
    sandbox_env = await sandbox_with_injected_tools(sandbox_name=sandbox)

    # resolve granted capabilities. These default to withheld: the sandboxed agent
    # can otherwise obtain them just by naming the native tool in a request, which
    # reaches the web through the model provider even when the sandbox has no
    # network egress of its own.
    web_search_grant = resolve_bridge_web_search(web_search, default_grant=False)
    code_execution_grant = resolve_bridge_code_execution(
        code_execution, default_grant=False
    )
    allow_remote_mcp = False if client_mcp_servers is None else client_mcp_servers

    # create a state value that will be used to track mesages going over the bridge
    state = state or AgentState(messages=[])

    # create the bridge and register its bridged tools
    bridge = SandboxAgentBridge(
        state=state,
        filter=filter,
        retry_refusals=retry_refusals,
        compaction=compaction,
        port=port,
        model=model,
        model_aliases=model_aliases,
        model_resolver=model_resolver,
        model_event_sink=model_event_sink,
        model_event_metadata_headers=model_event_metadata_headers,
        forward_generation_config=forward_generation_config,
        approval=approval,
        checkpointer=checkpointer,
        state_filter=state_filter,
        allow_remote_mcp=allow_remote_mcp,
        response_filter=response_filter,
        accumulate_conversations=accumulate_conversations,
    )
    _register_bridged_tool_specs(bridge, bridged_tools or [], port)

    async with _model_proxy_service(
        sandbox_env,
        model_service_methods(web_search_grant, code_execution_grant, bridge),
        bridge,
        port=port,
        polling_interval=2,
        caller="sandbox_agent_bridge",
        poll_timeout_recovery=poll_timeout_recovery,
    ):
        yield bridge


@contextlib.asynccontextmanager
async def _model_proxy_service(
    sandbox_env: SandboxEnvironment,
    methods: dict[str, SandboxServiceMethod],
    bridge: SandboxAgentBridge | None,
    *,
    port: int,
    polling_interval: float | None,
    caller: str,
    poll_timeout_recovery: float | None = None,
    raw_http: bool = False,
) -> AsyncIterator[None]:
    """Run the in-sandbox model proxy against `methods` for the duration of the block.

    The one implementation behind `sandbox_agent_bridge` and `sandbox_model_proxy`.
    Starts the model service (`run_model_service`) serving `methods`, then the
    `model_proxy` process of the injected sandbox tools, which listens on
    `localhost:<port>` inside `sandbox_env` and files each HTTP request into the
    service. The block fails if the proxy process dies or a handler asks to fail it
    through `bridge.request_fail`. `poll_timeout_recovery` is passed to the proxy
    process's `exec_remote` options. On exit the proxy is killed and the service
    stopped. An error raised by that cleanup after the block completed normally is
    logged (naming `caller`) rather than raised.
    """
    # instance id for this proxy
    instance = f"proxy_{uuid()}"

    # Track whether the block completed successfully. If so, cleanup errors
    # should be logged but not cause the sample to fail.
    block_completed = False
    closing = anyio.Event()

    try:
        async with anyio.create_task_group() as tg:
            # event to signal startup of model service
            started = anyio.Event()

            # sandbox service that receives model requests (and tool calls)
            tg.start_soon(
                run_model_service,
                sandbox_env,
                methods,
                instance,
                polling_interval,
                started,
            )

            # wait for model service to start
            await started.wait()

            # proxy server that runs in container and forwards to sandbox service
            proxy = await sandbox_env.exec_remote(
                cmd=[SANDBOX_CLI, "model_proxy"],
                options=ExecRemoteStreamingOptions(
                    concurrency=False,
                    user=sandbox_env._tools_user,
                    env={
                        f"{MODEL_SERVICE.upper()}_PORT": str(port),
                        f"{MODEL_SERVICE.upper()}_INSTANCE": instance,
                        **(
                            {"BRIDGE_MODEL_SERVICE_RAW_HTTP_VERSION": "1"}
                            if raw_http
                            else {}
                        ),
                        **(
                            {
                                "BRIDGE_MODEL_EVENT_METADATA_HEADERS": ",".join(
                                    sorted(bridge.model_event_metadata_headers)
                                )
                            }
                            if bridge is not None and bridge.model_event_metadata_headers
                            else {}
                        ),
                    },
                    poll_timeout=600,
                    poll_timeout_recovery=poll_timeout_recovery,
                ),
            )

            # monitor proxy for unexpected death
            tg.start_soon(_monitor_proxy, proxy, raw_http, closing)

            # monitor for a sample failure requested from the service task
            # (approver termination, fail_on_refusal, a host tool that raised)
            if bridge is not None:
                tg.start_soon(_monitor_failure, bridge)

            # the caller's block
            try:
                yield
                block_completed = True
            finally:
                closing.set()
                if bridge is not None:
                    bridge.close_conversation_spans()
                with anyio.CancelScope(shield=True):
                    # ensure the process terminates (no-op if already dead)
                    await proxy.kill()

                # ensure the scope is cancelled
                tg.cancel_scope.cancel()
    except Exception as ex:
        # If the block completed successfully but we got an error during cleanup,
        # log the error but don't fail the sample.
        if block_completed:
            logger.warning(
                f"Error during {caller} cleanup (agent completed successfully): {inner_exception(ex)}"
            )
        else:
            # Error occurred before or during the block
            raise inner_exception(ex)


def _register_bridged_tool_specs(
    bridge: SandboxAgentBridge, specs: Sequence[BridgedToolsSpec], port: int
) -> None:
    """Register each bridged tools spec with the bridge and record its MCP config."""
    seen_names: set[str] = set()
    for spec in specs:
        if spec.name in seen_names:
            raise ValueError(
                f"Duplicate bridged_tools name: '{spec.name}'. "
                "Each BridgedToolsSpec must have a unique name."
            )
        seen_names.add(spec.name)
        config = _register_bridged_tools(bridge, spec, port)
        bridge.mcp_server_configs.append(config)
    bridge.warn_indistinct_tools()


def _register_bridged_tools(
    bridge: SandboxAgentBridge, spec: BridgedToolsSpec, port: int
) -> MCPServerConfigHTTP:
    """Register bridged tools with the bridge and return MCP config.

    Tools are registered in bridge.bridged_tools for execution by the service.
    Returns an MCPServerConfigHTTP with URL pointing to the MCP HTTP endpoint.
    """
    bridge.register_bridged_tools(
        spec.name,
        {ToolDef(tool).name: tool for tool in spec.tools},
        require_proposal=spec.require_proposal,
    )

    # Return MCP config with HTTP URL
    return MCPServerConfigHTTP(
        name=spec.name,
        type="http",
        url=f"http://localhost:{port}/mcp/{spec.name}",
        tools="all",
    )


async def _monitor_failure(bridge: SandboxAgentBridge) -> None:
    """Raise the error a bridged generation or tool call asked the sample to fail with.

    Bridged generations and host tool calls run in the sandbox service task,
    whose exceptions never propagate (see `SandboxAgentBridge.request_fail`).
    Raising here instead puts the error in the bridge's own task group, so it
    unwinds the agent and reaches the sample runner.
    """
    await bridge._failure_requested.wait()
    raise bridge._failure or TerminateSampleError(
        "Sample terminated by tool call approver."
    )


async def _monitor_proxy(
    proxy: ExecRemoteProcess,
    require_running: bool = False,
    closing: anyio.Event | None = None,
) -> None:
    """Monitor the proxy process event stream and raise if it dies unexpectedly."""
    stderr: list[str] = []
    async for event in proxy:
        if isinstance(event, ExecStderr):
            stderr.append(event.data)
            logger.debug("model_proxy stderr: %s", event.data.rstrip())
        if isinstance(event, ExecCompleted):
            if require_running and closing is not None and closing.is_set():
                return
            if not event.success:
                raise RuntimeError(
                    f"Model proxy process exited unexpectedly with failure: {''.join(stderr)}."
                )
            if stderr:
                logger.warning(
                    "model_proxy stderr output on clean exit:\n%s",
                    "".join(stderr).rstrip(),
                )
            if require_running:
                raise RuntimeError("Raw HTTP model proxy process exited unexpectedly.")
    if require_running and (closing is None or not closing.is_set()):
        raise RuntimeError("Raw HTTP model proxy event stream ended unexpectedly.")
