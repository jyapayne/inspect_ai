"""Tests for model_aliases in resolve_inspect_model."""

from typing import Any, cast

import pytest

from inspect_ai.agent._bridge.util import resolve_inspect_model
from inspect_ai.model._model import Model, get_model


def test_resolve_inspect_model_bare_inspect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECT_EVAL_MODEL", "mockllm/default")
    model = resolve_inspect_model("inspect")
    assert str(model) == "mockllm/default"


def test_resolve_inspect_model_prefixed() -> None:
    model = resolve_inspect_model("inspect/mockllm/model")
    assert str(model) == "mockllm/model"


def test_resolve_inspect_model_alias_takes_priority() -> None:
    target = get_model("mockllm/alias-target")
    aliases: dict[str, str | Model] = {"my-alias": target}
    result = resolve_inspect_model("my-alias", model_aliases=aliases)
    assert result is target


def test_resolve_inspect_model_alias_string() -> None:
    aliases: dict[str, str | Model] = {"my-alias": "mockllm/alias-target"}
    result = resolve_inspect_model("my-alias", model_aliases=aliases)
    assert str(result) == "mockllm/alias-target"


def test_resolve_inspect_model_fallback_used_for_non_inspect() -> None:
    result = resolve_inspect_model(
        "some-random-model", fallback_model="inspect/mockllm/fallback"
    )
    assert str(result) == "mockllm/fallback"


def test_resolve_inspect_model_alias_over_fallback() -> None:
    """Aliases are checked before fallback."""
    target = get_model("mockllm/alias-target")
    aliases: dict[str, str | Model] = {"my-name": target}
    result = resolve_inspect_model(
        "my-name", model_aliases=aliases, fallback_model="inspect/mockllm/other"
    )
    assert result is target


def test_resolve_inspect_model_resolver_routes_request() -> None:
    """A model_resolver routes the requested name (checked after aliases)."""
    target = get_model("mockllm/resolver-target")
    result = resolve_inspect_model(
        "anything-at-all", model_resolver=lambda name: target
    )
    assert result is target


def test_resolve_inspect_model_resolver_string_spec() -> None:
    result = resolve_inspect_model(
        "foo", model_resolver=lambda name: "mockllm/spec-target"
    )
    assert str(result) == "mockllm/spec-target"


def test_resolve_inspect_model_resolver_none_defers_to_fallback() -> None:
    """Returning None from the resolver defers to the static fallback."""
    result = resolve_inspect_model(
        "gpt-4o",
        fallback_model="inspect/mockllm/fallback",
        model_resolver=lambda name: None,
    )
    assert str(result) == "mockllm/fallback"


def test_resolve_inspect_model_alias_over_resolver() -> None:
    """Aliases are checked before the resolver."""
    target = get_model("mockllm/alias-target")

    def resolver(name: str) -> Model:
        raise AssertionError("resolver must not run when an alias matches")

    result = resolve_inspect_model(
        "my-name", model_aliases={"my-name": target}, model_resolver=resolver
    )
    assert result is target


def test_sandbox_agent_bridge_threads_model_resolver() -> None:
    """model_resolver must reach the constructed bridge.

    (sandbox_agent_bridge -> SandboxAgentBridge -> AgentBridge).
    """
    from inspect_ai.agent._agent import AgentState
    from inspect_ai.agent._bridge.sandbox.types import SandboxAgentBridge

    def resolver(name: str) -> Model:
        return get_model("mockllm/model")

    bridge = SandboxAgentBridge(
        AgentState(messages=[]),
        None,  # filter
        None,  # retry_refusals
        None,  # compaction
        13131,  # port
        None,  # model (fallback)
        model_resolver=resolver,
    )
    assert bridge.model_resolver is resolver


class _CollectingSink:
    """ModelEventSink that keeps every event handed to it."""

    def __init__(self) -> None:
        self.pending: list[object] = []
        self.complete: list[object] = []

    def on_pending(self, event: object) -> None:
        self.pending.append(event)

    def on_complete(self, event: object) -> None:
        self.complete.append(event)


async def test_bridged_event_records_the_client_requested_model() -> None:
    """An alias-resolved request keeps the name the client actually asked for.

    Codex hardcodes the `codex-auto-review` guardian slug and the bridge maps it
    onto the eval model, so without this the reviewer's turns are recorded as
    the agent's own and reviewer-vs-agent attribution is impossible.
    """
    from inspect_ai.agent._agent import AgentState
    from inspect_ai.agent._bridge.types import AgentBridge
    from inspect_ai.agent._bridge.util import bridge_generate
    from inspect_ai.model._generate_config import GenerateConfig
    from inspect_ai.model._model import BRIDGE_REQUESTED_MODEL

    sink = _CollectingSink()
    bridge = AgentBridge(AgentState(messages=[]), model_event_sink=sink)
    model = get_model("mockllm/model")

    await bridge_generate(
        bridge,
        model,
        [],
        [],
        None,
        GenerateConfig(),
        requested_model="codex-auto-review",
    )

    assert sink.complete, "bridged generate emitted no ModelEvent"
    event = sink.complete[-1]
    metadata = getattr(event, "metadata", None) or {}
    assert metadata.get(BRIDGE_REQUESTED_MODEL) == "codex-auto-review"


@pytest.mark.parametrize(
    ("client_metadata", "expected"),
    [
        (
            {
                "thread_id": "root-thread",
                "installation_id": "do-not-record",
                "arbitrary": "do-not-record",
                "x-codex-parent-thread-id": "do-not-record",
                "x-openai-subagent": "other",
            },
            {"thread_id": "root-thread"},
        ),
        (
            {
                "thread_id": "child-thread",
                "x-codex-parent-thread-id": "root-thread",
                "x-openai-subagent": "collab_spawn",
                "installation_id": "do-not-record",
            },
            {
                "thread_id": "child-thread",
                "parent_thread_id": "root-thread",
                "subagent": "collab_spawn",
            },
        ),
    ],
    ids=["root", "child"],
)
async def test_responses_bridge_records_allowlisted_codex_client_metadata(
    client_metadata: dict[str, str], expected: dict[str, str]
) -> None:
    """Responses client metadata records only Codex thread provenance."""
    from inspect_ai.agent._agent import AgentState
    from inspect_ai.agent._bridge.responses import inspect_responses_api_request
    from inspect_ai.agent._bridge.types import AgentBridge
    from inspect_ai.agent._bridge.util import (
        default_code_execution_providers,
        internal_web_search_providers,
    )

    sink = _CollectingSink()
    bridge = AgentBridge(
        AgentState(messages=[]),
        model_aliases={"codex": "mockllm/model"},
        model_event_sink=sink,
    )

    response = await inspect_responses_api_request(
        {
            "model": "codex",
            "input": "record this provenance",
            "client_metadata": client_metadata,
        },
        None,
        internal_web_search_providers(),
        default_code_execution_providers(),
        bridge,
    )

    assert response.output_text
    assert sink.complete, "bridged generate emitted no ModelEvent"
    metadata = getattr(sink.complete[-1], "metadata", None) or {}
    assert metadata["agent_bridge"]["codex"] == expected


async def test_filter_answered_request_still_records_an_event() -> None:
    """Output a filter produced without generating stays traceable.

    The scaffold consumes it, so a grader reading the transcript would otherwise
    see an assistant message that no ModelEvent accounts for.
    """
    from inspect_ai.agent._agent import AgentState
    from inspect_ai.agent._bridge.types import AgentBridge
    from inspect_ai.agent._bridge.util import bridge_generate
    from inspect_ai.model._generate_config import GenerateConfig
    from inspect_ai.model._model import (
        BRIDGE_FILTER_SYNTHETIC,
        BRIDGE_REQUESTED_MODEL,
    )
    from inspect_ai.model._model_output import ModelOutput

    filtered = ModelOutput.from_content("mockllm/model", "answered by the filter")

    async def _filter(
        model_name: str,
        input: object,
        tools: object,
        tool_choice: object,
        config: object,
    ) -> ModelOutput:
        return filtered

    sink = _CollectingSink()
    bridge = AgentBridge(AgentState(messages=[]), model_event_sink=sink, filter=_filter)
    model = get_model("mockllm/model")

    output, _ = await bridge_generate(
        bridge,
        model,
        [],
        [],
        None,
        GenerateConfig(),
        requested_model="codex-auto-review",
    )

    assert output.completion == "answered by the filter"
    assert sink.complete, "filter-answered request emitted no ModelEvent"
    # Exactly one pending/complete pair: _record_model_interaction self-completes
    # when given output, so an extra explicit complete() would double-forward the
    # event through the sink (a duplicated assistant message downstream).
    assert len(sink.pending) == 1
    assert len(sink.complete) == 1
    event = sink.complete[-1]
    metadata = getattr(event, "metadata", None) or {}
    assert metadata.get(BRIDGE_FILTER_SYNTHETIC) is True
    assert metadata.get(BRIDGE_REQUESTED_MODEL) == "codex-auto-review"
    # The event must carry the real output so a consumer can trace the
    # resulting assistant message back to it by content.
    assert getattr(event, "output").completion == "answered by the filter"
    assert getattr(event, "call") is None


async def test_bridged_events_keep_allowlisted_headers_with_concurrent_requests() -> (
    None
):
    """Each request's selected headers reach only its own bridged event."""
    from inspect_ai._util._async import tg_collect
    from inspect_ai.agent._agent import AgentState
    from inspect_ai.agent._bridge.types import AgentBridge
    from inspect_ai.agent._bridge.util import bridge_generate
    from inspect_ai.model._chat_message import ChatMessage, ChatMessageUser
    from inspect_ai.model._generate_config import GenerateConfig
    from inspect_ai.model._model import (
        BRIDGE_FILTER_SYNTHETIC,
        BRIDGE_REQUEST_HEADERS,
    )
    from inspect_ai.model._model_output import ModelOutput

    async def _filter(
        model: Model,
        input: list[ChatMessage],
        tools: object,
        tool_choice: object,
        config: GenerateConfig,
    ) -> ModelOutput | None:
        del model, tools, tool_choice, config
        if input and getattr(input[-1], "text", None) == "filter request":
            return ModelOutput.from_content("mockllm/model", "filter output")
        return None

    sink = _CollectingSink()
    bridge = AgentBridge(
        AgentState(messages=[]),
        model_event_sink=sink,
        filter=_filter,
        model_event_metadata_headers=(
            "X-OpenCode-Session",
            "X-Session-Id",
            "X-Parent-Session-Id",
        ),
    )
    model = get_model("mockllm/model")

    outputs = await tg_collect(
        [
            lambda: bridge_generate(
                bridge,
                model,
                [],
                [],
                None,
                GenerateConfig(extra_headers={"x-provider-feature": "provider"}),
                metadata_headers={
                    "x-opencode-session": "provider-session",
                    "x-parent-session-id": "provider-parent",
                    "authorization": "Bearer must-not-record",
                    "x-unselected": "must-not-record",
                },
                requested_model="same-model",
            ),
            lambda: bridge_generate(
                bridge,
                model,
                [ChatMessageUser(content="filter request")],
                [],
                None,
                GenerateConfig(extra_headers={"x-provider-feature": "filter"}),
                metadata_headers={
                    "x-session-id": "filter-session",
                    "x-parent-session-id": "filter-parent",
                    "cookie": "must-not-record",
                    "x-unselected": "must-not-record",
                },
                requested_model="same-model",
            ),
        ]
    )

    assert outputs[1][0].completion == "filter output"

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
    assert headers_by_session["filter-session"][BRIDGE_REQUEST_HEADERS] == {
        "x-session-id": "filter-session",
        "x-parent-session-id": "filter-parent",
    }
    assert headers_by_session["filter-session"][BRIDGE_FILTER_SYNTHETIC] is True


async def test_metadata_headers_do_not_reach_model_provider() -> None:
    """Native session attribution stays out of the provider request."""
    from inspect_ai.agent._agent import AgentState
    from inspect_ai.agent._bridge.types import AgentBridge
    from inspect_ai.agent._bridge.util import bridge_generate
    from inspect_ai.model._generate_config import GenerateConfig

    provider_requests: list[dict[str, Any]] = []
    model = get_model(
        "openai/gpt-5",
        api_key="local-fake-key",
        base_url="http://127.0.0.1:9/v1",
        memoize=False,
        responses_api=False,
    )

    async def capture_create(**kwargs: Any) -> None:
        provider_requests.append(kwargs)
        raise RuntimeError("stop after capturing provider request")

    cast(Any, model.api).client.chat.completions.create = capture_create
    bridge = AgentBridge(
        AgentState(messages=[]),
        model_event_metadata_headers=("x-opencode-session",),
    )

    try:
        with pytest.raises(RuntimeError, match="stop after capturing"):
            await bridge_generate(
                bridge,
                model,
                [],
                [],
                None,
                GenerateConfig(extra_headers={"x-provider-feature": "preserve"}),
                metadata_headers={"x-opencode-session": "native-session"},
            )
    finally:
        await model.api.aclose()

    assert provider_requests
    provider_headers = provider_requests[0].get("extra_headers") or {}
    assert provider_headers["x-provider-feature"] == "preserve"
    assert "x-opencode-session" not in provider_headers


@pytest.mark.parametrize(
    ("factory_name", "adapter_name"),
    [
        ("generate_completions", "inspect_completions_api_request"),
        ("generate_responses", "inspect_responses_api_request"),
        ("generate_anthropic", "inspect_anthropic_api_request"),
        ("generate_google", "inspect_google_api_request"),
    ],
)
async def test_sandbox_service_passes_metadata_separately_from_provider_headers(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    adapter_name: str,
) -> None:
    """Each service dialect retains separate provider and metadata headers."""
    from inspect_ai.agent._bridge.sandbox import service

    captured: dict[str, Any] = {}

    class _Completion:
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            return {}

    async def fake_adapter(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {} if adapter_name == "inspect_google_api_request" else _Completion()

    monkeypatch.setattr(service, adapter_name, fake_adapter)
    bridge = cast(Any, object())
    factory = getattr(service, factory_name)
    generate = (
        factory(bridge)
        if factory_name == "generate_completions"
        else factory(None, None, bridge)
    )

    result = await generate(
        {"model": "inspect"},
        headers={"anthropic-beta": "preserve", "x-provider-feature": "dropped"},
        metadata_headers={"x-opencode-session": "native-session"},
    )

    assert result == {}
    # the service layer forwards only allowlisted client headers to the provider
    assert captured["args"][1] == {"anthropic-beta": "preserve"}
    assert captured["kwargs"] == {
        "metadata_headers": {"x-opencode-session": "native-session"}
    }


@pytest.mark.parametrize("header", ("authorization", "x-password"))
def test_bridge_rejects_sensitive_event_metadata_headers(header: str) -> None:
    """A sink opt-in cannot use authentication material as event metadata."""
    from inspect_ai.agent._agent import AgentState
    from inspect_ai.agent._bridge.types import AgentBridge

    with pytest.raises(ValueError, match="sensitive"):
        AgentBridge(
            AgentState(messages=[]),
            model_event_metadata_headers=(header,),
        )


def test_bridge_rejects_delimited_event_metadata_header_names() -> None:
    """Header names must survive the proxy's comma-delimited configuration."""
    from inspect_ai.agent._agent import AgentState
    from inspect_ai.agent._bridge.types import AgentBridge

    with pytest.raises(ValueError, match="valid HTTP token"):
        AgentBridge(
            AgentState(messages=[]),
            model_event_metadata_headers=("x-opencode-session,x-parent-session-id",),
        )
