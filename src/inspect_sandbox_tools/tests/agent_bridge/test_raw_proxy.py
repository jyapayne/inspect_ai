"""Direct regressions for the optional RAW model-service transport."""

import asyncio
import base64
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

import pytest
from inspect_sandbox_tools._agent_bridge import proxy, raw


def chunk(body: bytes, eof: bool = False) -> dict[str, Any]:
    return {
        "version": 1,
        "body_b64": base64.b64encode(body).decode("ascii"),
        "eof": eof,
    }


class Bridge:
    def __init__(self, chunks: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.status = 200
        self.headers = [["Content-Type", "application/json"]]
        self.chunks = iter(chunks or [chunk(b"{}", True)])
        self.start_gate: asyncio.Event | None = None
        self.read_gate: asyncio.Event | None = None
        self.started = asyncio.Event()
        self.reading = asyncio.Event()
        self.closed = asyncio.Event()

    async def __call__(self, method: str, /, **params: Any) -> dict[str, Any]:
        self.calls.append((method, params))
        assert params["version"] == 1
        if method == "raw_http_start":
            self.started.set()
            if self.start_gate is not None:
                await self.start_gate.wait()
            return {
                "version": 1,
                "response_id": "response-1",
                "status": self.status,
                "headers": self.headers,
            }
        if method == "raw_http_read":
            assert params["response_id"] == "response-1"
            self.reading.set()
            if self.read_gate is not None:
                await self.read_gate.wait()
            return next(self.chunks)
        if method == "raw_http_close":
            assert params["response_id"] == "response-1"
            self.closed.set()
        else:
            assert method == "raw_http_ready"
        return {"version": 1}


@asynccontextmanager
async def running(service: Any) -> AsyncIterator[raw.RawHTTPServer]:
    server = await raw.raw_http_proxy_server(0, service)
    task = asyncio.create_task(server.start())
    try:
        await asyncio.wait_for(server._ready.wait(), 2)
        yield server
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def request(
    server: raw.RawHTTPServer, body: bytes = b"{}", extra_headers: bytes = b""
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await asyncio.open_connection(server.host, server.port)
    writer.write(
        b"POST /v1/chat/completions?vendor=%2B&vendor=x HTTP/1.1\r\n"
        b"Host: localhost\r\nContent-Type: application/json\r\n"
        + extra_headers
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    await writer.drain()
    return reader, writer


async def read_response(
    reader: asyncio.StreamReader,
) -> tuple[bytes, list[bytes]]:
    headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2)
    chunks: list[bytes] = []
    while True:
        size = int(await asyncio.wait_for(reader.readline(), 2), 16)
        if size == 0:
            assert await reader.readexactly(2) == b"\r\n"
            return headers, chunks
        chunks.append(await reader.readexactly(size))
        assert await reader.readexactly(2) == b"\r\n"


@pytest.mark.parametrize(
    "body",
    [
        b'{ "model": "x", "vendor": {"untouched":true}, "stream":true }',
        b'{"model":"x","parallel_tool_calls":true,"stream":false,"new_field":7}',
        b'{"model":"x","parallel_tool_calls":false,"stream":true}',
        b"\x00\xffbinary\r\nbody",
    ],
)
async def test_raw_request_bytes_and_header_pairs_are_not_transformed(
    body: bytes,
) -> None:
    bridge = Bridge()
    async with running(bridge) as server:
        reader, writer = await request(
            server, body, b"X-Vendor: first\r\nx-vendor: second\r\nX-Byte: \x85\r\n"
        )
        try:
            _, chunks = await read_response(reader)
            assert chunks == [b"{}"]
            await asyncio.wait_for(bridge.closed.wait(), 2)
        finally:
            writer.close()
            await writer.wait_closed()
    start = next(
        params for method, params in bridge.calls if method == "raw_http_start"
    )
    assert base64.b64decode(start["body_b64"]) == body
    assert start["method"] == "POST"
    assert start["path"] == "/v1/chat/completions?vendor=%2B&vendor=x"
    assert start["headers"] == [
        ["Host", "localhost"],
        ["Content-Type", "application/json"],
        ["X-Vendor", "first"],
        ["x-vendor", "second"],
        ["X-Byte", "\x85"],
        ["Content-Length", str(len(body))],
    ]
    assert [method for method, _ in bridge.calls] == [
        "raw_http_ready",
        "raw_http_start",
        "raw_http_read",
        "raw_http_close",
    ]


async def test_raw_provider_status_error_fields_and_response_pairs_survive() -> None:
    body = b'{"error":{"message":"limited"},"vendor_debug":{"retry":17}}'
    bridge = Bridge([chunk(body, True)])
    bridge.status = 429
    bridge.headers += [
        ["Retry-After", "17"],
        ["X-Vendor", "a"],
        ["X-Vendor", "b"],
        ["Content-Length", str(len(body))],
        ["Connection", "keep-alive, x-hop"],
        ["X-Hop", "remove"],
        ["Transfer-Encoding", "chunked"],
    ]
    async with running(bridge) as server:
        reader, writer = await request(server)
        try:
            headers, chunks = await read_response(reader)
            assert headers.startswith(b"HTTP/1.1 429 Too Many Requests\r\n")
            assert b"Retry-After: 17\r\n" in headers
            assert b"X-Vendor: a\r\nX-Vendor: b\r\n" in headers
            assert b"X-Hop:" not in headers and b"Content-Length:" not in headers
            assert headers.count(b"Transfer-Encoding: chunked") == 1
            assert chunks == [body]
        finally:
            writer.close()
            await writer.wait_closed()


async def test_raw_sse_is_incremental_and_keeps_actual_chunk_bytes() -> None:
    first = b': keepalive\r\nevent: vendor.delta\r\ndata: {"part":"\xe2'
    second = b'\x82\xac"}\r\n\r\ndata: [DONE]\r\n\r\n'
    bridge = Bridge([chunk(first), chunk(second), chunk(b"", True)])
    bridge.headers = [["Content-Type", "text/event-stream"]]
    second_gate = asyncio.Event()
    read_count = 0

    async def service(method: str, /, **params: Any) -> dict[str, Any]:
        nonlocal read_count
        if method == "raw_http_read":
            read_count += 1
            if read_count == 2:
                await second_gate.wait()
        return await bridge(method, **params)

    async with running(service) as server:
        reader, writer = await request(server)
        try:
            headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2)
            assert b"text/event-stream" in headers
            size = int(await asyncio.wait_for(reader.readline(), 2), 16)
            assert await reader.readexactly(size) == first
            assert await reader.readexactly(2) == b"\r\n"
            second_gate.set()
            size = int(await asyncio.wait_for(reader.readline(), 2), 16)
            assert await reader.readexactly(size) == second
            assert await reader.readexactly(2) == b"\r\n"
            assert await reader.read() == b"0\r\n\r\n"
            await asyncio.wait_for(bridge.closed.wait(), 2)
        finally:
            writer.close()
            await writer.wait_closed()


@pytest.mark.parametrize(
    "failure",
    [
        {"version": 1, "error": "upstream_disconnected"},
        {"version": 1, "error": "unknown_response"},
        {"version": 1, "error": "unsupported_or_inactive_protocol"},
        {"version": 2, "body_b64": "", "eof": True},
        {"version": 1, "body_b64": "not base64", "eof": False},
        chunk(b"x" * (raw.RAW_HTTP_CHUNK_BYTES + 1)),
    ],
)
async def test_raw_stream_failures_abort_without_success_terminator(
    failure: dict[str, Any],
) -> None:
    bridge = Bridge([chunk(b"actual bytes"), failure])
    async with running(bridge) as server:
        reader, writer = await request(server)
        try:
            wire = await asyncio.wait_for(reader.read(), 2)
            assert wire.startswith(b"HTTP/1.1 200 OK\r\n")
            assert wire.endswith(b"C\r\nactual bytes\r\n")
            assert b"raw_bridge_error" not in wire
            assert b"0\r\n\r\n" not in wire
            await asyncio.wait_for(bridge.closed.wait(), 2)
        finally:
            writer.close()
            await writer.wait_closed()


async def test_disconnect_interrupts_blocked_read_and_closes_response() -> None:
    bridge = Bridge()
    bridge.read_gate = asyncio.Event()
    async with running(bridge) as server:
        _, writer = await request(server)
        await asyncio.wait_for(bridge.reading.wait(), 2)
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(bridge.closed.wait(), 2)
    assert sum(method == "raw_http_close" for method, _ in bridge.calls) == 1


async def test_disconnect_during_start_closes_the_eventual_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disconnected = asyncio.Event()
    wait_for_disconnect = raw._wait_for_disconnect

    async def observe_disconnect(reader: asyncio.StreamReader) -> None:
        await wait_for_disconnect(reader)
        disconnected.set()

    monkeypatch.setattr(raw, "_wait_for_disconnect", observe_disconnect)
    bridge = Bridge()
    bridge.start_gate = asyncio.Event()
    async with running(bridge) as server:
        _, writer = await request(server)
        await asyncio.wait_for(bridge.started.wait(), 2)
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(disconnected.wait(), 2)
        bridge.start_gate.set()
        await asyncio.wait_for(bridge.closed.wait(), 2)
    assert not bridge.reading.is_set()


async def test_shutdown_cancels_active_read_and_closes_response() -> None:
    bridge = Bridge()
    bridge.read_gate = asyncio.Event()
    async with running(bridge) as server:
        _, writer = await request(server)
        await asyncio.wait_for(bridge.reading.wait(), 2)
        await asyncio.wait_for(server.stop(), 2)
        assert bridge.closed.is_set()
        writer.close()
        await writer.wait_closed()


@pytest.mark.parametrize("phase", ["raw_http_start", "raw_http_read"])
async def test_raw_rpc_timeout_is_local_or_aborts_after_headers(
    phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(raw, "RAW_HTTP_RPC_TIMEOUT_S", 0.02)
    bridge = Bridge()

    async def service(method: str, /, **params: Any) -> dict[str, Any]:
        if method == phase:
            await asyncio.Event().wait()
        return await bridge(method, **params)

    async with running(service) as server:
        reader, writer = await request(server)
        try:
            wire = await asyncio.wait_for(reader.read(), 2)
            if phase == "raw_http_start":
                assert wire.startswith(b"HTTP/1.1 504 Gateway Timeout\r\n")
                assert b"raw_bridge_error" in wire
            else:
                assert wire.startswith(b"HTTP/1.1 200 OK\r\n")
                assert wire.endswith(b"\r\n\r\n")
                assert not wire.endswith(b"0\r\n\r\n")
                await asyncio.wait_for(bridge.closed.wait(), 2)
        finally:
            writer.close()
            await writer.wait_closed()


async def test_invalid_start_response_closes_its_handle_before_failing() -> None:
    bridge = Bridge()

    async def service(method: str, /, **params: Any) -> dict[str, Any]:
        result = await bridge(method, **params)
        if method == "raw_http_start":
            result["version"] = 2
        return result

    async with running(service) as server:
        reader, writer = await request(server)
        try:
            assert (await asyncio.wait_for(reader.read(), 2)).startswith(
                b"HTTP/1.1 502 "
            )
            await asyncio.wait_for(bridge.closed.wait(), 2)
        finally:
            writer.close()
            await writer.wait_closed()


async def test_ready_handshake_observes_bound_listener_before_clients_run() -> None:
    bridge = Bridge()
    connected = asyncio.Event()
    release = asyncio.Event()
    probe_writer: asyncio.StreamWriter | None = None
    server: raw.RawHTTPServer

    async def service(method: str, /, **params: Any) -> dict[str, Any]:
        nonlocal probe_writer
        if method == "raw_http_ready":
            assert server.server is not None
            _, probe_writer = await request(server)
            connected.set()
            await release.wait()
        return await bridge(method, **params)

    server = await raw.raw_http_proxy_server(0, service)
    task = asyncio.create_task(server.start())
    try:
        await asyncio.wait_for(connected.wait(), 2)
        assert not bridge.started.is_set()
        release.set()
        await asyncio.wait_for(bridge.started.wait(), 2)
    finally:
        if probe_writer is not None:
            probe_writer.close()
            await probe_writer.wait_closed()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize(
    "reply",
    [{"version": 2}, {"version": True}, {"version": 1, "error": "unsupported"}, None],
)
async def test_bad_ready_reply_fails_startup_and_closes_listener(reply: Any) -> None:
    callback = AsyncMock(return_value=reply)
    server = await raw.raw_http_proxy_server(0, callback)
    with pytest.raises(raw.RawHTTPProtocolError):
        await server.start()
    assert server.server is None
    assert not server._ready.is_set()
    callback.assert_awaited_once_with("raw_http_ready", version=1)


async def test_ready_timeout_closes_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(raw, "RAW_HTTP_CONTROL_TIMEOUT_S", 0.02)

    async def callback(method: str, /, **params: Any) -> None:
        await asyncio.Event().wait()

    server = await raw.raw_http_proxy_server(0, callback)
    with pytest.raises(asyncio.TimeoutError):
        await server.start()
    assert server.server is None


@pytest.mark.parametrize("version", ["", "0", "2", "01", "true"])
async def test_invalid_opt_in_version_never_calls_either_factory(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(raw.RAW_HTTP_VERSION_ENV, version)
    legacy, byte_proxy = AsyncMock(), AsyncMock()
    monkeypatch.setattr(proxy, "model_proxy_server", legacy)
    monkeypatch.setattr(raw, "raw_http_proxy_server", byte_proxy)
    with pytest.raises(ValueError, match="Unsupported RAW HTTP"):
        await proxy.run_model_proxy_server()
    legacy.assert_not_called()
    byte_proxy.assert_not_called()


@pytest.mark.parametrize("version", [None, "1"])
async def test_cli_selects_only_the_explicitly_requested_factory(
    version: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if version is None:
        monkeypatch.delenv(raw.RAW_HTTP_VERSION_ENV, raising=False)
    else:
        monkeypatch.setenv(raw.RAW_HTTP_VERSION_ENV, version)
    monkeypatch.setenv("BRIDGE_MODEL_SERVICE_PORT", "15321")
    legacy, byte_proxy = AsyncMock(), AsyncMock()
    monkeypatch.setattr(proxy, "model_proxy_server", legacy)
    monkeypatch.setattr(raw, "raw_http_proxy_server", byte_proxy)
    await proxy.run_model_proxy_server()
    selected, unused = (legacy, byte_proxy) if version is None else (byte_proxy, legacy)
    selected.assert_awaited_once_with(15321)
    selected.return_value.start.assert_awaited_once_with()
    unused.assert_not_called()


async def test_body_reader_has_one_chunk_of_backpressure() -> None:
    bridge = Bridge([chunk(b"a" * raw.RAW_HTTP_CHUNK_BYTES), chunk(b"last", True)])
    server = await raw.raw_http_proxy_server(0, bridge)
    draining = asyncio.Event()
    release = asyncio.Event()

    class Writer:
        def write(self, value: bytes) -> None:
            pass

        async def drain(self) -> None:
            draining.set()
            await release.wait()

    writer: Any = Writer()
    task = asyncio.create_task(server._send_raw_body(writer, "response-1"))
    try:
        await asyncio.wait_for(draining.wait(), 2)
        assert sum(method == "raw_http_read" for method, _ in bridge.calls) == 1
        release.set()
        await asyncio.wait_for(task, 2)
        assert sum(method == "raw_http_read" for method, _ in bridge.calls) == 2
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("method,status", [("HEAD", 200), ("GET", 204), ("GET", 304)])
def test_bodyless_response_framing(method: str, status: int) -> None:
    block, has_body = raw._response_headers(status, [["Content-Length", "18"]], method)
    assert not has_body
    assert b"Transfer-Encoding" not in block
    assert (b"Content-Length" in block) == (status != 204)


@pytest.mark.parametrize(
    "headers",
    [
        [["X-Bad", "ok\r\nInjected: yes"]],
        [["", "value"]],
        [["Bad Name", "value"]],
        [["X-Bad", "\u0100"]],
        {"Content-Type": "application/json"},
    ],
)
def test_invalid_response_headers_fail_closed(headers: Any) -> None:
    with pytest.raises(raw.RawHTTPProtocolError):
        raw._response_headers(200, headers, "POST")


@pytest.mark.parametrize(
    "body",
    [b"-1\r\n", b"100000000\r\n", b"0\r\nX-Trailer: value\r\n\r\n"],
)
async def test_chunked_request_limits_and_trailers_fail_before_rpc(body: bytes) -> None:
    server = await raw.raw_http_proxy_server(0, AsyncMock())
    reader = asyncio.StreamReader()
    reader.feed_data(body)
    reader.feed_eof()
    with pytest.raises(ValueError):
        await server._read_raw_chunked(reader)


async def test_chunked_request_entity_bytes_are_unchanged() -> None:
    server = await raw.raw_http_proxy_server(0, AsyncMock())
    reader = asyncio.StreamReader()
    reader.feed_data(b"2;extension=x\r\n\xff\x00\r\n3\r\nabc\r\n0\r\n\r\n")
    reader.feed_eof()
    assert await server._read_raw_chunked(reader) == b"\xff\x00abc"


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionError("private details"),
        {"version": 1, "error": "upstream_disconnected"},
    ],
)
async def test_preheader_failure_is_a_local_bridge_error(failure: Any) -> None:
    bridge = Bridge()

    async def service(method: str, /, **params: Any) -> dict[str, Any]:
        if method == "raw_http_start":
            if isinstance(failure, Exception):
                raise failure
            return failure
        return await bridge(method, **params)

    async with running(service) as server:
        reader, writer = await request(server)
        try:
            wire = await asyncio.wait_for(reader.read(), 2)
            assert wire.startswith(b"HTTP/1.1 502 Bad Gateway\r\n")
            assert b"raw_bridge_error" in wire
            assert b"private details" not in wire
        finally:
            writer.close()
            await writer.wait_closed()
    assert not bridge.closed.is_set()


async def test_shutdown_during_start_finishes_and_closes_the_handle() -> None:
    bridge = Bridge()
    bridge.start_gate = asyncio.Event()
    async with running(bridge) as server:
        _, writer = await request(server)
        await asyncio.wait_for(bridge.started.wait(), 2)
        stopping = asyncio.create_task(server.stop())
        await asyncio.sleep(0)
        bridge.start_gate.set()
        await asyncio.wait_for(stopping, 2)
        assert bridge.closed.is_set()
        writer.close()
        await writer.wait_closed()


@pytest.mark.parametrize(
    "headers",
    [
        b"Content-Length: -1\r\n",
        b"Content-Length: 52428801\r\n",
        b"Content-Length: 0\r\nContent-Length: 0\r\n",
        b"Content-Length: 0\r\nTransfer-Encoding: chunked\r\n",
        b"Transfer-Encoding: gzip\r\n",
    ],
)
async def test_ambiguous_or_oversized_request_framing_never_reaches_host(
    headers: bytes,
) -> None:
    bridge = Bridge()
    async with running(bridge) as server:
        reader, writer = await asyncio.open_connection(server.host, server.port)
        writer.write(b"POST /v1/responses HTTP/1.1\r\n" + headers + b"\r\n")
        await writer.drain()
        try:
            assert (await asyncio.wait_for(reader.read(), 2)).startswith(
                b"HTTP/1.1 400 "
            )
        finally:
            writer.close()
            await writer.wait_closed()
    assert not bridge.started.is_set()


async def test_empty_headers_and_unrecognized_method_are_host_policy() -> None:
    bridge = Bridge([chunk(b"policy denied", True)])
    bridge.status = 403
    async with running(bridge) as server:
        reader, writer = await asyncio.open_connection(server.host, server.port)
        writer.write(b"DELETE /vendor/opaque?name=%2F HTTP/1.1\r\n\r\n")
        await writer.drain()
        try:
            headers, chunks = await read_response(reader)
            assert headers.startswith(b"HTTP/1.1 403 Forbidden\r\n")
            assert chunks == [b"policy denied"]
        finally:
            writer.close()
            await writer.wait_closed()
    start = next(
        params for method, params in bridge.calls if method == "raw_http_start"
    )
    assert start["headers"] == []
    assert start["method"] == "DELETE"
    assert start["path"] == "/vendor/opaque?name=%2F"
    assert start["body_b64"] == ""


def test_model_proxy_cli_reports_raw_capability_without_starting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from inspect_sandbox_tools._cli import main as cli

    monkeypatch.setattr(
        "sys.argv", ["inspect-sandbox-tools", "model_proxy", "--capabilities"]
    )
    run = AsyncMock()
    monkeypatch.setattr(cli, "run_model_proxy_server", run)
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"raw_http_version": 1}
    run.assert_not_called()
