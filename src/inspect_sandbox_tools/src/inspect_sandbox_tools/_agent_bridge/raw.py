"""Opt-in HTTP byte transport over the existing sandbox model-service RPC."""

from __future__ import annotations

import asyncio
import base64
import binascii
from typing import Any, Awaitable, Callable

from inspect_sandbox_tools._agent_bridge.proxy import (
    HOP_BY_HOP,
    MAX_BODY_BYTES,
    MAX_HEADER_BYTES,
    READ_TIMEOUT_S,
    WRITE_TIMEOUT_S,
    AsyncHTTPServer,
    _call_bridge_model_service_async,
    _http_reason_phrase,
    _is_valid_http_header_name,
)

RAW_HTTP_VERSION = 1
RAW_HTTP_VERSION_ENV = "BRIDGE_MODEL_SERVICE_RAW_HTTP_VERSION"
RAW_HTTP_CHUNK_BYTES = 64 * 1024
RAW_HTTP_RPC_TIMEOUT_S = 660
RAW_HTTP_CONTROL_TIMEOUT_S = 30

BridgeService = Callable[..., Awaitable[Any]]
HeaderPairs = list[list[str]]


class RawHTTPProtocolError(RuntimeError):
    """The host did not return a valid response for the negotiated protocol."""


def _result(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != RAW_HTTP_VERSION
        or "error" in value
    ):
        raise RawHTTPProtocolError("Invalid RAW HTTP service response")
    return value


def _response_id(value: Any) -> str | None:
    # Capture the handle even when other response fields fail validation, so it
    # can still be closed. Handles are opaque and scoped by the host service.
    if isinstance(value, dict):
        response_id = value.get("response_id")
        if isinstance(response_id, str) and response_id:
            return response_id
    return None


def _header_pair(name: Any, value: Any) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and _is_valid_http_header_name(name)
        and isinstance(value, str)
        and all(
            character == "\t" or 32 <= ord(character) <= 255 and ord(character) != 127
            for character in value
        )
    )


def _response_headers(status: Any, headers: Any, method: str) -> tuple[bytes, bool]:
    if type(status) is not int or not 200 <= status <= 599:
        raise RawHTTPProtocolError("Invalid RAW HTTP response status")
    if not isinstance(headers, list):
        raise RawHTTPProtocolError("Invalid RAW HTTP response headers")
    for pair in headers:
        if not isinstance(pair, list) or len(pair) != 2 or not _header_pair(*pair):
            raise RawHTTPProtocolError("Invalid RAW HTTP response header pair")

    # The RPC carries entity bytes, not upstream HTTP transfer framing. Preserve
    # every end-to-end pair, including duplicates, but frame this connection
    # ourselves. A missing terminal chunk then truthfully signals stream failure.
    excluded = HOP_BY_HOP | {"trailer"}
    for name, value in headers:
        if name.lower() == "connection":
            excluded.update(token.strip().lower() for token in value.split(","))
    has_body = method != "HEAD" and status not in (204, 304)
    if has_body or status == 204:
        excluded.add("content-length")
    lines = [f"HTTP/1.1 {status} {_http_reason_phrase(status)}\r\n"]
    lines.extend(
        f"{name}: {value}\r\n"
        for name, value in headers
        if name.lower() not in excluded
    )
    lines.append("Connection: close\r\n")
    if has_body:
        lines.append("Transfer-Encoding: chunked\r\n")
    lines.append("\r\n")
    block = "".join(lines).encode("latin-1")
    if len(block) > MAX_HEADER_BYTES:
        raise RawHTTPProtocolError("RAW HTTP response headers too large")
    return block, has_body


def _body_chunk(value: Any) -> tuple[bytes, bool]:
    result = _result(value)
    encoded, eof = result.get("body_b64"), result.get("eof")
    if (
        not isinstance(encoded, str)
        or len(encoded) > 4 * ((RAW_HTTP_CHUNK_BYTES + 2) // 3)
        or type(eof) is not bool
    ):
        raise RawHTTPProtocolError("Invalid RAW HTTP body chunk")
    try:
        body = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as ex:
        raise RawHTTPProtocolError("Invalid RAW HTTP base64 body") from ex
    if len(body) > RAW_HTTP_CHUNK_BYTES:
        raise RawHTTPProtocolError("RAW HTTP body chunk too large")
    return body, eof


class RawHTTPServer(AsyncHTTPServer):
    """Byte-preserving mode of the existing loopback server, with no model routes."""

    def __init__(
        self, port: int, call_bridge_model_service_async: BridgeService
    ) -> None:
        super().__init__(host="127.0.0.1", port=port)
        self._call_service = call_bridge_model_service_async
        self._ready = asyncio.Event()
        self._stopped = asyncio.Event()
        self._clients: set[asyncio.Task[Any]] = set()
        self._stop_lock = asyncio.Lock()

    async def _call(self, method: str, /, **params: Any) -> Any:
        timeout = (
            RAW_HTTP_CONTROL_TIMEOUT_S
            if method in ("raw_http_ready", "raw_http_close")
            else RAW_HTTP_RPC_TIMEOUT_S
        )
        return await asyncio.wait_for(
            self._call_service(method, version=RAW_HTTP_VERSION, **params), timeout
        )

    async def start(self) -> None:
        """Bind loopback before proving readiness and protocol support to the host."""
        self._ready.clear()
        self._stopped.clear()
        self.server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        self.port = self.server.sockets[0].getsockname()[1]
        try:
            _result(await self._call("raw_http_ready"))
            self._ready.set()
            print(
                f"RAW HTTP v{RAW_HTTP_VERSION} server running on http://{self.host}:{self.port}"
            )
            # start_server already accepts connections. Waiting for our own
            # lifecycle signal lets cancellation close clients before waiting
            # for the listener (Server.wait_closed also waits for connections).
            await self._stopped.wait()
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop accepting clients and finalize all known response handles."""
        async with self._stop_lock:
            server = self.server
            if server is not None:
                server.close()
                self.server = None
            clients = [
                task for task in self._clients if task is not asyncio.current_task()
            ]
            for task in clients:
                task.cancel()
            if clients:
                await asyncio.gather(*clients, return_exceptions=True)
            if server is not None:
                await server.wait_closed()
            self._stopped.set()

    async def _read_raw_request(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> tuple[str, str, HeaderPairs, bytes]:
        line = await asyncio.wait_for(reader.readline(), READ_TIMEOUT_S)
        if len(line) > MAX_HEADER_BYTES:
            raise ValueError("Request line too large")
        parts = line.decode("ascii").strip().split()
        if len(parts) != 3:
            raise ValueError("Invalid request line")
        method, path, version = parts
        if (
            not _is_valid_http_header_name(method)
            or not path.startswith("/")
            or "#" in path
            or any(ord(character) < 33 or ord(character) == 127 for character in path)
            or version not in ("HTTP/1.0", "HTTP/1.1")
        ):
            raise ValueError("Invalid HTTP request target or version")
        headers: HeaderPairs = []
        framing: dict[str, str] = {}
        header_bytes = 0
        while True:
            header_line = await asyncio.wait_for(reader.readline(), READ_TIMEOUT_S)
            header_bytes += len(header_line)
            if header_bytes > MAX_HEADER_BYTES:
                raise ValueError("Header section too large")
            if header_line == b"\r\n":
                break
            if not header_line.endswith(b"\r\n"):
                raise ValueError("Incomplete HTTP request headers")
            name, separator, value = header_line[:-2].decode("latin-1").partition(":")
            value = value.strip(" \t")
            if not separator or not _header_pair(name, value):
                raise ValueError("Invalid HTTP request header")
            headers.append([name, value])
            key = name.lower()
            if key in ("content-length", "transfer-encoding", "expect"):
                if key in framing:
                    raise ValueError("Ambiguous HTTP request framing")
                framing[key] = value
        transfer_encoding = framing.get("transfer-encoding")
        length = framing.get("content-length")
        if transfer_encoding is not None and (
            transfer_encoding.lower() != "chunked" or length is not None
        ):
            raise ValueError("Ambiguous HTTP request framing")
        if length is not None and (not length.isascii() or not length.isdecimal()):
            raise ValueError("Invalid content length")
        content_length = int(length or "0")
        if content_length > MAX_BODY_BYTES:
            raise ValueError("Body too large")
        if framing.get("expect", "").lower() == "100-continue":
            writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            await asyncio.wait_for(writer.drain(), WRITE_TIMEOUT_S)
        if transfer_encoding is not None:
            body = await self._read_raw_chunked(reader)
        elif content_length:
            body = await asyncio.wait_for(
                reader.readexactly(content_length), READ_TIMEOUT_S
            )
        else:
            body = b""
        return method, path, headers, body

    async def _read_raw_chunked(self, reader: asyncio.StreamReader) -> bytes:
        body = bytearray()
        while True:
            line = await asyncio.wait_for(reader.readline(), READ_TIMEOUT_S)
            size_token = line.partition(b";")[0].strip()
            if not size_token or any(
                byte not in b"0123456789abcdefABCDEF" for byte in size_token
            ):
                raise ValueError("Invalid chunk size")
            size = int(size_token, 16)
            if size > MAX_BODY_BYTES - len(body):
                raise ValueError("Body too large")
            if size == 0:
                # RPC v1 has no request-trailer field. Do not silently drop
                # end-to-end data which the host could need for authorization.
                trailer = await asyncio.wait_for(reader.readline(), READ_TIMEOUT_S)
                if trailer != b"\r\n":
                    raise ValueError("Request trailers are not supported")
                return bytes(body)
            body.extend(
                await asyncio.wait_for(reader.readexactly(size), READ_TIMEOUT_S)
            )
            if await asyncio.wait_for(reader.readexactly(2), READ_TIMEOUT_S) != b"\r\n":
                raise ValueError("Invalid chunk terminator")

    async def _send_raw_body(
        self, writer: asyncio.StreamWriter, response_id: str
    ) -> None:
        while True:
            body, eof = _body_chunk(
                await self._call("raw_http_read", response_id=response_id)
            )
            if body:
                writer.write(f"{len(body):X}\r\n".encode("ascii"))
                writer.write(body)
                writer.write(b"\r\n")
                await asyncio.wait_for(writer.drain(), WRITE_TIMEOUT_S)
            if eof:
                writer.write(b"0\r\n\r\n")
                await asyncio.wait_for(writer.drain(), WRITE_TIMEOUT_S)
                return

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._clients.add(task)
        start: asyncio.Task[Any] | None = None
        disconnect: asyncio.Task[None] | None = None
        stream: asyncio.Task[None] | None = None
        response_id: str | None = None
        headers_sent = False
        failure_status = 400
        try:
            await self._ready.wait()
            method, path, headers, body = await self._read_raw_request(reader, writer)
            failure_status = 502
            disconnect = asyncio.create_task(_wait_for_disconnect(reader))
            start = asyncio.create_task(
                self._call(
                    "raw_http_start",
                    method=method,
                    path=path,
                    headers=headers,
                    body_b64=base64.b64encode(body).decode("ascii"),
                )
            )
            del body
            # start allocates the handle on the host. Do not abandon its reply
            # on disconnect/cancellation: finish the bounded RPC and close it.
            value = await asyncio.shield(start)
            response_id = _response_id(value)
            result = _result(value)
            if response_id is None:
                raise RawHTTPProtocolError("Missing RAW HTTP response handle")
            if disconnect.done():
                return
            block, has_body = _response_headers(
                result.get("status"), result.get("headers"), method
            )
            writer.write(block)
            headers_sent = True
            await asyncio.wait_for(writer.drain(), WRITE_TIMEOUT_S)
            if has_body:
                stream = asyncio.create_task(self._send_raw_body(writer, response_id))
                done, _ = await asyncio.wait(
                    (stream, disconnect), return_when=asyncio.FIRST_COMPLETED
                )
                if stream in done:
                    await stream
        except Exception as ex:
            if not headers_sent:
                if isinstance(ex, asyncio.TimeoutError):
                    failure_status = 408 if failure_status == 400 else 504
                try:
                    writer.write(
                        self._build_response(
                            failure_status,
                            {
                                "error": {
                                    "type": "raw_bridge_error",
                                    "message": "RAW bridge request failed",
                                    "code": failure_status,
                                }
                            },
                        )
                    )
                    await asyncio.wait_for(writer.drain(), WRITE_TIMEOUT_S)
                except (ConnectionError, asyncio.TimeoutError):
                    pass
            # After headers, closing without a terminal HTTP chunk is the only
            # truthful representation of a failed stream; never invent SSE/JSON.
        finally:
            cleanup = asyncio.create_task(
                self._finish_client(writer, start, stream, disconnect, response_id)
            )
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            finally:
                self._clients.discard(task)

    async def _finish_client(
        self,
        writer: asyncio.StreamWriter,
        start: asyncio.Task[Any] | None,
        stream: asyncio.Task[None] | None,
        disconnect: asyncio.Task[None] | None,
        response_id: str | None,
    ) -> None:
        pending = [task for task in (stream, disconnect) if task is not None]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        writer.close()
        if response_id is None and start is not None:
            try:
                response_id = _response_id(await asyncio.shield(start))
            except Exception:
                pass
        if response_id is not None:
            try:
                _result(await self._call("raw_http_close", response_id=response_id))
            except Exception:
                # Revocation can remove the service first. Its lease owns the
                # remaining response lifetime in that case.
                pass
        try:
            await asyncio.wait_for(writer.wait_closed(), WRITE_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError):
            pass


async def _wait_for_disconnect(reader: asyncio.StreamReader) -> None:
    # This server handles one request per connection. Drain any pipelined bytes
    # without retaining them, so a disconnect interrupts even a stalled read RPC.
    while await reader.read(8192):
        pass


async def raw_http_proxy_server(
    port: int, call_bridge_model_service_async: BridgeService | None = None
) -> RawHTTPServer:
    """Create the RAW transport; start() performs its mandatory ready handshake."""
    return RawHTTPServer(
        port, call_bridge_model_service_async or _call_bridge_model_service_async
    )
