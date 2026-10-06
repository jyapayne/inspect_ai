"""Black-box HTTP/file-RPC contract tests, without importing the proxy implementation.

By default these launch the installed source CLI in a subprocess. To exercise a
real extracted PyInstaller bundle, set INSPECT_RAW_HTTP_LAUNCHER to its
inspect-sandbox-tools launcher. Also set INSPECT_RAW_HTTP_IMAGE to a glibc image
(e.g. debian:bookworm-slim) to run that bundle in Docker: the image must have no
python/python3 executable. Only the bundle and a private file-RPC directory are
mounted; Python, pytest, the fake host service, and the mock gateway stay outside.
The test container shares the host network solely to reach its loopback listener;
there is no service TCP listener in the container or provider/model traffic.
"""

import base64
import contextlib
import http.client
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

DEADLINE = 20.0
SERVICE_ROOT = Path("/var/tmp/sandbox-services/bridge_model_service")
REQUEST_BODY = b'{ "model" : "fixture-model", "messages":[], "vendor": {"keep":true} }'
SSE_FIRST = b': vendor-comment\r\nevent: vendor.delta\r\ndata: {"token":"one"}\r\n\r\n'
SSE_LAST = b'data: {"token":"two","vendor_extra":7}\n\ndata: [DONE]\n\n'


def wait_until(predicate: Callable[[], bool], description: str) -> None:
    deadline = time.monotonic() + DEADLINE
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for {description}")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class GatewayPlan:
    status: int = 200
    headers: list[tuple[str, str]] = field(
        default_factory=lambda: [("Content-Type", "application/json")]
    )
    chunks: tuple[bytes, ...] = (b'{"fixture":"nonstream","vendor_extra":7}',)
    streaming: bool = False
    release: threading.Event | None = None
    truncate: bool = False
    first_sent: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)


class Gateway:
    """A real loopback HTTP origin with independently specified wire fixtures."""

    def __init__(self, plan: GatewayPlan) -> None:
        self.plan = plan
        self.requests: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                owner.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": list(self.headers.raw_items()),
                        "body": self.rfile.read(
                            int(self.headers.get("Content-Length", 0))
                        ),
                    }
                )
                self.send_response(plan.status)
                for name, value in plan.headers:
                    self.send_header(name, value)
                if plan.streaming:
                    self.send_header("Transfer-Encoding", "chunked")
                else:
                    self.send_header("Content-Length", str(sum(map(len, plan.chunks))))
                self.end_headers()
                try:
                    for index, chunk in enumerate(plan.chunks):
                        if index and plan.release is not None:
                            if not plan.release.wait(DEADLINE):
                                raise TimeoutError(
                                    "Consumer did not release gateway stream"
                                )
                        if plan.streaming:
                            self.wfile.write(f"{len(chunk):x}\r\n".encode())
                        self.wfile.write(chunk)
                        if plan.streaming:
                            self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        plan.first_sent.set()
                        if plan.truncate:
                            self.close_connection = True
                            self.connection.shutdown(socket.SHUT_RDWR)
                            return
                    if plan.streaming:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    plan.finished.set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def close(self) -> None:
        if self.plan.release is not None:
            self.plan.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(DEADLINE)


@dataclass
class GatewayResponse:
    connection: http.client.HTTPConnection
    response: http.client.HTTPResponse
    lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        # Wake a blocked read before taking its lock. HTTPResponse.close itself
        # must not race read1: it replaces the reader's file object with None.
        if self.connection.sock is not None:
            with contextlib.suppress(OSError):
                self.connection.sock.shutdown(socket.SHUT_RDWR)
        with self.lock:
            self.connection.close()
            self.response.close()


class FileService:
    """An independent host consumer of the existing request/result file envelopes.

    It uses stdlib HTTPResponse.read1, not any production RAW bridge code, to
    expose actual gateway bytes via the frozen four-method RPC contract.
    """

    def __init__(self, directory: Path, gateway: Gateway, port: int) -> None:
        self.directory = directory
        self.gateway = gateway
        self.port = port
        self.calls: list[dict[str, Any]] = []
        self.responses: dict[str, GatewayResponse] = {}
        self.policy_bodies: dict[str, bytes] = {}
        self.closed: list[str] = []
        self.chunk_sizes: list[int] = []
        self.errors: list[BaseException] = []
        self.bad_versions: dict[str, int] = {}
        self.preheader_error = False
        self.read_error = False
        self.reject_ready = False
        self.bound_at_ready = False
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.start_entered = threading.Event()
        self.start_release: threading.Event | None = None
        self.ready_release: threading.Event | None = None
        self.read_entered = threading.Event()
        self.read_release: threading.Event | None = None
        self.active_reads = 0
        self.max_active_reads = 0
        self.lock = threading.Lock()
        (directory / "requests").mkdir(parents=True)
        (directory / "responses").mkdir()
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def _poll(self) -> None:
        while not self.stop_event.wait(0.01):
            for request in (self.directory / "requests").glob("*.json"):
                try:
                    payload = json.loads(request.read_bytes())
                except (json.JSONDecodeError, FileNotFoundError):
                    continue  # Existing client writes requests non-atomically.
                request.unlink()
                self.calls.append(payload)
                self.pool.submit(self._answer, payload)

    def _answer(self, request: dict[str, Any]) -> None:
        try:
            result = self._dispatch(request["method"], request["params"])
            envelope: dict[str, Any] = {"result": result}
        except BaseException as error:
            self.errors.append(error)
            envelope = {"error": str(error)}
        response = self.directory / "responses" / (request["id"] + ".json")
        temporary = response.with_suffix(".partial")
        temporary.write_text(json.dumps(envelope))
        temporary.replace(response)

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "generate_completions":
            return {
                "id": "legacy-fixture",
                "object": "chat.completion",
                "created": 123,
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "legacy"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        assert params["version"] == 1
        version = self.bad_versions.get(method, 1)
        if method == "raw_http_ready":
            with socket.create_connection(("127.0.0.1", self.port), timeout=5):
                self.bound_at_ready = True
            self.ready.set()
            if self.ready_release is not None:
                assert self.ready_release.wait(DEADLINE)
            if self.reject_ready:
                return {"version": version, "error": "unsupported_raw_http"}
            return {"version": version}
        if method == "raw_http_start":
            self.start_entered.set()
            if self.start_release is not None:
                assert self.start_release.wait(DEADLINE)
            if self.preheader_error:
                return {"version": version, "error": "upstream_disconnected"}
            response_id = uuid.uuid4().hex
            pairs = params["headers"]
            names = [name.lower() for name, _ in pairs]
            if len(names) != len(set(names)):
                self.policy_bodies[response_id] = (
                    b'{"error":"duplicate_header","host_policy":true}'
                )
                return {
                    "version": version,
                    "response_id": response_id,
                    "status": 403,
                    "headers": [["Content-Type", "application/json"]],
                }
            connection = http.client.HTTPConnection(
                "127.0.0.1", self.gateway.port, timeout=DEADLINE
            )
            connection.putrequest(
                params["method"],
                params["path"],
                skip_host=True,
                skip_accept_encoding=True,
            )
            for name, value in pairs:
                connection.putheader(name, value)
            connection.endheaders(base64.b64decode(params["body_b64"], validate=True))
            response = connection.getresponse()
            self.responses[response_id] = GatewayResponse(connection, response)
            return {
                "version": version,
                "response_id": response_id,
                "status": response.status,
                "headers": [list(pair) for pair in response.getheaders()],
            }
        response_id = params["response_id"]
        if method == "raw_http_close":
            pair = self.responses.pop(response_id, None)
            if pair is not None:
                pair.close()
            self.policy_bodies.pop(response_id, None)
            self.closed.append(response_id)
            return {"version": version}
        assert method == "raw_http_read"
        self.read_entered.set()
        if self.read_release is not None:
            assert self.read_release.wait(DEADLINE)
        if self.read_error:
            return {"version": version, "error": "upstream_disconnected"}
        with self.lock:
            self.active_reads += 1
            self.max_active_reads = max(self.max_active_reads, self.active_reads)
        try:
            if response_id in self.policy_bodies:
                chunk = self.policy_bodies[response_id]
                self.policy_bodies[response_id] = b""
            else:
                try:
                    state = self.responses[response_id]
                    with state.lock:
                        if response_id not in self.responses:
                            return {
                                "version": version,
                                "error": "upstream_disconnected",
                            }
                        chunk = state.response.read1(65536)
                except (OSError, http.client.HTTPException, KeyError):
                    return {"version": version, "error": "upstream_disconnected"}
            self.chunk_sizes.append(len(chunk))
            return {
                "version": version,
                "body_b64": base64.b64encode(chunk).decode(),
                "eof": not chunk,
            }
        finally:
            with self.lock:
                self.active_reads -= 1

    def close(self) -> None:
        for release in (self.start_release, self.ready_release, self.read_release):
            if release is not None:
                release.set()
        if self.gateway.plan.release is not None:
            self.gateway.plan.release.set()
        self.stop_event.set()
        self.thread.join(DEADLINE)
        self.pool.shutdown(wait=True)
        for state in self.responses.values():
            state.close()


class Bridge:
    def __init__(
        self, service: FileService, instance: str, version: str | None
    ) -> None:
        self.service = service
        self.port = service.port
        self.name: str | None = None
        self.log = service.directory / "proxy.log"
        env = os.environ.copy()
        env.pop("BRIDGE_MODEL_SERVICE_RAW_HTTP_VERSION", None)
        env.pop("BRIDGE_MODEL_EVENT_METADATA_HEADERS", None)
        settings = {
            "BRIDGE_MODEL_SERVICE_PORT": str(self.port),
            "BRIDGE_MODEL_SERVICE_INSTANCE": instance,
        }
        if version is not None:
            settings["BRIDGE_MODEL_SERVICE_RAW_HTTP_VERSION"] = version
        env.update(settings)
        launcher = os.environ.get("INSPECT_RAW_HTTP_LAUNCHER")
        image = os.environ.get("INSPECT_RAW_HTTP_IMAGE")
        if image:
            assert launcher, (
                "Docker consumer proof requires an extracted built launcher"
            )
            bundle = Path(launcher).resolve().parent
            self.name = "inspect-raw-proof-" + uuid.uuid4().hex
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                self.name,
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--network",
                "host",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--memory",
                "256m",
                "--cpus",
                "1",
                "--pids-limit",
                "64",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                "/tmp",
                "--mount",
                f"type=bind,src={bundle},dst=/bundle,readonly",
                "--mount",
                f"type=bind,src={service.directory},dst={SERVICE_ROOT / instance}",
            ]
            for key, value in settings.items():
                command += ["--env", f"{key}={value}"]
            command += [
                image,
                "sh",
                "-ec",
                "if command -v python || command -v python3; then exit 91; fi; "
                "printf 'INSPECT_RAW_PROOF_NO_IMAGE_PYTHON\\n'; "
                "exec /bundle/inspect-sandbox-tools model_proxy",
            ]
        elif launcher:
            command = [str(Path(launcher).resolve()), "model_proxy"]
        else:
            command = [
                sys.executable,
                "-m",
                "inspect_sandbox_tools._cli.main",
                "model_proxy",
            ]
        with self.log.open("wb") as log:
            self.process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )

    def check_alive(self) -> None:
        assert self.process.poll() is None, self.log.read_text()
        assert not self.service.errors, repr(self.service.errors)

    def wait_ready(self, raw: bool = True) -> None:
        def available() -> bool:
            self.check_alive()
            if raw:
                return self.service.ready.is_set()
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    return True
            except OSError:
                return False

        wait_until(available, "proxy readiness")
        if raw:
            assert self.service.bound_at_ready

    def request(
        self,
        body: bytes = REQUEST_BODY,
        path: str = "/v1/chat/completions?fixture=raw%2Bbytes",
        headers: list[tuple[str, str]] | None = None,
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=DEADLINE
        )
        connection.putrequest("POST", path)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(len(body)))
        for name, value in headers or []:
            connection.putheader(name, value)
        connection.endheaders(body)
        return connection, connection.getresponse()

    def socket_request(self, body: bytes = REQUEST_BODY) -> socket.socket:
        client = socket.create_connection(("127.0.0.1", self.port), timeout=DEADLINE)
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: fixture\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )
        return client

    def assert_closed(self) -> None:
        wait_until(lambda: bool(self.service.closed), "response close RPC")
        assert not self.service.responses
        assert not self.service.policy_bodies
        assert not self.service.errors, repr(self.service.errors)

    def close(self) -> None:
        if self.name is not None:
            subprocess.run(
                ["docker", "rm", "-f", self.name],
                capture_output=True,
                check=False,
                timeout=DEADLINE,
            )
        elif self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


@pytest.fixture
def bridge_factory(tmp_path: Path) -> Iterator[Callable[..., Any]]:
    @contextlib.contextmanager
    def start(
        plan: GatewayPlan | None = None,
        version: str | None = "1",
        configure: Callable[[FileService], None] | None = None,
        ready: bool = True,
    ) -> Iterator[Bridge]:
        instance = "consumer-" + uuid.uuid4().hex
        directory = (
            tmp_path / instance
            if os.environ.get("INSPECT_RAW_HTTP_IMAGE")
            else SERVICE_ROOT / instance
        )
        gateway = Gateway(plan or GatewayPlan())
        service = FileService(directory, gateway, free_port())
        if configure is not None:
            configure(service)
        bridge = Bridge(service, instance, version)
        try:
            if ready:
                bridge.wait_ready(raw=version is not None)
            yield bridge
        finally:
            service.close()
            bridge.close()
            gateway.close()
            shutil.rmtree(directory)

    yield start


@pytest.mark.parametrize("parallel", [None, True, False])
@pytest.mark.parametrize("stream", [None, False, True])
def test_raw_consumer_preserves_request_bytes(
    bridge_factory: Any, parallel: bool | None, stream: bool | None
) -> None:
    fields = b""
    if parallel is not None:
        fields += b', "parallel_tool_calls" : ' + (b"true" if parallel else b"false")
    if stream is not None:
        fields += b', "stream" : ' + (b"true" if stream else b"false")
    body = REQUEST_BODY[:-1] + fields + b"}\n"
    with bridge_factory() as bridge:
        connection, response = bridge.request(body)
        try:
            assert response.status == 200
            assert response.read() == GatewayPlan().chunks[0]
        finally:
            connection.close()
        request = bridge.service.gateway.requests[0]
        assert request["body"] == body
        assert request["path"] == "/v1/chat/completions?fixture=raw%2Bbytes"
        call = next(
            call for call in bridge.service.calls if call["method"] == "raw_http_start"
        )
        assert base64.b64decode(call["params"]["body_b64"]) == body
        assert call["params"]["method"] == "POST"
        assert not any(
            call["method"].startswith("generate_") for call in bridge.service.calls
        )
        bridge.assert_closed()


def test_raw_consumer_preserves_provider_error(bridge_factory: Any) -> None:
    body = (
        b'{ "error":{"message":"slow down","vendor_code":42}, "vendor_root":[1,2] }\n'
    )
    plan = GatewayPlan(
        status=429,
        headers=[
            ("Content-Type", "application/json"),
            ("Retry-After", "17"),
            ("X-Vendor", "first"),
            ("X-Vendor", "second"),
        ],
        chunks=(body,),
    )
    with bridge_factory(plan) as bridge:
        connection, response = bridge.request(REQUEST_BODY[:-1] + b',"stream":true}')
        try:
            assert response.status == 429
            assert response.getheader("Retry-After") == "17"
            assert [
                value
                for key, value in response.getheaders()
                if key.lower() == "x-vendor"
            ] == ["first", "second"]
            assert response.read() == body
        finally:
            connection.close()
        bridge.assert_closed()


def test_raw_consumer_delegates_duplicate_header_policy(bridge_factory: Any) -> None:
    with bridge_factory() as bridge:
        connection, response = bridge.request(
            headers=[("X-Tenant", "first"), ("x-tenant", "second")]
        )
        try:
            assert response.status == 403
            assert response.read() == b'{"error":"duplicate_header","host_policy":true}'
        finally:
            connection.close()
        call = next(
            call for call in bridge.service.calls if call["method"] == "raw_http_start"
        )
        assert [
            value
            for name, value in call["params"]["headers"]
            if name.lower() == "x-tenant"
        ] == ["first", "second"]
        assert not bridge.service.gateway.requests
        bridge.assert_closed()


def test_raw_consumer_streams_before_gateway_finishes(bridge_factory: Any) -> None:
    release = threading.Event()
    plan = GatewayPlan(
        headers=[("Content-Type", "text/event-stream")],
        chunks=(SSE_FIRST, SSE_LAST),
        streaming=True,
        release=release,
    )
    with bridge_factory(plan) as bridge:
        connection, response = bridge.request(REQUEST_BODY[:-1] + b',"stream":true}')
        try:
            assert response.status == 200
            assert response.getheader("Content-Type") == "text/event-stream"
            assert response.read(len(SSE_FIRST)) == SSE_FIRST
            assert plan.first_sent.is_set()
            assert not plan.finished.is_set(), (
                "Proxy must not aggregate the entire stream"
            )
            release.set()
            assert response.read() == SSE_LAST
        finally:
            release.set()
            connection.close()
        bridge.assert_closed()
        assert bridge.service.max_active_reads == 1


def test_raw_consumer_binary_response_and_bounded_chunks(bridge_factory: Any) -> None:
    body = bytes(range(256)) * 1024 + b"\xff\x00end"
    with bridge_factory(
        GatewayPlan(
            headers=[("Content-Type", "application/octet-stream")], chunks=(body,)
        )
    ) as bridge:
        connection, response = bridge.request(b"\x00\xffnot-json\r\n")
        try:
            assert response.read() == body
        finally:
            connection.close()
        assert bridge.service.gateway.requests[0]["body"] == b"\x00\xffnot-json\r\n"
        bridge.assert_closed()
        assert max(bridge.service.chunk_sizes) <= 65536
        assert sum(size > 0 for size in bridge.service.chunk_sizes) >= 5
        assert bridge.service.max_active_reads == 1


def test_raw_consumer_partial_gateway_failure_is_not_success_eof(
    bridge_factory: Any,
) -> None:
    plan = GatewayPlan(
        headers=[("Content-Type", "text/event-stream")],
        chunks=(SSE_FIRST,),
        streaming=True,
        truncate=True,
    )
    with bridge_factory(plan) as bridge:
        connection, response = bridge.request()
        try:
            assert response.status == 200
            assert response.read(len(SSE_FIRST)) == SSE_FIRST
            with pytest.raises((http.client.IncompleteRead, ConnectionResetError)):
                response.read()
        finally:
            connection.close()
        bridge.assert_closed()


def test_raw_consumer_client_disconnect_closes_active_response(
    bridge_factory: Any,
) -> None:
    release = threading.Event()
    plan = GatewayPlan(
        headers=[("Content-Type", "text/event-stream")],
        chunks=(SSE_FIRST, SSE_LAST),
        streaming=True,
        release=release,
    )
    with bridge_factory(plan) as bridge:
        client = bridge.socket_request()
        received = b""
        while SSE_FIRST not in received:
            received += client.recv(65536)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()
        # Do not release the gateway to make EOF/disconnect detection the oracle.
        wait_until(
            lambda: any(
                call["method"] == "raw_http_close" for call in bridge.service.calls
            ),
            "close while upstream read is blocked",
        )
        release.set()
        bridge.assert_closed()


def test_raw_consumer_disconnect_during_start_closes_late_handle(
    bridge_factory: Any,
) -> None:
    release = threading.Event()

    def configure(service: FileService) -> None:
        service.start_release = release

    with bridge_factory(configure=configure) as bridge:
        client = bridge.socket_request()
        assert bridge.service.start_entered.wait(DEADLINE)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()
        release.set()
        bridge.assert_closed()


@pytest.mark.parametrize("method", ["raw_http_start", "raw_http_read"])
def test_raw_consumer_transport_errors_are_explicit(
    bridge_factory: Any, method: str
) -> None:
    def configure(service: FileService) -> None:
        service.preheader_error = method == "raw_http_start"
        service.read_error = method == "raw_http_read"

    with bridge_factory(configure=configure) as bridge:
        connection, response = bridge.request()
        try:
            if method == "raw_http_start":
                assert response.status == 502
                assert b"fixture" not in response.read()
                assert not bridge.service.gateway.requests
            else:
                assert response.status == 200
                with pytest.raises((http.client.IncompleteRead, ConnectionResetError)):
                    response.read()
                bridge.assert_closed()
        finally:
            connection.close()


@pytest.mark.parametrize("version", ["0", "2", "true", ""])
def test_raw_consumer_unsupported_environment_version_fails_closed(
    bridge_factory: Any, version: str
) -> None:
    with bridge_factory(version=version, ready=False) as bridge:
        assert bridge.process.wait(timeout=DEADLINE) != 0
        assert not bridge.service.calls
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", bridge.port), timeout=0.1)


@pytest.mark.parametrize("reject", [False, True])
def test_raw_consumer_failed_capability_handshake_exits(
    bridge_factory: Any, reject: bool
) -> None:
    def configure(service: FileService) -> None:
        if reject:
            service.reject_ready = True
        else:
            service.bad_versions["raw_http_ready"] = 2

    with bridge_factory(configure=configure, ready=False) as bridge:
        assert bridge.process.wait(timeout=DEADLINE) != 0
        assert bridge.service.bound_at_ready
        assert [call["method"] for call in bridge.service.calls] == ["raw_http_ready"]


@pytest.mark.parametrize("method", ["raw_http_start", "raw_http_read"])
def test_raw_consumer_response_version_mismatch_fails_closed(
    bridge_factory: Any, method: str
) -> None:
    def configure(service: FileService) -> None:
        service.bad_versions[method] = 2

    with bridge_factory(configure=configure) as bridge:
        connection, response = bridge.request()
        try:
            if method == "raw_http_start":
                assert response.status == 502
                response.read()
            else:
                assert response.status == 200
                with pytest.raises((http.client.IncompleteRead, ConnectionResetError)):
                    response.read()
            bridge.assert_closed()
        finally:
            connection.close()


def test_raw_consumer_handshake_precedes_request_forwarding(
    bridge_factory: Any,
) -> None:
    release = threading.Event()

    def configure(service: FileService) -> None:
        service.ready_release = release

    with bridge_factory(configure=configure, ready=False) as bridge:
        assert bridge.service.ready.wait(DEADLINE)
        assert bridge.service.bound_at_ready
        client = bridge.socket_request()
        try:
            client.settimeout(0.2)
            with pytest.raises(socket.timeout):
                client.recv(1)
            assert [call["method"] for call in bridge.service.calls] == [
                "raw_http_ready"
            ]
            release.set()
            client.settimeout(DEADLINE)
            response = http.client.HTTPResponse(client)
            response.begin()
            assert response.status == 200
            assert response.read() == GatewayPlan().chunks[0]
            bridge.assert_closed()
        finally:
            release.set()
            client.close()


@pytest.mark.parametrize("parallel", [None, True, False])
def test_legacy_default_remains_legacy_over_real_file_rpc(
    bridge_factory: Any, parallel: bool | None
) -> None:
    body: dict[str, Any] = {
        "model": "fixture-model",
        "messages": [],
        "vendor": {"keep": True},
    }
    if parallel is not None:
        body["parallel_tool_calls"] = parallel
    with bridge_factory(version=None) as bridge:
        connection, response = bridge.request(
            json.dumps(body).encode(), path="/v1/chat/completions"
        )
        try:
            assert response.status == 200
            assert json.loads(response.read())["id"] == "legacy-fixture"
        finally:
            connection.close()
        assert [call["method"] for call in bridge.service.calls] == [
            "generate_completions"
        ]
        forwarded = bridge.service.calls[0]["params"]["json_data"]
        assert forwarded == {**body, "parallel_tool_calls": False}
        assert not bridge.service.gateway.requests


def test_built_consumer_image_has_no_python(bridge_factory: Any) -> None:
    if not os.environ.get("INSPECT_RAW_HTTP_IMAGE"):
        pytest.skip(
            "Set INSPECT_RAW_HTTP_IMAGE and INSPECT_RAW_HTTP_LAUNCHER for built-binary proof"
        )
    with bridge_factory() as bridge:
        assert "INSPECT_RAW_PROOF_NO_IMAGE_PYTHON" in bridge.log.read_text()
        connection, response = bridge.request()
        try:
            assert response.status == 200
            assert response.read() == GatewayPlan().chunks[0]
        finally:
            connection.close()
        bridge.assert_closed()


def test_raw_consumer_client_read_timeout_closes_response(bridge_factory: Any) -> None:
    release = threading.Event()

    def configure(service: FileService) -> None:
        service.read_release = release

    with bridge_factory(configure=configure) as bridge:
        client = bridge.socket_request()
        received = b""
        while b"\r\n\r\n" not in received:
            received += client.recv(65536)
        assert bridge.service.read_entered.wait(DEADLINE)
        client.settimeout(0.1)
        with pytest.raises(socket.timeout):
            client.recv(1)
        client.close()
        # Consumer cancellation, not a manufactured provider timeout response.
        wait_until(
            lambda: any(
                call["method"] == "raw_http_close" for call in bridge.service.calls
            ),
            "close after client timeout",
        )
        release.set()
        bridge.assert_closed()


def test_raw_consumer_revoked_host_response_aborts_after_headers(
    bridge_factory: Any,
) -> None:
    release = threading.Event()

    def configure(service: FileService) -> None:
        service.read_release = release

    with bridge_factory(configure=configure) as bridge:
        connection, response = bridge.request()
        try:
            assert response.status == 200
            assert bridge.service.read_entered.wait(DEADLINE)
            # Model the frozen host revocation result, not Hawk's lease logic.
            bridge.service.read_error = True
            release.set()
            with pytest.raises((http.client.IncompleteRead, ConnectionResetError)):
                response.read()
            bridge.assert_closed()
        finally:
            release.set()
            connection.close()


def test_raw_consumer_slow_client_applies_rpc_backpressure(bridge_factory: Any) -> None:
    # Reuse one chunk rather than allocating a 64 MiB fixture on the host.
    plan = GatewayPlan(
        headers=[("Content-Type", "application/octet-stream")],
        chunks=(b"x" * 65536,) * 1024,
    )
    with bridge_factory(plan) as bridge:
        client = socket.socket()
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        client.settimeout(DEADLINE)
        client.connect(("127.0.0.1", bridge.port))
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: fixture\r\nContent-Length: 0\r\n\r\n"
        )
        assert bridge.service.read_entered.wait(DEADLINE)
        previous = -1
        unchanged_since = time.monotonic()

        def stalled() -> bool:
            nonlocal previous, unchanged_since
            bridge.check_alive()
            current = len(bridge.service.chunk_sizes)
            if current != previous:
                previous = current
                unchanged_since = time.monotonic()
            return current > 0 and time.monotonic() - unchanged_since >= 0.75

        try:
            wait_until(stalled, "bounded RPC reads with a non-reading client")
            assert not plan.finished.is_set()
            assert sum(bridge.service.chunk_sizes) < 16 * 1024 * 1024
            assert max(bridge.service.chunk_sizes) <= 65536
            assert bridge.service.max_active_reads == 1
        finally:
            client.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            client.close()
        bridge.assert_closed()
