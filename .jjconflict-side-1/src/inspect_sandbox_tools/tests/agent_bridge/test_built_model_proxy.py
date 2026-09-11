"""What a bridged client observes from the BUILT sandbox-tools executable.

The in-process tests exercise `proxy.py` from source; a consumer never runs that.
It runs the PyInstaller bundle that Inspect injects into the sandbox, so a fix
that reaches the source but not the shipped artifact is invisible to every
test above this one. These tests start the built launcher's `model_proxy`, play
the host side of the bridge's filesystem RPC, and read the wire with a real
Anthropic SDK client.

Opt-in: `pytest --sandbox-tools-artifact PATH` (see `built_sandbox_tools` in
conftest). Without an artifact they skip, and a skip is "did not run".
"""

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from anthropic import Anthropic, APIStatusError

from tests.conftest import ANTHROPIC_WIRE_ERROR_TYPES

# Mirrors the proxy's own constants: the RPC root is fixed in `proxy.py`
# (`_bridge_model_service_service_dir`) and the marker key is `PROVIDER_ERROR_KEY`.
_SERVICE_ROOT = Path("/var/tmp/sandbox-services/bridge_model_service")
_PROVIDER_ERROR_KEY = "__inspect_provider_error__"


class StubBridgeHost:
    """The host end of the bridge RPC, answering every request with a provider error.

    The proxy writes `requests/<id>.json` and polls `responses/<id>.json`; Inspect's
    sandbox service does the same from the host. This stand-in makes every model
    call fail with the status set on `status`, so a test can choose what the
    provider "said" and watch what the client is told.
    """

    def __init__(self, instance: str) -> None:
        self.status = 500
        self.requests = _SERVICE_ROOT / instance / "requests"
        self.responses = _SERVICE_ROOT / instance / "responses"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        for directory in (self.requests, self.responses):
            directory.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        shutil.rmtree(_SERVICE_ROOT / self.requests.parent.name, ignore_errors=True)

    def _serve(self) -> None:
        while not self._stop.is_set():
            for request_path in sorted(self.requests.glob("*.json")):
                try:
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue  # still being written; the next pass picks it up
                request_path.unlink()
                response = {
                    "id": request["id"],
                    "result": {
                        _PROVIDER_ERROR_KEY: {
                            "status": self.status,
                            "message": f"provider said {self.status}",
                        }
                    },
                }
                # Written whole, then renamed, so the proxy never sees a partial file.
                partial = self.responses / f"{request['id']}.json.partial"
                partial.write_text(json.dumps(response), encoding="utf-8")
                os.replace(partial, self.responses / f"{request['id']}.json")
            time.sleep(0.02)


class BuiltModelProxy:
    """A running `model_proxy` from the built launcher, with its stub host."""

    def __init__(self, base_url: str, host: StubBridgeHost) -> None:
        self.base_url = base_url
        self.host = host


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _listening(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _stop(proxy: subprocess.Popen[str]) -> None:
    proxy.terminate()
    try:
        proxy.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proxy.kill()
        proxy.wait()


@pytest.fixture(scope="module")
def built_model_proxy(built_sandbox_tools: Path) -> Iterator[BuiltModelProxy]:
    host = StubBridgeHost(instance=f"test-{uuid.uuid4().hex}")
    host.start()
    try:
        port = _free_port()
        proxy = subprocess.Popen(
            [str(built_sandbox_tools), "model_proxy"],
            env={
                **os.environ,
                "BRIDGE_MODEL_SERVICE_PORT": str(port),
                "BRIDGE_MODEL_SERVICE_INSTANCE": host.requests.parent.name,
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            deadline = time.monotonic() + 30
            while not _listening(port):
                if proxy.poll() is not None:
                    output = proxy.stdout.read() if proxy.stdout else ""
                    raise RuntimeError(
                        f"built model_proxy exited {proxy.returncode} before serving:\n{output}"
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError(f"built model_proxy never listened on {port}")
                time.sleep(0.1)
            yield BuiltModelProxy(f"http://127.0.0.1:{port}", host)
        finally:
            _stop(proxy)
    finally:
        host.stop()


@pytest.mark.parametrize(("status", "expected_type"), ANTHROPIC_WIRE_ERROR_TYPES)
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
def test_built_proxy_reports_the_providers_error_type_to_an_anthropic_client(
    built_model_proxy: BuiltModelProxy, status: int, expected_type: str, stream: bool
) -> None:
    """The shipped executable tells a real Anthropic client the provider's error type.

    Same contract as the in-process `test_anthropic_sdk_observes_the_providers_error_type`,
    proven on the artifact a consumer actually runs.
    """
    built_model_proxy.host.status = status
    client = Anthropic(
        api_key="test", base_url=built_model_proxy.base_url, max_retries=0
    )
    request: dict[str, Any] = {
        "model": "claude-x",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    }
    with pytest.raises(APIStatusError) as exc_info:
        if stream:
            with client.messages.stream(**request) as events:
                for _ in events:
                    pass
        else:
            client.messages.create(**request)

    body = exc_info.value.body
    assert isinstance(body, dict)
    assert body["error"]["type"] == expected_type
    assert body["error"]["message"] == f"provider said {status}"
    if not stream:
        assert exc_info.value.status_code == status
