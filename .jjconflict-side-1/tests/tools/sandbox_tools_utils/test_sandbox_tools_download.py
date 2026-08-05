"""Resolution must fall through to the local build when the bucket is unusable.

A DNS failure on the (placeholder) distribution bucket used to escape
`_download_from_s3` as `httpx.ConnectError`, killing injection with
`SandboxInjectionError` before the local-build fallback could run.

`_download_from_s3` verifies against the vendored SHA256SUMS (see
`test_sandbox_tools_digests.py` for the verified/unverified/mismatch paths);
these tests cover names with no sums entry, which take the warn-and-download
unverified path and so exercise `httpx.stream` directly, same as production
does for an unpinned name.
"""

from pathlib import Path
from typing import Iterator

import httpx
import pytest
from unittest.mock import MagicMock, patch

from inspect_ai.tool._sandbox_tools_utils import sandbox as sandbox_mod


class _FakeStream:
    """Drop-in replacement for the context manager returned by httpx.stream."""

    def __init__(self, status_code: int, content: bytes):
        self._status_code = status_code
        self._content = content

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self._status_code >= 400:
            request = httpx.Request("GET", "http://test.example")
            response = httpx.Response(self._status_code, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self._status_code}", request=request, response=response
            )

    def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
        size = chunk_size or 1024
        for start in range(0, len(self._content), size):
            yield self._content[start : start + size]


async def test_default_url_attempts_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-tree default URL is a REAL distribution point and must be tried.

    (A previous revision short-circuited on the default when it was a
    placeholder; that guard must never come back now that the default works.)
    """
    filename = "inspect-sandbox-tools-amd64-v26-tl1"  # no SHA256SUMS entry
    monkeypatch.setattr(sandbox_mod, "_binaries_dir", lambda: tmp_path)

    requested: list[str] = []

    def stream(method: str, url: str, **kwargs: object) -> _FakeStream:
        requested.append(url)
        return _FakeStream(404, b"")

    # NOT overriding _BUCKET_BASE_URL: the point is that the DEFAULT gets tried.
    assert sandbox_mod._BUCKET_BASE_URL == sandbox_mod._DEFAULT_BUCKET_BASE_URL
    with patch("inspect_ai._util.download.httpx.stream", MagicMock(side_effect=stream)):
        assert (
            await sandbox_mod._download_from_s3(filename)
            is False  # 404 degrades gracefully...
        )
    assert requested == [
        f"{sandbox_mod._DEFAULT_BUCKET_BASE_URL}/{filename}"
    ]  # ...but the network WAS attempted against the default URL.


async def test_unreachable_bucket_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transport errors (DNS, refused, timeout) mean 'not available', not fatal."""
    filename = "inspect-sandbox-tools-amd64-v26-tl1"  # no SHA256SUMS entry
    monkeypatch.setattr(sandbox_mod, "_binaries_dir", lambda: tmp_path)
    monkeypatch.setattr(
        sandbox_mod, "_BUCKET_BASE_URL", "https://definitely-not-resolvable.invalid"
    )

    def stream(method: str, url: str, **kwargs: object) -> _FakeStream:
        raise httpx.ConnectError("DNS resolution failed")

    with patch("inspect_ai._util.download.httpx.stream", MagicMock(side_effect=stream)):
        assert await sandbox_mod._download_from_s3(filename) is False


async def test_http_500_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-404/403 HTTP errors keep raising: the bucket exists but is broken."""
    filename = "inspect-sandbox-tools-amd64-v26-tl1"  # no SHA256SUMS entry
    monkeypatch.setattr(sandbox_mod, "_binaries_dir", lambda: tmp_path)
    monkeypatch.setattr(sandbox_mod, "_BUCKET_BASE_URL", "https://bucket.example")

    stream_mock = MagicMock(side_effect=lambda method, url, **kwargs: _FakeStream(500, b""))
    with patch("inspect_ai._util.download.httpx.stream", stream_mock):
        with pytest.raises(httpx.HTTPStatusError):
            await sandbox_mod._download_from_s3(filename)
