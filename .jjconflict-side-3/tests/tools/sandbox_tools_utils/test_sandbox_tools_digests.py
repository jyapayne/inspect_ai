"""Tests for sandbox-tools binary digest verification (SHA256SUMS pinning).

Covers the `_digests` owner module, the verified runtime download path in
`sandbox._download_from_s3`, the committed SHA256SUMS format, and the
verification helpers in `scripts/pypi-release.py`. See
`src/inspect_sandbox_tools/design/BINARY_INTEGRITY.md`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Iterator
from unittest.mock import MagicMock, patch

import httpx
import pytest

import inspect_ai.tool._sandbox_tools_utils.sandbox as sandbox_module
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.tool._sandbox_tools_utils._build_config import (
    SandboxToolsArch,
    SandboxToolsBuildConfig,
    config_to_filename,
    filename_to_config,
)
from inspect_ai.tool._sandbox_tools_utils._digests import (
    lookup_digest,
    parse_sha256sums,
    read_sha256sums,
    write_sha256sums,
)
from inspect_ai.tool._sandbox_tools_utils.sandbox import (
    _get_sandbox_tools_fork_revision,
    _get_sandbox_tools_version,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# _digests.py
# ---------------------------------------------------------------------------


def test_digests_write_read_round_trip(tmp_path: Path) -> None:
    entries = {
        "inspect-sandbox-tools-amd64-v1": "a" * 64,
        "inspect-sandbox-tools-arm64-v1": "b" * 64,
    }
    sums = tmp_path / "SHA256SUMS"
    write_sha256sums(entries, sums)
    assert read_sha256sums(sums) == entries
    # standard sha256sum format: two spaces, sorted by filename, no markers
    lines = sums.read_text().splitlines()
    assert lines == [
        f"{'a' * 64}  inspect-sandbox-tools-amd64-v1",
        f"{'b' * 64}  inspect-sandbox-tools-arm64-v1",
    ]


def test_digests_parse_tolerates_binary_marker_and_case() -> None:
    digest = "AB" * 32
    text = f"{digest} *some-file\n\nnot a sums line\n"
    assert parse_sha256sums(text) == {"some-file": digest.lower()}


def test_digests_lookup_missing_entry_raises(tmp_path: Path) -> None:
    sums = tmp_path / "SHA256SUMS"
    write_sha256sums({"present-file": "c" * 64}, sums)
    assert lookup_digest("present-file", sums) == "c" * 64
    with pytest.raises(RuntimeError, match="No SHA256 entry for absent-file"):
        lookup_digest("absent-file", sums)


def test_digests_unreadable_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unreadable"):
        read_sha256sums(tmp_path / "does-not-exist")


def test_committed_sha256sums_format() -> None:
    """The committed sums file pins upstream's four -v{N} release artifacts.

    Deliberately does NOT assert that the shared version equals
    sandbox_tools_version.txt's: on a release PR the version bumps at PR-open
    while the sums are rewritten only at post-approval upload, so a lockstep
    assertion here would keep the fast suite red for the whole review window.
    Version lockstep belongs solely to the slow-tool-tests-release CI gate.

    Upstream's own rows carry no fork-revision suffix, so `filename_to_config`
    (which requires one) cannot parse them; matched by a plain regex instead.
    """
    upstream_pattern = re.compile(
        r"^inspect-sandbox-tools-(?P<arch>amd64|arm64)"
        r"(?P<musl>-musl)?-v(?P<version>\d+)$"
    )
    entries = read_sha256sums()
    upstream_entries = {
        name: match
        for name in entries
        if (match := upstream_pattern.match(name)) is not None
    }
    assert len(upstream_entries) == 4
    assert len({m.group("version") for m in upstream_entries.values()}) == 1
    assert {
        (m.group("arch"), m.group("musl") is not None)
        for m in upstream_entries.values()
    } == {
        ("amd64", False),
        ("amd64", True),
        ("arm64", False),
        ("arm64", True),
    }


def test_committed_sha256sums_has_current_fork_revision_rows() -> None:
    """Assert every producible fork artifact name has a SHA256SUMS row.

    Every artifact name the fork's build config can produce for the
    currently pinned (version, fork revision) must have a SHA256SUMS row.

    The fork serves binaries from its own GitHub release under names carrying
    `-tl{fork_rev}`, distinct from upstream's `-v{N}` rows asserted by
    `test_committed_sha256sums_format`. If a revision bump lands without the
    corresponding publish step writing digests for it, every fresh install
    falls back to unverified downloads (fatal under
    INSPECT_SANDBOX_TOOLS_STRICT_DIGESTS) — this test catches that at commit
    time instead of at install time.
    """
    version = int(_get_sandbox_tools_version())
    fork_rev = _get_sandbox_tools_fork_revision()
    archs: tuple[SandboxToolsArch, ...] = ("amd64", "arm64")
    expected_names = {
        config_to_filename(
            SandboxToolsBuildConfig(
                arch=arch, version=version, fork_rev=fork_rev, suffix=None, musl=musl
            )
        )
        for arch in archs
        for musl in (False, True)
    }
    entries = read_sha256sums()
    missing = expected_names - entries.keys()
    assert not missing, (
        f"SHA256SUMS is missing digest rows for {sorted(missing)}; publish "
        f"v{version}-tl{fork_rev} (upload_to_github_release.py writes these "
        f"rows) and commit the rewritten SHA256SUMS."
    )
    for name in expected_names:
        config = filename_to_config(name)
        assert config.version == version
        assert config.fork_rev == fork_rev


# ---------------------------------------------------------------------------
# sandbox._download_from_s3
# ---------------------------------------------------------------------------


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


def _stream_factory(*responses: _FakeStream) -> MagicMock:
    iterator = iter(responses)
    mock = MagicMock()
    mock.side_effect = lambda method, url, **kwargs: next(iterator)
    return mock


async def test_download_from_s3_success_verifies_chmods_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"verified binary bytes" * 100
    filename = "inspect-sandbox-tools-amd64-v999"
    monkeypatch.setattr(sandbox_module, "_binaries_dir", lambda: tmp_path)
    monkeypatch.setattr(sandbox_module, "lookup_digest", lambda name: _sha256(content))

    stream_mock = _stream_factory(_FakeStream(200, content))
    with patch("inspect_ai._util.download.httpx.stream", stream_mock):
        assert await sandbox_module._download_from_s3(filename) is True

    dest = tmp_path / filename
    assert dest.read_bytes() == content
    assert dest.stat().st_mode & 0o755 == 0o755
    # atomic: no tempfiles or partials left behind
    assert [p.name for p in tmp_path.iterdir()] == [filename]


async def test_download_from_s3_mismatch_raises_and_caches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "inspect-sandbox-tools-amd64-v999"
    monkeypatch.setattr(sandbox_module, "_binaries_dir", lambda: tmp_path)
    monkeypatch.setattr(sandbox_module, "lookup_digest", lambda name: "0" * 64)

    stream_mock = _stream_factory(_FakeStream(200, b"tampered bytes"))
    with patch("inspect_ai._util.download.httpx.stream", stream_mock):
        with pytest.raises(PrerequisiteError, match="Digest verification failed"):
            await sandbox_module._download_from_s3(filename)

    assert list(tmp_path.iterdir()) == []
    stream_mock.assert_called_once()


async def test_download_from_s3_404_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "inspect-sandbox-tools-amd64-v999"
    monkeypatch.setattr(sandbox_module, "_binaries_dir", lambda: tmp_path)
    monkeypatch.setattr(sandbox_module, "lookup_digest", lambda name: "0" * 64)

    stream_mock = _stream_factory(_FakeStream(404, b""))
    with patch("inspect_ai._util.download.httpx.stream", stream_mock):
        assert await sandbox_module._download_from_s3(filename) is False

    assert list(tmp_path.iterdir()) == []


async def test_download_from_s3_missing_sums_entry_raises_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sums = tmp_path / "SHA256SUMS"
    write_sha256sums({"some-other-file": "d" * 64}, sums)
    monkeypatch.setattr(sandbox_module, "_binaries_dir", lambda: tmp_path)
    monkeypatch.setattr(
        sandbox_module, "lookup_digest", lambda name: lookup_digest(name, sums)
    )

    stream_mock = _stream_factory()
    with patch("inspect_ai._util.download.httpx.stream", stream_mock):
        with pytest.raises(PrerequisiteError, match="No SHA256 entry"):
            await sandbox_module._download_from_s3("inspect-sandbox-tools-amd64-v999")

    stream_mock.assert_not_called()
    assert [p.name for p in tmp_path.iterdir()] == ["SHA256SUMS"]


# ---------------------------------------------------------------------------
# scripts/pypi-release.py verification helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pypi_release() -> ModuleType:
    script = Path(__file__).parents[3] / "scripts" / "pypi-release.py"
    spec = importlib.util.spec_from_file_location("pypi_release", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeUrlResponse:
    def __init__(self, content: bytes):
        self._content = content
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self) -> "_FakeUrlResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self, n: int) -> bytes:
        chunk, self._content = self._content[:n], self._content[n:]
        return chunk


def test_pypi_download_file_verifies_and_lands_atomically(
    pypi_release: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"wheel-bound binary"
    dest = tmp_path / "artifact"
    monkeypatch.setattr(
        pypi_release.urllib.request,
        "urlopen",
        lambda url, timeout: _FakeUrlResponse(content),
    )

    assert pypi_release.download_file("http://x", dest, _sha256(content)) is True
    assert dest.read_bytes() == content
    assert dest.stat().st_mode & 0o755 == 0o755
    assert not (tmp_path / "artifact.partial").exists()


def test_pypi_download_file_mismatch_fails_and_writes_nothing(
    pypi_release: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "artifact"
    monkeypatch.setattr(
        pypi_release.urllib.request,
        "urlopen",
        lambda url, timeout: _FakeUrlResponse(b"tampered"),
    )

    assert pypi_release.download_file("http://x", dest, "0" * 64) is False
    assert not dest.exists()
    assert not (tmp_path / "artifact.partial").exists()


def test_pypi_check_exist_rejects_wrong_digest(
    pypi_release: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binaries = tmp_path / "src" / "inspect_ai" / "binaries"
    binaries.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    (binaries / "inspect-sandbox-tools-amd64-v9").write_bytes(b"stale")
    (binaries / "inspect-sandbox-tools-arm64-v9").write_bytes(b"stale")
    digests = {
        "inspect-sandbox-tools-amd64-v9": _sha256(b"fresh"),
        "inspect-sandbox-tools-arm64-v9": _sha256(b"fresh"),
    }
    assert pypi_release.check_sandbox_tools_exist("9", digests) is False

    (binaries / "inspect-sandbox-tools-amd64-v9").write_bytes(b"fresh")
    (binaries / "inspect-sandbox-tools-arm64-v9").write_bytes(b"fresh")
    assert pypi_release.check_sandbox_tools_exist("9", digests) is True


def test_pypi_pre_build_gate(pypi_release: ModuleType, tmp_path: Path) -> None:
    amd64, arm64 = (
        "inspect-sandbox-tools-amd64-v9",
        "inspect-sandbox-tools-arm64-v9",
    )
    digests = {amd64: _sha256(b"amd64 bytes"), arm64: _sha256(b"arm64 bytes")}

    # missing artifact
    with pytest.raises(RuntimeError, match="must contain exactly"):
        pypi_release.verify_sandbox_tools_bundle("9", digests, tmp_path)

    (tmp_path / amd64).write_bytes(b"amd64 bytes")
    (tmp_path / arm64).write_bytes(b"arm64 bytes")
    pypi_release.verify_sandbox_tools_bundle("9", digests, tmp_path)

    # extra file
    extra = tmp_path / "inspect-sandbox-tools-amd64-v8"
    extra.write_bytes(b"old")
    with pytest.raises(RuntimeError, match="must contain exactly"):
        pypi_release.verify_sandbox_tools_bundle("9", digests, tmp_path)
    extra.unlink()

    # wrong digest
    (tmp_path / amd64).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="does not match its pinned digest"):
        pypi_release.verify_sandbox_tools_bundle("9", digests, tmp_path)


def test_pypi_wheel_contents_gate(pypi_release: ModuleType, tmp_path: Path) -> None:
    required = [
        "inspect_ai/tool/_sandbox_tools_utils/SHA256SUMS",
        "inspect_ai/tool/_sandbox_tools_utils/sandbox_tools_version.txt",
        "inspect_ai/binaries/inspect-sandbox-tools-amd64-v9",
        "inspect_ai/binaries/inspect-sandbox-tools-arm64-v9",
    ]

    complete = tmp_path / "complete.whl"
    with zipfile.ZipFile(complete, "w") as wheel:
        for member in required:
            wheel.writestr(member, "content")
    pypi_release.verify_wheel_contents(complete, "9")

    incomplete = tmp_path / "incomplete.whl"
    with zipfile.ZipFile(incomplete, "w") as wheel:
        for member in required[1:]:
            wheel.writestr(member, "content")
    with pytest.raises(RuntimeError, match="SHA256SUMS"):
        pypi_release.verify_wheel_contents(incomplete, "9")
