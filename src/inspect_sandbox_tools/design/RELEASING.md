# Releasing `inspect_sandbox_tools`

End-to-end process for shipping a new version of the sandbox tools executables.

## Overview

The sandbox tools are compressed PyInstaller `--onedir` Linux bundles for matching architecture/libc variants (amd64/arm64 × glibc/musl), not standalone static ELF files. Each contains a launcher and its `_internal` runtime sidecars; injection must preserve the entire extracted tree. They are distributed via:

1. **S3** — runtime downloads for editable/dev installs
2. **PyPI** — bundled into the `inspect_ai` wheel for pip installs

The version is a simple integer in `src/inspect_ai/tool/_sandbox_tools_utils/sandbox_tools_version.txt`.

The fork additionally selects `sandbox_tools_fork_revision.txt` and `-v{VERSION}-tl{N}` artifact names. Its default runtime distribution is the rolling GitHub `sandbox-tools` release configured in `sandbox.py`; `INSPECT_SANDBOX_TOOLS_BASE_URL` can override that URL. The S3 commands below describe the upstream publication flow. Never overwrite an existing published asset with different bytes or relabel old bytes as a new build.

## Release Steps

Steps 3–6 are scripted as an interactive wizard — run it from a checkout of
the PR's head branch (`--dry-run` builds and validates without publishing;
`--auto` answers every prompt with its safe default, for agents/unattended
runs):

```sh
scripts/release-sandbox-tools.sh [--dry-run] [--auto]
```

The sections below document the underlying commands.

### 1. Make code changes

Edit source code under `src/inspect_sandbox_tools/`.

### 2. Bump the version

Increment the integer in:

```
src/inspect_ai/tool/_sandbox_tools_utils/sandbox_tools_version.txt
```

### 3. Build executables

Build production binaries for both architectures:

```bash
python src/inspect_ai/tool/_sandbox_tools_utils/build_within_container.py --all --dev=false
```

This creates Docker containers with PyInstaller, builds `--onedir` bundles, tars each bundle tree, and places the tar artifacts in `src/inspect_ai/binaries/`. It builds all four arch × libc variants:

- `inspect-sandbox-tools-amd64-v{VERSION}` (glibc)
- `inspect-sandbox-tools-arm64-v{VERSION}` (glibc)
- `inspect-sandbox-tools-amd64-musl-v{VERSION}` (musl, built via `Dockerfile.pyinstaller.musl`)
- `inspect-sandbox-tools-arm64-musl-v{VERSION}` (musl)

Requires Docker with multi-architecture support. (Build a single variant with `--arch {amd64,arm64} [--musl]`.)

### 4. Validate across distros

```bash
python -m inspect_ai.tool._sandbox_tools_utils.validate_distros
```

Routes each artifact to the matching distro set: glibc variants across Ubuntu/Debian/Kali, the musl variant across Alpine.

### 5. Upload to S3

```bash
python src/inspect_ai/tool/_sandbox_tools_utils/upload_to_s3.py {VERSION}
```

Uploads all four artifacts (amd64/arm64 × glibc/musl) to the `inspect-sandbox-tools` bucket (us-east-2) with public-read ACL so runtime S3 downloads work without credentials.

The script refuses to run unless `{VERSION}` matches the committed `sandbox_tools_version.txt`, refuses to overwrite a published version with different bytes (published S3 objects are immutable — if a published object is wrong, bump the version and publish fresh artifacts; never fix a problem by re-uploading), rewrites `src/inspect_ai/tool/_sandbox_tools_utils/SHA256SUMS` with the digests of the four local artifacts before uploading, and round-trip verifies each uploaded object.

**URL pattern:** `https://inspect-sandbox-tools.s3.us-east-2.amazonaws.com/inspect-sandbox-tools-{arch}[-musl]-v{version}`

### 6. Commit the digest file

Commit the rewritten `SHA256SUMS` to the release PR branch (the upload script prints the exact commands). The digests must land in the same PR as the version bump: every consumer (runtime S3 download, `pypi-release.py`, the `slow-tool-tests-release` CI gate) verifies fetched bytes against them, and the release gate stays red until both the upload completes and the sums are pushed.

Fail-closed window: while a version bump is merged (or installed non-editably from the PR branch) with the sums not yet rewritten, clean/pypi runtime downloads treat the missing sums entry as a fatal integrity failure. Once the rewritten sums are committed but before the S3 objects land, downloads 404 and fall back to the local-build prompt; the upload-before-commit ordering makes that window hard to reach. After upload, a digest mismatch anywhere is a stop-the-line signal — investigate; never re-upload over a published object.

### 7. Merge the PR

Once `slow-tool-tests-release` is green, merge through the normal review process.

### 8. PyPI release (when releasing inspect_ai)

The `inspect_ai` release script automatically pulls sandbox tools from S3:

```bash
python scripts/pypi-release.py release v{INSPECT_AI_VERSION}
```

This downloads the **glibc** binaries from S3 into `src/inspect_ai/binaries/`, bundles them into the wheel as package data, and publishes to PyPI. The musl variants are intentionally **not** bundled — they live on S3 and are fetched at runtime when a musl sandbox is detected (keeps the wheel small for the common case).

## How binaries are resolved at runtime

Injection first detects the sandbox's arch and libc (`recon.detect_sandbox_os` →
`architecture` + `libc`) and resolves the matching artifact name (adding `-musl` for
musl sandboxes). `src/inspect_ai/tool/_sandbox_tools_utils/sandbox.py` then uses a
three-tier fallback:

1. **Local** — looks for the executable in `inspect_ai/binaries/` (only the glibc variants are bundled into the wheel)
2. **S3 download** — if the install is "clean" (no local edits to sandbox tools), downloads from S3, verified against the vendored `SHA256SUMS` (this is the normal path for the musl variants)
3. **Local build** — prompts the user to build locally via Docker (`--musl` for the musl variant)

The install state detection (`_get_install_state`) determines which tiers are attempted:

- **pypi** install → expects binary in package, warns if missing
- **clean** editable install → tries S3 download
- **edited** editable install → builds a `-dev` suffixed binary locally

## Key files

| File | Purpose |
|------|---------|
| `src/inspect_ai/tool/_sandbox_tools_utils/sandbox_tools_version.txt` | Version (simple integer) |
| `src/inspect_ai/tool/_sandbox_tools_utils/SHA256SUMS` | Pinned digests for the four published artifacts (written by `upload_to_s3.py`; see `BINARY_INTEGRITY.md`) |
| `src/inspect_ai/tool/_sandbox_tools_utils/build_within_container.py` | Build orchestrator |
| `src/inspect_ai/tool/_sandbox_tools_utils/build_executable.py` | Runs inside Docker container |
| `src/inspect_ai/tool/_sandbox_tools_utils/validate_distros.py` | Cross-distro validation |
| `src/inspect_ai/tool/_sandbox_tools_utils/sandbox.py` | Runtime resolution and injection |
| `scripts/pypi-release.py` | PyPI release script (downloads from S3) |

## Optional RAW HTTP model bridge

The public `inspect_ai.agent.sandbox_model_proxy(sandbox, methods=handlers, raw_http=True)` API reuses the released bridge service, without an Inspect Model or provider loop. Default model-aware bridges are unchanged. Raw mode requires exactly the following protocol-v1 handlers:

- `raw_http_ready(version)` returns `{"version": 1}` after the listener binds.
- `raw_http_start(version, method, path, headers, body_b64)` returns `version`, an opaque `response_id`, the actual HTTP `status`, and `headers`.
- `raw_http_read(version, response_id)` returns `version`, `body_b64`, and boolean `eof`. Each decoded chunk is at most 65,536 bytes; the next read follows downstream drain.
- `raw_http_close(version, response_id)` returns `version` and must be idempotent.

Every successful RPC response carries integer `version=1`; an `error` field signals protocol/transport failure. Headers are ordered `[name, value]` pairs. Request entity bytes and the path/query are preserved; model JSON is not parsed. Request bodies retain the 50 MiB limit, ambiguous framing is rejected, and request trailers are unsupported. Response entity bytes, status and end-to-end header pairs (including duplicates) are preserved. Hop-by-hop headers, connection-nominated headers, trailers and transfer framing are removed or rebuilt; this is not preservation of upstream TCP/HTTP chunk boundaries. HEAD and no-body statuses retain their HTTP semantics. HTTP upgrades and WebSockets are unsupported.

Before headers, transport failures return a local `raw_bridge_error` 502 (504 on timeout); malformed client requests return 400 (408 on timeout). After headers, a failed read closes without a terminal HTTP chunk, never fabricating a JSON/SSE error or success terminator. Known handles are closed on completion, disconnect, cancellation and failure. A disconnected start request retains its bounded RPC until its handle can be closed; host lease revocation must own cleanup when the service disappears or a handle is lost.

The CLI uses legacy routes unless `BRIDGE_MODEL_SERVICE_RAW_HTTP_VERSION=1` is explicitly set. Any other value, including empty, fails startup. The public SDK first runs `model_proxy --capabilities` against the injected launcher and requires integer `raw_http_version=1`. It then starts the same model-proxy process with explicit protocol negotiation and waits up to 30 seconds for its bound-loopback ready handshake. The binary accepts no requests until that handshake succeeds. Ready/close RPC deadlines are 30 seconds; start/read deadlines are 660 seconds, so host upstream timeouts must fit within that bound. The SDK fails if the raw proxy exits, even with exit code zero.

The host owns authorization, model allow-lists, provider credentials, tenant-header policy, redaction, limits and capabilities, including remote media, web, code execution and MCP. Raw mode does not mint Inspect proposal grants or apply Inspect approval/state semantics; `bridged_tools` and generate/tool handlers are refused. Hosts exposing MCP via raw routes must own their trusted execution grants.

### Build and provenance gates

An aligned SDK source/wheel and a compatible newly built bundle are separate prerequisites. Source version metadata stays authoritative; a version label is not evidence of raw capability. The existing injection path retains architecture/libc selection, framework-directory ownership checks, fork-revision checks and digest-verified downloads. The new capability query and live handshake reject an old binary even when its package-version label matches. They do not replace artifact provenance or make an agent-writable tools directory trusted.

Build from the final recorded source using the normal Docker/PyInstaller build orchestrator after source freeze. A local single-variant build uses `--arch amd64` or `--arch arm64`, with `--musl` only for musl. The default selector builds `-dev`; edited installs select that name, whereas clean/package installs select non-dev. Never rename stale bytes to satisfy the selector, suppress dependencies, or invent digest entries. Preserve the entire extracted tree. Record the actual source commit, source-derived wheel version, wheel digest, bundle digest, architecture/libc and executable version independently. Publication requires the normal immutable asset/digest release procedure; building one local variant is not publication or coverage of the other three.

Before handoff, verify the installed public SDK/default Inspect bridge, explicit external-sandbox MCP and bare/proxied remote exec. Run `tests/agent/test_sandbox_model_proxy.py`, `tests/agent/test_monitor_proxy.py`, `tests/tools/test_mcp_tools.py` and `tests/util/sandbox/test_exec_remote.py` with the normally resolved SDK. The sandbox-tools tests `tests/agent_bridge/test_raw_proxy.py` and `test_raw_http_consumer.py` cover raw transport, real SSE bytes, provider error statuses, header/body preservation, disconnect/backpressure and legacy mode. For an actual bundle smoke, set `INSPECT_RAW_HTTP_LAUNCHER` to its extracted launcher and `INSPECT_RAW_HTTP_IMAGE` to a matching Linux image with no `python`/`python3`; the consumer fixture keeps its mock gateway on the host. These are verification instructions, not claims that any gate has run on the aligned release.
