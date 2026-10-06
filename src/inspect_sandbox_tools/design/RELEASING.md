# Releasing `inspect_sandbox_tools`

End-to-end process for shipping a new version of the sandbox tools executables.

## Overview

The sandbox tools are packaged as compressed PyInstaller `--onedir` Linux bundles for matching architecture/libc variants, not standalone static ELF files. Each archive contains the `inspect-sandbox-tools` launcher and its `_internal` runtime sidecars; injection extracts the entire tree. The upstream release process distributes these artifacts via:

1. **S3** — runtime downloads for editable/dev installs
2. **PyPI** — bundled into the `inspect_ai` wheel for pip installs

The version is a simple integer in `src/inspect_ai/tool/_sandbox_tools_utils/sandbox_tools_version.txt`.

The fork additionally pins `sandbox_tools_fork_revision.txt` and uses
`-v{VERSION}-tl{N}` artifact names. Its runtime download default is the rolling
GitHub `sandbox-tools` release configured in `sandbox.py`, overridable with
`INSPECT_SANDBOX_TOOLS_BASE_URL`; the S3 procedure below describes upstream
publication, not proof that any fork revision has been published. See
`upload_to_github_release.py` for the fork publisher. Build availability and
validation must be established separately for each architecture/libc variant.

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

For a fork-only executable-source change, keep the upstream integer and increment
`src/inspect_ai/tool/_sandbox_tools_utils/sandbox_tools_fork_revision.txt` instead.
Update the package version's matching `+tl.N` build metadata in
`src/inspect_sandbox_tools/pyproject.toml`. The build orchestrator and injection
check consume these values: fork artifacts include `-v{VERSION}-tl{N}` in their
filenames, and the executable's version RPC must report the matching metadata.
An upstream integer bump resets the fork revision to 1. Never invent digests for
unbuilt architectures or publish an earlier revision's bytes under a new name.

### 3. Build executables

Build production binaries for both architectures:

```bash
python src/inspect_ai/tool/_sandbox_tools_utils/build_within_container.py --all --dev=false
```

This creates Docker containers with PyInstaller, builds `--onedir` bundles, tars each bundle tree, and places the tar artifacts in `src/inspect_ai/binaries/`. It builds all four arch × libc variants:

- `inspect-sandbox-tools-amd64-v{VERSION}-tl{N}` (glibc)
- `inspect-sandbox-tools-arm64-v{VERSION}-tl{N}` (glibc)
- `inspect-sandbox-tools-amd64-musl-v{VERSION}-tl{N}` (musl, built via `Dockerfile.pyinstaller.musl`)
- `inspect-sandbox-tools-arm64-musl-v{VERSION}-tl{N}` (musl)

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
2. **Download** — for clean or package installs, fetches the configured distribution URL and verifies against vendored `SHA256SUMS` (normally used for unbundled musl variants)
3. **Local build** — prompts the user to build locally via Docker (`--musl` for the musl variant)

The install state detection (`_get_install_state`) determines which tiers are attempted:

- **pypi** install → expects binary in package, warns if missing
- **clean** editable install → tries the configured distribution URL
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

For embedding guidance and the distinction from model-aware Python bridges,
see [Advanced: Raw HTTP Embedding](../../../docs/agent-bridge.qmd#advanced-raw-http-embedding).

Use a compatible SDK checkout or wheel plus sandbox-tools built from compatible
sources. For a local single-variant build, replace `--all` in the build command
above with `--arch amd64` or `--arch arm64`, adding `--musl` only for a musl target.
The default build selector produces a `-dev` artifact; `--dev=false` produces
the non-dev name. Runtime `edited` installs select `-dev`, whereas `clean` and
`pypi` select non-dev. `INSPECT_SANDBOX_TOOLS_INSTALL_STATE` can override that
selection, but does not rebuild, refresh, or prove the capability of an artifact
already present locally. Do not rename stale bytes to satisfy a selector.

Record the SDK source/wheel provenance separately from the bundle's digest,
architecture, libc, integer version, fork revision, and package version returned
by the executable's `version` RPC. Injection checks the `tl.N` build metadata;
the post-bind raw-ready handshake below is still required to establish protocol
capability. Keep the launcher and all `_internal` sidecars together. Building
one variant locally neither publishes it nor validates the other variants; see
[binary integrity guidance](BINARY_INTEGRITY.md#current-artifact-and-runtime-contract).

The existing `inspect-sandbox-tools model_proxy` command uses its legacy model
routes unless `BRIDGE_MODEL_SERVICE_RAW_HTTP_VERSION=1` is explicitly set.
Any other value, including an empty value, fails startup rather than falling back.
`BRIDGE_MODEL_SERVICE_PORT` and `BRIDGE_MODEL_SERVICE_INSTANCE` retain their existing
meanings. The listener binds only to `127.0.0.1`; it uses the existing sandbox
file-service RPC, not a new sandbox-to-runner TCP connection.

Host-side remote-execution polling accepts a direct public `SandboxEnvironment`;
an embedding does not need to implement Inspect's transcript-proxy `no_events`
method. Actual `SandboxEnvironmentProxy` instances still suppress poll events
and resume recording other commands normally. Changes to this host SDK helper
require rebuilding the SDK wheel, not the sandbox executable; binary and SDK
provenance must therefore be recorded separately.

RAW mode binds the listener before calling `raw_http_ready(version=1)` and does
not process clients before the host acknowledges version 1. The host must wait
for this handshake while monitoring the process, so an older binary is not
mistaken for a RAW-capable bridge. Startup/close RPCs have 30-second deadlines;
start/read RPCs have 660-second deadlines. The embedding must configure its own
upstream timeout within that budget. A missing or incompatible ready response
fails startup.

The version-1 service methods are:

- `raw_http_ready(version)` returns `version` after the listener has bound.

- `raw_http_start(version, method, path, headers, body_b64)` returns `version`,
  an opaque `response_id`, the actual `status`, and `headers`.
- `raw_http_read(version, response_id)` returns `version`, `body_b64`, and `eof`.
  Each decoded body chunk is at most 65,536 bytes. The next chunk is requested
  only after the previous chunk has drained to the downstream connection.
- `raw_http_close(version, response_id)` returns `version` and is idempotent.

Headers are ordered `[name, value]` pairs, not dictionaries. Request duplicates
remain visible to host policy (ambiguous HTTP framing is rejected locally).
The host owns allowed methods/paths, model authorization, tenant-header policy,
and provider credentials. Requests retain their entity bytes and query string;
RAW mode does not parse model JSON, set `parallel_tool_calls`, translate API
dialects, or synthesize streaming events. Binary request/response bodies work
too. Request bodies are bounded by the existing 50 MiB limit; request trailers
are rejected because protocol v1 has no trailer field.
The host also owns secret redaction, limits, and capability policy, including
provider-side web access, remote media, code execution, and MCP. Raw transport
does not automatically apply Inspect `Model` behavior, generation configuration,
tool approval, or transcript/state tracking. Request header preservation is not
authorization to forward client credentials to a provider.

Responses preserve actual status, end-to-end header pairs, and body bytes,
including provider errors and SSE. Hop-by-hop response headers (including names
nominated by `Connection`), trailers, and upstream transfer framing are filtered
or replaced with this connection's framing. Header validation and normal
no-body status/HEAD semantics still apply. This is entity byte preservation,
not preservation of upstream TCP or HTTP chunk boundaries. HTTP upgrades and
WebSockets are not supported.
Before headers, a host transport/protocol failure is a local `raw_bridge_error`
502 (504 on timeout), not a provider response. Invalid client requests instead
receive 400 (408 on a request-read timeout). After headers, an error-bearing read
reply or transport failure closes the connection without a terminal HTTP chunk;
it never manufactures a JSON/SSE error or a success terminator.

Every known response handle is closed on completion, disconnect, cancellation,
or failure. A disconnect during `raw_http_start` retains that bounded RPC until
its handle arrives, then closes it. If the service vanishes before returning a
handle or acknowledging close, the host's trial-lease revocation owns cleanup.
RAW mode does not alter the default route builder or its model SDK behavior.

### Known legacy regression conflict

The legacy OpenAI completions, Responses, and Anthropic routes pass original
request headers to the model-service RPC alongside filtered `metadata_headers`.
Those headers can include client authorization and cookies. Existing regression
assertions in `test_model_proxy_forwards_only_configured_event_metadata_headers`
and the corresponding three cases of
`test_model_proxy_routes_metadata_separately_for_each_dialect` instead require
the raw-header RPC field to be absent. Both that production behavior and those
assertions predate the optional RAW transport; they remain unchanged here.
Consequently, these four failures must not be described as a green legacy suite
or silently removed to validate a RAW release. Legacy privacy behavior requires
a separate, explicit compatibility/security decision. RAW header authorization
and credential handling remain the responsibility of its host-side policy.

## Server control-state identity

The CLI's private `server.pid` JSON retains `pid` and `created_at` fields.
Process lookup and shutdown require an exact positive integer PID and a numeric
creation time; booleans, fractional PIDs, and nonpositive PIDs are rejected
before any process lookup or signal. Valid metadata still identifies a process
by both PID and creation time, so a reused PID does not identify the old server.
