#!/usr/bin/env python3
"""Upload sandbox tools executables to the fork's rolling GitHub release.

The fork's counterpart to upload_to_s3.py: upstream publishes to the
inspect-sandbox-tools S3 bucket and rewrites its vendored SHA256SUMS, the fork
publishes to the 'sandbox-tools' release on the fork repository (the
distribution point the in-tree default INSPECT_SANDBOX_TOOLS_BASE_URL
resolves to) and merges this fork's own digests into the same SHA256SUMS file
alongside upstream's rows, so the runtime's digest lookup (``_digests.py``)
never falls back to unverified downloads for the currently published fork
revision. Filenames carry the version and fork revision, so every
version/revision's assets coexist on the one release and the base URL never
changes.

Usage: python -m inspect_ai.tool._sandbox_tools_utils.upload_to_github_release 26
(after build_within_container.py has produced the four artifacts).
"""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_ai.tool._sandbox_tools_utils._build_config import (
        SandboxToolsArch,
        SandboxToolsBuildConfig,
        config_to_filename,
    )
    from inspect_ai.tool._sandbox_tools_utils._digests import (
        read_sha256sums,
        write_sha256sums,
    )
else:
    try:
        from ._build_config import (
            SandboxToolsArch,
            SandboxToolsBuildConfig,
            config_to_filename,
        )
        from ._digests import read_sha256sums, write_sha256sums
    except ImportError:
        from _build_config import (
            SandboxToolsArch,
            SandboxToolsBuildConfig,
            config_to_filename,
        )
        from _digests import read_sha256sums, write_sha256sums

BINARIES_DIR = Path(__file__).parent.parent.parent / "binaries"
# Overridable so CI publishes to the repository it is running in rather than a
# hardcoded one (a fork of the fork must not push assets to ours).
REPO = os.environ.get("SANDBOX_TOOLS_REPO", "trajectory-labs-pbc/inspect_ai")
RELEASE_TAG = "sandbox-tools"
ARCHS: tuple[SandboxToolsArch, ...] = ("amd64", "arm64")


def _read_fork_revision() -> int:
    revision_file = Path(__file__).parent / "sandbox_tools_fork_revision.txt"
    return int(revision_file.read_text().strip())


def _sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload sandbox tools to the fork's GitHub release"
    )
    parser.add_argument("version", type=int, help="Version number to upload")
    parser.add_argument(
        "--arch",
        choices=[*ARCHS],
        action="append",
        help="Limit to specific architecture(s); default uploads all",
    )
    args = parser.parse_args()
    archs = tuple(args.arch) if args.arch else ARCHS

    fork_rev = _read_fork_revision()
    filenames: list[str] = []
    digests: dict[str, str] = {}
    for arch in archs:
        for musl in (False, True):
            filename = config_to_filename(
                SandboxToolsBuildConfig(
                    arch=arch,
                    version=args.version,
                    fork_rev=fork_rev,
                    suffix=None,
                    musl=musl,
                )
            )
            filepath = BINARIES_DIR / filename
            if not filepath.exists():
                print(f"Error: {filepath} not found", file=sys.stderr)
                sys.exit(1)
            filenames.append(str(filepath))
            digests[filename] = _sha256_of_file(filepath)

    # Merge into the existing SHA256SUMS (which also carries upstream's own
    # -v{N} rows) rather than replacing it, so publishing our artifacts never
    # clobbers upstream's digests, and vice versa.
    entries = read_sha256sums()
    entries.update(digests)
    write_sha256sums(entries)
    print(
        f"Wrote {len(digests)} digest(s) for v{args.version}-tl{fork_rev} to SHA256SUMS"
    )

    # --clobber: re-publishing a version replaces its assets rather than failing,
    # which is what you want after a rebuild at the same version.
    subprocess.run(
        [
            "gh",
            "release",
            "upload",
            RELEASE_TAG,
            *filenames,
            "--repo",
            REPO,
            "--clobber",
        ],
        check=True,
    )
    for f in filenames:
        print(f"uploaded {Path(f).name}")


if __name__ == "__main__":
    main()
