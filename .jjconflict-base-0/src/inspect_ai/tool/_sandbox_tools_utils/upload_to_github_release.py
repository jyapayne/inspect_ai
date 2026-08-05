#!/usr/bin/env python3
"""Upload sandbox tools executables to the fork's rolling GitHub release.

The fork's counterpart to upload_to_s3.py: upstream publishes to the
inspect-sandbox-tools S3 bucket, the fork publishes to the 'sandbox-tools'
release on the fork repository (the distribution point the in-tree default
INSPECT_SANDBOX_TOOLS_BASE_URL resolves to). Filenames carry the version, so
every version's assets coexist on the one release and the base URL never
changes.

Usage: python -m inspect_ai.tool._sandbox_tools_utils.upload_to_github_release 26
(after build_within_container.py has produced the four artifacts).
"""

import argparse
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
else:
    try:
        from ._build_config import (
            SandboxToolsArch,
            SandboxToolsBuildConfig,
            config_to_filename,
        )
    except ImportError:
        from _build_config import (
            SandboxToolsArch,
            SandboxToolsBuildConfig,
            config_to_filename,
        )

BINARIES_DIR = Path(__file__).parent.parent.parent / "binaries"
# Overridable so CI publishes to the repository it is running in rather than a
# hardcoded one (a fork of the fork must not push assets to ours).
REPO = os.environ.get("SANDBOX_TOOLS_REPO", "trajectory-labs-pbc/inspect_ai")
RELEASE_TAG = "sandbox-tools"
ARCHS: tuple[SandboxToolsArch, ...] = ("amd64", "arm64")


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

    filenames: list[str] = []
    for arch in archs:
        for musl in (False, True):
            filename = config_to_filename(
                SandboxToolsBuildConfig(
                    arch=arch, version=args.version, suffix=None, musl=musl
                )
            )
            filepath = BINARIES_DIR / filename
            if not filepath.exists():
                print(f"Error: {filepath} not found", file=sys.stderr)
                sys.exit(1)
            filenames.append(str(filepath))

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
