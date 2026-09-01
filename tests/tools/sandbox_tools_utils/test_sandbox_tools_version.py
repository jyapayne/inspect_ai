from pathlib import Path
from typing import Final


# release/2026-07-26 carried version 24. The value is intentionally independent
# of the sandbox-tools package semver, which can remain unchanged across rebuilds.
PREVIOUS_DATED_RELEASE_SANDBOX_TOOLS_VERSION: Final = 24


def test_sandbox_tools_version_does_not_regress_from_previous_dated_release() -> None:
    # Given: the version file packaged with inspect_ai.
    version_path = (
        Path(__file__).parents[3]
        / "src/inspect_ai/tool/_sandbox_tools_utils/sandbox_tools_version.txt"
    )

    # When: its numeric cache version is read.
    current_version = int(version_path.read_text(encoding="utf-8").strip())

    # Then: it remains compatible with the previous dated release.
    assert current_version >= PREVIOUS_DATED_RELEASE_SANDBOX_TOOLS_VERSION
