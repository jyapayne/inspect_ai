"""Guard: the committed viewer dist must be one coherent build.

``src/inspect_ai/_view/dist`` is a build artifact: a single rolldown build
emits the whole chunk graph. Release cuts octopus-merge member branches, and
git resolves the dist files per-file, so two members carrying dists from
different builds can be spliced into one tree. If the spliced builds pin
different React versions, react-dom throws React error #527 at module scope
and the viewer -- and every ``inspect view bundle`` output -- renders a blank
page with no console error. This shipped at release/2026-08-11.2 (5551b122):
index.js carried react-dom@19.2.7 while jsx-runtime.js carried react@19.2.8.

Rolldown output preserves pnpm store paths (``.pnpm/react@X.Y.Z``) in region
comments, identifying the package versions baked into each chunk. Assert the
chunk graph resolves exactly one version of react and of react-dom, and that
the two are identical (react-dom enforces exact equality at runtime).
"""

import re
from pathlib import Path

import inspect_ai._view

_REACT_PACKAGE = re.compile(r"\.pnpm/(react|react-dom)@(\d+\.\d+\.\d+)")


def _react_versions(assets_dir: Path) -> dict[str, dict[str, set[str]]]:
    """Map package name -> version -> chunk filenames embedding that version."""
    versions: dict[str, dict[str, set[str]]] = {}
    for asset in sorted(assets_dir.glob("*.js")):
        text = asset.read_text(encoding="utf-8")
        for found in _REACT_PACKAGE.finditer(text):
            package, version = found.group(1), found.group(2)
            versions.setdefault(package, {}).setdefault(version, set()).add(asset.name)
    return versions


def test_dist_chunks_share_one_react_version() -> None:
    view_file = inspect_ai._view.__file__
    assert view_file is not None
    assets_dir = Path(view_file).parent / "dist" / "assets"
    versions = _react_versions(assets_dir)

    assert versions.get("react") and versions.get("react-dom"), (
        f"no react/react-dom markers found under {assets_dir} -- dist assets "
        "are missing, are LFS pointers that were not smudged, or the build "
        "stopped embedding pnpm store paths (update this guard's extraction)"
    )

    resolved: dict[str, str] = {}
    for package, by_version in versions.items():
        mixed = {v: sorted(files) for v, files in by_version.items()}
        assert len(by_version) == 1, (
            f"dist chunk graph mixes {package} versions, so it was spliced "
            f"from more than one build (per-file merge of dist?): {mixed}"
        )
        resolved[package] = next(iter(by_version))

    assert resolved["react"] == resolved["react-dom"], (
        "react and react-dom versions differ; react-dom enforces exact "
        f"equality at runtime (React error #527): {resolved}"
    )
