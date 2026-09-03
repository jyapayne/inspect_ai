"""Guard: the committed viewer dist must be one coherent build.

``src/inspect_ai/_view/dist`` is a build artifact: a single rolldown build
emits the whole chunk graph. Release cuts octopus-merge member branches, and
git resolves the dist files per-file, so two members carrying dists from
different builds can be spliced into one tree. If the spliced builds pin
different React versions, react-dom throws React error #527 at module scope
and the viewer -- and every ``inspect view bundle`` output -- renders a blank
page with no console error. This shipped at release/2026-08-11.2 (5551b122):
index.js carried react-dom@19.2.7 while jsx-runtime.js carried react@19.2.8.

Extraction reads the two packages' own runtime version stamps rather than
build-tool debug metadata (pnpm store paths in region comments), which a
production-minification build is not obliged to preserve and, as of the
viewer-pin/inspect-ai-fork-glue bump, no longer does:

- react-dom always compiles in a DevTools hook registration object literal
  (``{bundleType, version, rendererPackageName, ...}``) naming itself
  ``react-dom`` next to its own version -- present since React's DevTools
  hook protocol was introduced and independent of minifier settings.
- react-dom also always compiles in its React error #527 peer-version guard
  (``if(actual!==`V`)throw Error(...(527,actual,`V`))``), comparing the
  react instance it actually loaded against the version it expects -- the
  literal ``V`` is the same react-dom-embedded expectation as the hook's.
- react itself has no comparable hook registration (that is renderer-only),
  so it is identified by its own version stamp immediately following the
  ``useTransition`` hook export in its public API object -- ``useTransition``
  is a real exported hook name, not a local identifier, so it survives
  minification/renaming.
"""

import re
from pathlib import Path

import inspect_ai._view

_REACT_DOM_HOOK = re.compile(
    r"bundleType:\d+,version:`([^`]+)`,rendererPackageName:`react-dom`"
)
_REACT_DOM_GUARD = re.compile(r"throw Error\(\w+\(527,\s*\w+,\s*`([^`]+)`")
_REACT_VERSION = re.compile(r"useTransition\(\)\}[),]\s*[\w$]+\.version=`([^`]+)`")


def _react_versions(assets_dir: Path) -> dict[str, dict[str, set[str]]]:
    """Map package name -> version -> chunk filenames embedding that version."""
    versions: dict[str, dict[str, set[str]]] = {}
    for asset in sorted(assets_dir.glob("*.js")):
        text = asset.read_text(encoding="utf-8")
        for pattern, package in (
            (_REACT_DOM_HOOK, "react-dom"),
            (_REACT_DOM_GUARD, "react-dom"),
            (_REACT_VERSION, "react"),
        ):
            for version in pattern.findall(text):
                versions.setdefault(package, {}).setdefault(version, set()).add(
                    asset.name
                )
    return versions


def test_dist_chunks_share_one_react_version() -> None:
    view_file = inspect_ai._view.__file__
    assert view_file is not None
    assets_dir = Path(view_file).parent / "dist" / "assets"
    versions = _react_versions(assets_dir)

    assert versions.get("react") and versions.get("react-dom"), (
        f"no react/react-dom markers found under {assets_dir} -- dist assets "
        "are missing, are LFS pointers that were not smudged, or the build "
        "stopped embedding the react-dom hook/guard and react useTransition "
        "markers (update this guard's extraction)"
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
