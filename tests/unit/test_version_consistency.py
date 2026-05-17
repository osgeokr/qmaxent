"""Cross-file version consistency check.

QMaxent ships its version string in three places:

  - ``__init__.py``        (``__version__``)
  - ``metadata.txt``       (QGIS plugin manifest ``version=`` line)
  - ``CITATION.cff``       (``version:`` field)

These three MUST agree, or one of the following user-visible
inconsistencies appears:

  - The "About QMaxent" dialog shows a different version than the
    QGIS plugin manager,
  - GitHub's "Cite this repository" widget claims a different
    version than the accompanying manuscript references (§ Reference
    [31] of the manuscript cites a specific version of QMaxent),
  - Zenodo / DOI metadata drifts from the released artefact.

This test is the cheapest possible guard against that whole class
of mistake — it adds < 10 ms to CI and turns a silent drift into a
blocking red flag at pull-request time.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def versions(repo_root: Path) -> dict:
    """Extract the version literal from each of the three sources."""
    init_text = _read_text(repo_root / "__init__.py")
    m_init = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', init_text)
    assert m_init, "__init__.py: __version__ string not found"

    md_text = _read_text(repo_root / "metadata.txt")
    m_md = re.search(r'(?m)^version\s*=\s*(\S+)\s*$', md_text)
    assert m_md, "metadata.txt: version= line not found"

    cff_text = _read_text(repo_root / "CITATION.cff")
    m_cff = re.search(r'(?m)^version:\s*([0-9][\w.\-+]*)\s*$', cff_text)
    assert m_cff, "CITATION.cff: version field not found"

    return {
        "__init__.py":  m_init.group(1),
        "metadata.txt": m_md.group(1),
        "CITATION.cff": m_cff.group(1),
    }


def test_versions_match(versions):
    distinct = set(versions.values())
    assert len(distinct) == 1, (
        "Version strings disagree across files:\n  "
        + "\n  ".join(f"{src}: {v!r}" for src, v in versions.items())
    )


def test_version_is_pep440_like(versions):
    # We don't enforce full PEP 440 (we don't use pre-/post-release tags),
    # but we do require the X.Y.Z shape used by every QMaxent release
    # so the QGIS plugin repository's lexicographic version compare
    # behaves intuitively.
    v = next(iter(versions.values()))
    assert re.fullmatch(r"\d+\.\d+\.\d+", v), (
        f"Version {v!r} doesn't match expected X.Y.Z shape"
    )
