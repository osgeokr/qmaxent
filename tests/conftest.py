"""Shared pytest configuration and fixtures for the QMaxent test suite.

This conftest deliberately solves one cross-cutting problem: the
plugin source imports ``qgis`` and ``PyQt5`` at package load time, but
the unit tier MUST be runnable from a plain Python install (so CI
matrix entries on Windows / Linux / macOS Python 3.10 / 3.11 / 3.12
all stay green without installing the QGIS host). The unit tests
target pure-Python modules under ``core/`` and ``bridge/`` that do
NOT need QGIS at runtime — only their package ancestors import it.

We install lightweight stubs for ``qgis.*`` and ``PyQt5.*`` into
``sys.modules`` BEFORE pytest collects any test that imports the
plugin. This is the standard pattern (matches what
qgis-plugin-ci, plugin_reloader, pb_tool, and several CI templates
do): real QGIS is not available in the matrix runner, so we stub the
attribute surface our pure-Python modules actually touch.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Path setup: make the plugin importable as a package.
# ---------------------------------------------------------------------------
# QGIS loads plugins by name from the plugins directory, so the source
# tree itself doesn't include a ``qmaxent/`` parent. For tests we
# expose the repository root one level up on sys.path so ``from core
# import ...`` and ``from bridge import ...`` work the same way they
# do inside the running plugin.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# QGIS / Qt stubs.
# ---------------------------------------------------------------------------
def _install_qgis_stubs() -> None:
    """Plant a minimal ``qgis`` / ``PyQt5`` shim into ``sys.modules``.

    We only stub the symbols that the pure-Python core / bridge layers
    reference at import time (``QgsMessageLog``, ``Qgis``,
    ``QgsSettings``). Anything that needs the *real* QGIS / Qt
    should live behind a lazy ``import`` and be marked as a UI test
    that runs separately (not in this matrix).
    """
    if "qgis" in sys.modules:
        return

    qgis = types.ModuleType("qgis")
    qgis_core = types.ModuleType("qgis.core")

    class _FakeQgis:  # noqa: D401 — naming mirrors qgis.core.Qgis
        """Stand-in for ``qgis.core.Qgis`` log-level enum."""
        Info = 0
        Warning = 1
        Critical = 2
        Success = 3

    class _FakeQgsMessageLog:
        @staticmethod
        def logMessage(*_args, **_kwargs):
            # No-op: tests run far from QGIS's Log Messages panel.
            return None

    class _FakeQgsSettings:
        """Tiny dict-backed stand-in for ``QSettings``-style access."""

        def __init__(self):
            self._data = {}

        def value(self, key, default=None, type=None):  # noqa: A002
            v = self._data.get(key, default)
            if type is bool and isinstance(v, str):
                return v.lower() in ("1", "true", "yes")
            return v

        def setValue(self, key, value):
            self._data[key] = value

    qgis_core.Qgis = _FakeQgis
    qgis_core.QgsMessageLog = _FakeQgsMessageLog
    qgis_core.QgsSettings = _FakeQgsSettings

    sys.modules["qgis"] = qgis
    sys.modules["qgis.core"] = qgis_core
    qgis.core = qgis_core  # so `import qgis; qgis.core.Qgis` works too


_install_qgis_stubs()


# ---------------------------------------------------------------------------
# Common pytest fixtures.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to ``tests/fixtures/``."""
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def tmp_outdir(tmp_path: Path) -> Path:
    """A clean per-test directory for raster / GeoPackage outputs."""
    out = tmp_path / "out"
    out.mkdir()
    return out


@pytest.fixture(scope="session")
def fixed_seed() -> int:
    """The plugin-wide default seed (mirrors the dock's default of 42)."""
    return 42


# ---------------------------------------------------------------------------
# Optional-dependency gating.
# ---------------------------------------------------------------------------
def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


# These markers let the regression tier skip cleanly on CI matrix rows
# where the (large) optional dependency isn't installed, instead of
# failing with a confusing ImportError. Skip-with-reason is the canonical
# pytest way to communicate "this row of the matrix isn't responsible
# for this claim" — the green/red grid in PRs stays meaningful.
def pytest_collection_modifyitems(config, items):
    skip_no_elapid = pytest.mark.skip(reason="elapid not installed")
    skip_no_rio    = pytest.mark.skip(reason="rasterio not installed")
    skip_no_sklearn = pytest.mark.skip(reason="scikit-learn not installed")

    has_elapid = _has("elapid")
    has_rio    = _has("rasterio")
    has_sklearn = _has("sklearn")

    for item in items:
        if "needs_elapid" in item.keywords and not has_elapid:
            item.add_marker(skip_no_elapid)
        if "needs_rasterio" in item.keywords and not has_rio:
            item.add_marker(skip_no_rio)
        if "needs_sklearn" in item.keywords and not has_sklearn:
            item.add_marker(skip_no_sklearn)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_elapid: requires the elapid Python package",
    )
    config.addinivalue_line(
        "markers", "needs_rasterio: requires the rasterio Python package",
    )
    config.addinivalue_line(
        "markers", "needs_sklearn: requires the scikit-learn Python package",
    )
