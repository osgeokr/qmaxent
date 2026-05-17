"""Unit tests for the venv path helpers in ``core/venv_manager``.

These tests defend the cross-platform path layout that the Windows /
Linux installation logic depends on. The manuscript (§ 2.1) calls out
the Windows-specific quirk that ``sys.executable`` is ``qgis-bin.exe``
rather than ``python.exe`` — the fix is to derive the interpreter
from ``sys.prefix`` instead. Path helpers are the boundary at which
that fix is observable in pure-Python form (no subprocess needed),
so they are testable here without booting QGIS.
"""

from __future__ import annotations

import os
import sys

from core import venv_manager as vm


def test_venv_python_path_layout_matches_platform(tmp_path):
    fake_venv = str(tmp_path / "venv_py3.12")
    p = vm.get_venv_python_path(fake_venv)
    if sys.platform == "win32":
        assert p.endswith(os.path.join("Scripts", "python.exe"))
    else:
        assert p.endswith(os.path.join("bin", "python3"))


def test_venv_pip_path_layout_matches_platform(tmp_path):
    fake_venv = str(tmp_path / "venv_py3.12")
    p = vm.get_venv_pip_path(fake_venv)
    if sys.platform == "win32":
        assert p.endswith(os.path.join("Scripts", "pip.exe"))
    else:
        assert p.endswith(os.path.join("bin", "pip"))


def test_venv_site_packages_layout_matches_platform(tmp_path):
    fake_venv = str(tmp_path / "venv_py3.12")
    sp = vm.get_venv_site_packages(fake_venv)
    # On Windows the path ends ``Lib/site-packages``; on POSIX it ends
    # ``lib/pythonX.Y/site-packages``. We accept either of the two
    # versioned subdirs (e.g. python3.10, python3.12) so this test
    # stays green across the CI matrix.
    if sys.platform == "win32":
        assert sp.endswith(os.path.join("Lib", "site-packages"))
    else:
        assert "site-packages" in sp
        assert "python" in os.path.basename(os.path.dirname(sp))


def test_venv_exists_false_for_nonexistent_dir(tmp_path):
    """venv_exists() is the first guard in many entry points. It must
    not raise when handed a path that does not exist (the default
    state on a fresh install)."""
    nope = str(tmp_path / "definitely_not_here")
    assert vm.venv_exists(nope) is False


def test_required_packages_metadata_shape():
    """REQUIRED_PACKAGES is read both by pip (specifier strings) and by
    the verification pass (import-only). Each entry must be a 2-tuple
    of strings, with the version part being a valid PEP-440 specifier
    (we approximate with: starts with one of >, <, =, !, ~, or empty)."""
    assert vm.REQUIRED_PACKAGES, "REQUIRED_PACKAGES is empty"
    for entry in vm.REQUIRED_PACKAGES:
        assert isinstance(entry, tuple) and len(entry) == 2
        name, spec = entry
        assert isinstance(name, str) and name
        assert isinstance(spec, str)
        if spec:
            assert spec[0] in "><=~!", f"Package {name!r} has unusual version spec {spec!r}"


def test_required_packages_include_runtime_critical():
    """The packages cited in the manuscript (elapid, rasterio, geopandas,
    scikit-learn, NumPy) MUST be in the install list — otherwise a
    first-run install will produce a half-working plugin that fails
    far away from the dependency manager."""
    names = {n for n, _ in vm.REQUIRED_PACKAGES}
    missing = {"elapid", "rasterio", "geopandas", "scikit-learn", "numpy"} - names
    assert not missing, f"Critical runtime dependencies missing: {missing}"
