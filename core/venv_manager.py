"""Virtual environment manager for QMaxent QGIS Plugin.

Manages creation and verification of an isolated Python environment
for QMaxent's dependencies (elapid, rasterio, geopandas, etc.).

Adapted from GeoAI plugin's venv_manager.py (TerraLabAI team).
Key fix vs naive approach: sys.executable on Windows QGIS is qgis-bin.exe,
NOT python.exe.  Calling subprocess([sys.executable, "-m", "venv", ...])
launches another QGIS instance which treats "-m", "venv", VENV_DIR as
data source paths — producing the "invalid data source: -m / venv" errors.
We resolve the real Python interpreter explicitly via sys.prefix.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, List, Optional, Tuple

from qgis.core import Qgis, QgsMessageLog, QgsSettings

PYTHON_VERSION = f"py{sys.version_info.major}.{sys.version_info.minor}"
CACHE_DIR = (
    os.environ.get("QMAXENT_CACHE_DIR")
    or os.path.expanduser("~/.qgis_qmaxent")
)
VENV_DIR = os.path.join(CACHE_DIR, f"venv_{PYTHON_VERSION}")
DEPS_HASH_FILE = os.path.join(VENV_DIR, "deps_hash.txt")

_INSTALL_LOGIC_VERSION = "2"

# Compatible-release version ranges: every package has both an explicit
# floor (lowest tested) and an explicit ceiling at the next major. This
# keeps patch and minor releases (incl. security fixes) flowing in while
# blocking ABI/API breaks at major-version boundaries — the specific
# class of failure that produced the "init rasterio._base" cython
# ImportError after Remove → Install on Windows.
#
# Why not pin everything with `==`? Because (a) a fixed version may not
# have a wheel for every OS/Python combination users run (Py 3.13 is
# coming), (b) we'd block our users from receiving security patches
# until we cut a new release, and (c) some users mix their own data
# pipelines with QMaxent's venv and benefit from minor compatibility.
# A "verified lock" mode for paper-reproducibility may be added later
# as an opt-in, once the test suite confirms a known-good combination.
REQUIRED_PACKAGES: List[Tuple[str, str]] = [
    ("numpy",        ">=2.0,<3"),
    ("pandas",       ">=2.0,<3"),
    ("pyproj",       ">=3.6,<4"),
    ("scipy",        ">=1.13,<2"),
    ("scikit-learn", ">=1.5,<1.6"),
    ("matplotlib",   ">=3.7,<4"),
    # rasterio 1.4 is the first release line that ships NumPy-2-ABI
    # wheels for Python 3.12 on Windows/macOS/Linux. Older 1.3.x wheels
    # are NumPy-1-ABI only, which produces an "init rasterio._base"
    # ImportError on pip cache reuse against numpy 2.x. Pin >=1.4 so
    # pip never resolves a NumPy-1-ABI wheel into a NumPy-2 venv.
    ("rasterio",     ">=1.4,<2"),
    ("rtree",        ">=1.0,<2"),
    ("geopandas",    ">=1.0,<2"),
    ("tqdm",         ">=4.60,<5"),
    # openpyxl ships the styled XLSX results writer (multi-sheet
    # academic-paper-style tables). >=3.1 picks up the Border/Side
    # API used in _save_xlsx; the major version has been stable since
    # 2.x so a <4 ceiling is conservative.
    ("openpyxl",     ">=3.1,<4"),
    ("elapid",       ">=1.0.3,<2.0"),
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(message: str, level=Qgis.Info):
    QgsMessageLog.logMessage(str(message), "QMaxent", level=level)


# ---------------------------------------------------------------------------
# Dependency hash
# ---------------------------------------------------------------------------

def _compute_deps_hash() -> str:
    data = repr(REQUIRED_PACKAGES).encode("utf-8")
    data += _INSTALL_LOGIC_VERSION.encode("utf-8")
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _read_deps_hash() -> Optional[str]:
    try:
        with open(DEPS_HASH_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, IOError):
        return None


def _write_deps_hash():
    try:
        os.makedirs(os.path.dirname(DEPS_HASH_FILE), exist_ok=True)
        with open(DEPS_HASH_FILE, "w", encoding="utf-8") as f:
            f.write(_compute_deps_hash())
    except (OSError, IOError) as e:
        _log(f"Failed to write deps hash: {e}", Qgis.Warning)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _get_qgis_proxy_settings() -> Optional[str]:
    try:
        from urllib.parse import quote as url_quote
        settings = QgsSettings()
        if not settings.value("proxy/proxyEnabled", False, type=bool):
            return None
        host = settings.value("proxy/proxyHost", "", type=str)
        if not host:
            return None
        port     = settings.value("proxy/proxyPort",     "", type=str)
        user     = settings.value("proxy/proxyUser",     "", type=str)
        password = settings.value("proxy/proxyPassword", "", type=str)
        url = "http://"
        if user:
            url += url_quote(user, safe="")
            if password:
                url += ":" + url_quote(password, safe="")
            url += "@"
        url += host
        if port:
            url += f":{port}"
        return url
    except Exception as e:
        _log(f"Could not read proxy settings: {e}", Qgis.Warning)
        return None


def _get_subprocess_kwargs() -> dict:
    """Platform-specific kwargs. Sets cwd=CACHE_DIR so subprocess args are
    not confused with QGIS data source paths in the working directory."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    kwargs: dict = {"cwd": CACHE_DIR}
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = si
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def _get_clean_env_for_pip() -> dict:
    """Environment for running pip INSIDE the venv.

    Strips QGIS-specific paths so the venv pip uses only its own libraries.
    The venv's own python.exe does not need PYTHONHOME to locate its stdlib.
    """
    env = os.environ.copy()
    for var in (
        "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
        "QGIS_PREFIX_PATH", "QGIS_PLUGINPATH",
        "PROJ_DATA", "PROJ_LIB", "GDAL_DATA", "GDAL_DRIVER_PATH",
    ):
        env.pop(var, None)
    env["PYTHONIOENCODING"] = "utf-8"
    proxy = _get_qgis_proxy_settings()
    if proxy:
        env.setdefault("HTTP_PROXY", proxy)
        env.setdefault("HTTPS_PROXY", proxy)
    return env


def _get_env_for_verify(site_packages: str) -> dict:
    """Environment for running ``import pkg`` inside the venv.

    Differs from ``_get_clean_env_for_pip`` in that PROJ_DATA / PROJ_LIB /
    GDAL_DATA are populated to point at the venv's bundled native data
    directories. Without this, rasterio / pyproj / pyogrio can fail their
    cython init step on Windows because the parent QGIS process's
    PROJ_DATA was stripped but no replacement was provided. The bundled
    wheel data dirs are usually findable by rasterio itself, but being
    explicit costs us nothing and removes a class of cryptic
    "init rasterio._base" failures from the verification pass.
    """
    env = _get_clean_env_for_pip()
    proj_candidates = [
        os.path.join(site_packages, "pyproj",   "proj_dir", "share", "proj"),
        os.path.join(site_packages, "rasterio", "proj_data"),
        os.path.join(site_packages, "pyogrio",  "proj_data"),
    ]
    for c in proj_candidates:
        if os.path.isfile(os.path.join(c, "proj.db")):
            env["PROJ_DATA"] = c
            env["PROJ_LIB"]  = c
            break
    gdal_candidates = [
        os.path.join(site_packages, "rasterio", "gdal_data"),
        os.path.join(site_packages, "pyogrio",  "gdal_data"),
    ]
    for c in gdal_candidates:
        if os.path.isdir(c):
            env["GDAL_DATA"] = c
            break
    return env


# ---------------------------------------------------------------------------
# Python executable resolution
# ---------------------------------------------------------------------------

def _verify_python(path: str) -> Optional[str]:
    """Quick sanity-check: run python --version and return path if OK."""
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run(
            [path, "-c", "import sys; print(sys.version)"],
            capture_output=True, text=True, timeout=15,
            env=env, **_get_subprocess_kwargs(),
        )
        if r.returncode == 0:
            _log(f"Python OK ({path}): {r.stdout.strip()[:60]}", Qgis.Info)
            return path
        _log(f"Python failed ({path}): {r.stderr.strip()[:120]}", Qgis.Warning)
    except Exception as e:
        _log(f"Python check error ({path}): {e}", Qgis.Warning)
    return None


def _get_qgis_python_windows() -> Optional[str]:
    """Find QGIS's bundled python.exe on Windows.

    On Windows, sys.executable is the QGIS application binary, not Python.
    sys.prefix points to the Python apps directory, e.g.:
      C:\\Program Files\\QGIS 3.36\\apps\\Python312
      C:\\OSGeo4W\\apps\\Python312

    We look for python.exe there, then fall back to the parent directory.
    This mirrors GeoAI's _get_qgis_python() pattern exactly.
    """
    if sys.platform != "win32":
        return None

    candidates = [
        os.path.join(sys.prefix, "python.exe"),
        os.path.join(sys.prefix, "python3.exe"),
    ]
    # OSGeo4W puts python.exe one level up (e.g. C:\OSGeo4W\bin\python3.exe)
    parent = os.path.dirname(sys.prefix)
    candidates += [
        os.path.join(parent, "python.exe"),
        os.path.join(parent, "python3.exe"),
        os.path.join(parent, "bin", "python.exe"),
        os.path.join(parent, "bin", "python3.exe"),
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            result = _verify_python(candidate)
            if result:
                return result

    _log(
        f"python.exe not found near sys.prefix ({sys.prefix}). "
        "Candidates tried: " + ", ".join(candidates),
        Qgis.Warning,
    )
    return None


def _get_linux_system_python() -> Optional[str]:
    """Find a usable python3 on Linux that includes the venv module."""
    if sys.platform in ("win32", "darwin"):
        return None

    candidates: List[str] = []

    # sys.executable on Linux QGIS is usually /usr/bin/python3 — try it first
    if sys.executable and os.path.isfile(sys.executable):
        real = os.path.realpath(sys.executable)
        if real not in candidates:
            candidates.append(real)

    import shutil as _shutil
    for name in ("python3", "python"):
        p = _shutil.which(name)
        if p and p not in candidates:
            candidates.append(p)

    for candidate in candidates:
        try:
            r = subprocess.run(
                [candidate, "-c",
                 "import sys, venv; "
                 "print(sys.version_info.major, sys.version_info.minor)"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                parts = r.stdout.strip().split()
                if len(parts) == 2:
                    major, minor = int(parts[0]), int(parts[1])
                    if (major, minor) >= (3, 9):
                        _log(f"System Python: {candidate} "
                             f"({major}.{minor})", Qgis.Info)
                        return candidate
            elif "No module named" in r.stderr and "venv" in r.stderr:
                _log(
                    f"{candidate} lacks venv module. "
                    "Fix: sudo apt install python3-venv",
                    Qgis.Warning,
                )
        except Exception as e:
            _log(f"Checking {candidate}: {e}", Qgis.Warning)

    return None


def _get_python_for_venv() -> str:
    """Return the real Python interpreter for creating venvs.

    Windows → sys.prefix/python.exe   (NOT sys.executable = qgis-bin.exe)
    Linux   → system python3 with venv module
    macOS   → sys.executable (works correctly on Mac QGIS)
    """
    if sys.platform == "win32":
        py = _get_qgis_python_windows()
        if py:
            return py
        raise RuntimeError(
            "Could not find python.exe for QGIS.\n"
            f"sys.prefix = {sys.prefix}\n"
            "Reinstall QGIS or set QMAXENT_CACHE_DIR to a writable location."
        )

    if sys.platform == "darwin":
        py = _verify_python(sys.executable)
        if py:
            return py
        raise RuntimeError(
            f"sys.executable is not a working Python on macOS: {sys.executable}"
        )

    # Linux
    py = _get_linux_system_python()
    if py:
        return py
    raise RuntimeError(
        "No suitable Python3 found on Linux.\n"
        "Install python3-venv: sudo apt install python3-venv"
    )


# ---------------------------------------------------------------------------
# Venv path helpers
# ---------------------------------------------------------------------------

def get_venv_python_path(venv_dir: str = None) -> str:
    venv_dir = venv_dir or VENV_DIR
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python3")


def get_venv_pip_path(venv_dir: str = None) -> str:
    venv_dir = venv_dir or VENV_DIR
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "pip.exe")
    return os.path.join(venv_dir, "bin", "pip")


def get_venv_site_packages(venv_dir: str = None) -> str:
    venv_dir = venv_dir or VENV_DIR
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Lib", "site-packages")
    lib = os.path.join(venv_dir, "lib")
    if os.path.exists(lib):
        for entry in os.listdir(lib):
            if entry.startswith("python"):
                sp = os.path.join(lib, entry, "site-packages")
                if os.path.exists(sp):
                    return sp
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return os.path.join(venv_dir, "lib", py, "site-packages")


def venv_exists(venv_dir: str = None) -> bool:
    return os.path.isfile(get_venv_python_path(venv_dir or VENV_DIR))


# ---------------------------------------------------------------------------
# PROJ / GDAL path fix (critical for rasterio + QGIS coexistence)
# ---------------------------------------------------------------------------

def _fix_proj_data(site_packages: str) -> None:
    proj_candidates = [
        os.path.join(site_packages, "pyproj", "proj_dir", "share", "proj"),
        os.path.join(site_packages, "rasterio", "proj_data"),
        os.path.join(site_packages, "pyogrio", "proj_data"),
    ]
    for candidate in proj_candidates:
        if os.path.isfile(os.path.join(candidate, "proj.db")):
            try:
                import pyproj.datadir
                pyproj.datadir.set_data_dir(candidate)
                _log(f"Set pyproj.datadir={candidate}", Qgis.Info)
            except Exception as exc:
                os.environ["PROJ_DATA"] = candidate
                os.environ["PROJ_LIB"] = candidate
                _log(f"Set PROJ_DATA={candidate} (API unavailable: {exc})",
                     Qgis.Info)
            break
    for candidate in [
        os.path.join(site_packages, "rasterio", "gdal_data"),
        os.path.join(site_packages, "pyogrio", "gdal_data"),
    ]:
        if os.path.isdir(candidate):
            os.environ["GDAL_DATA"] = candidate
            _log(f"Set GDAL_DATA={candidate}", Qgis.Info)
            break


# ---------------------------------------------------------------------------
# sys.path injection
# ---------------------------------------------------------------------------

def ensure_venv_packages_available() -> bool:
    if not venv_exists():
        _log("Venv does not exist", Qgis.Warning)
        return False
    site_packages = get_venv_site_packages()
    if not os.path.exists(site_packages):
        _log(f"site-packages not found: {site_packages}", Qgis.Warning)
        return False
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
        _log(f"Added to sys.path: {site_packages}", Qgis.Info)
    _fix_proj_data(site_packages)
    return True


# ---------------------------------------------------------------------------
# Venv creation
# ---------------------------------------------------------------------------

def create_venv(
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[bool, str]:
    """Create the virtual environment using the QGIS-bundled Python.

    CRITICAL: uses _get_python_for_venv(), NOT sys.executable.
    On Windows, sys.executable == qgis-bin.exe and would launch another
    QGIS instance, treating the subprocess arguments as data source paths.
    """
    _log(f"Creating venv at: {VENV_DIR}", Qgis.Info)
    if progress_callback:
        progress_callback(5, "Locating Python interpreter...")

    try:
        system_python = _get_python_for_venv()
    except RuntimeError as e:
        return False, str(e)

    _log(f"Using Python for venv: {system_python}", Qgis.Info)
    if progress_callback:
        progress_callback(8, "Creating virtual environment...")

    # Keep full environment (including PYTHONHOME) — the system Python needs
    # it to find its own stdlib when spawned as a child process.
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONIOENCODING"] = "utf-8"
    kwargs = _get_subprocess_kwargs()

    try:
        result = subprocess.run(
            [system_python, "-m", "venv", VENV_DIR],
            capture_output=True, text=True, timeout=120,
            env=env, **kwargs,
        )
        if result.returncode != 0:
            err = result.stderr or result.stdout or f"exit {result.returncode}"
            _log(f"venv creation failed: {err}", Qgis.Critical)
            if os.path.exists(VENV_DIR):
                shutil.rmtree(VENV_DIR, ignore_errors=True)
            return False, f"Failed to create venv: {err[:400]}"

        # Bootstrap pip if missing (some QGIS builds exclude ensurepip)
        if not os.path.isfile(get_venv_pip_path()):
            _log("pip not in venv, bootstrapping via ensurepip...", Qgis.Info)
            r2 = subprocess.run(
                [get_venv_python_path(), "-m", "ensurepip", "--upgrade"],
                capture_output=True, text=True, timeout=60,
                env=env, **kwargs,
            )
            if r2.returncode != 0:
                err = r2.stderr or r2.stdout
                shutil.rmtree(VENV_DIR, ignore_errors=True)
                return False, f"pip bootstrap failed: {err[:200]}"

        _log("Venv created successfully", Qgis.Success)
        if progress_callback:
            progress_callback(12, "Virtual environment created")
        return True, "Virtual environment created"

    except subprocess.TimeoutExpired:
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        return False, "Venv creation timed out"
    except Exception as e:
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        return False, f"Error creating venv: {e}"


# ---------------------------------------------------------------------------
# Error detection helpers
# ---------------------------------------------------------------------------

_SSL_PATTERNS = [
    "ssl", "certificate verify failed", "SSLError",
    "CERTIFICATE_VERIFY_FAILED", "tlsv1 alert",
]
_NET_PATTERNS = [
    "connectionreseterror", "connection aborted", "remotedisconnected",
    "newconnectionerror", "network is unreachable",
    "name or service not known", "readtimeouterror", "connecttimeouterror",
]


def _is_ssl_error(s: str) -> bool:
    sl = s.lower()
    return any(p.lower() in sl for p in _SSL_PATTERNS)


def _is_network_error(s: str) -> bool:
    sl = s.lower()
    return not _is_ssl_error(s) and any(p in sl for p in _NET_PATTERNS)


def _pip_ssl_flags() -> List[str]:
    return [
        "--trusted-host", "pypi.org",
        "--trusted-host", "pypi.python.org",
        "--trusted-host", "files.pythonhosted.org",
    ]


def _pip_proxy_flags() -> List[str]:
    proxy = _get_qgis_proxy_settings()
    return ["--proxy", proxy] if proxy else []


# ---------------------------------------------------------------------------
# Package installation
# ---------------------------------------------------------------------------

def install_dependencies(
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, str]:
    """Install REQUIRED_PACKAGES into the venv.

    Uses temp-file output instead of PIPE to avoid Windows QThread
    pipe-deadlock (same pattern as GeoAI).
    """
    if not venv_exists():
        return False, "Venv does not exist"

    python = get_venv_python_path()
    env    = _get_clean_env_for_pip()
    kwargs = _get_subprocess_kwargs()
    specs  = [f"{name}{ver}" for name, ver in REQUIRED_PACKAGES]

    _log(f"Installing {len(specs)} packages", Qgis.Info)
    if progress_callback:
        progress_callback(15, "Installing dependencies (this may take several minutes)...")

    cmd = (
        [python, "-m", "pip", "install",
         "--upgrade", "--no-warn-script-location",
         "--disable-pip-version-check", "--prefer-binary"]
        + _pip_ssl_flags()
        + _pip_proxy_flags()
        + specs
    )

    os.makedirs(CACHE_DIR, exist_ok=True)
    fd, out_path = tempfile.mkstemp(suffix="_qmaxent_pip.txt", dir=CACHE_DIR)
    os.close(fd)

    try:
        with open(out_path, "w", encoding="utf-8", errors="replace") as out_f:
            proc = subprocess.Popen(
                cmd,
                stdout=out_f,
                stderr=subprocess.STDOUT,
                env=env,
                **kwargs,
            )

        start    = time.monotonic()
        last_pos = 0
        # Phase milestones used to map pip log lines to progress bar values.
        # pip's installation has roughly four observable phases:
        #   Collecting/Downloading metadata     →  ~25%
        #   Downloading wheel files             →  ~50%
        #   Building wheels (if any)            →  ~70%
        #   Installing collected packages       →  ~85%
        # We watch for these tokens in the pip output and step the bar to
        # the matching milestone, so the user sees real progress instead
        # of a time-based extrapolation that stalls near 90%.
        phase_pct = 15

        while proc.poll() is None:
            if cancel_check and cancel_check():
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return False, "Installation cancelled"

            time.sleep(2)
            elapsed = int(time.monotonic() - start)

            try:
                with open(out_path, "r",
                          encoding="utf-8", errors="replace") as f:
                    f.seek(last_pos)
                    chunk = f.read(4096)
                    last_pos = f.tell()
                if chunk.strip() and progress_callback:
                    last_line = chunk.strip().splitlines()[-1]
                    # Phase-based step ups (only ratchet upward).
                    chunk_l = chunk.lower()
                    if "successfully installed" in chunk_l:
                        phase_pct = max(phase_pct, 90)
                    elif "installing collected packages" in chunk_l:
                        phase_pct = max(phase_pct, 85)
                    elif "building wheel" in chunk_l or "running setup.py" in chunk_l:
                        phase_pct = max(phase_pct, 70)
                    elif "downloading " in chunk_l:
                        phase_pct = max(phase_pct, 50)
                    elif "collecting " in chunk_l:
                        phase_pct = max(phase_pct, 25)
                    # Within a phase, time-based extrapolation gives some
                    # micro-progress so the bar isn't completely static.
                    time_floor = min(15 + int(elapsed * 0.4), 88)
                    pct = max(phase_pct, time_floor)
                    pct = min(pct, 90)   # leave 90+ for post-pip steps
                    progress_callback(pct, last_line[:100])
            except Exception:
                pass

        proc.wait(timeout=30)

        try:
            with open(out_path, "r",
                      encoding="utf-8", errors="replace") as f:
                output = f.read()
        except Exception:
            output = ""

        if proc.returncode != 0:
            _log(f"pip failed:\n{output[-2000:]}", Qgis.Critical)
            if _is_ssl_error(output):
                return False, "Installation failed: SSL certificate error"
            if _is_network_error(output):
                return False, "Installation failed: network error"
            return False, (
                f"pip failed (exit {proc.returncode}): {output[-400:]}"
            )

        _log("All packages installed", Qgis.Success)
        if progress_callback:
            progress_callback(92, "Packages installed")
        return True, "Dependencies installed"

    except subprocess.TimeoutExpired:
        return False, "Installation timed out"
    except Exception as e:
        return False, f"Error during installation: {e}"
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

_VERIFY_CODE = {
    "numpy":        "import numpy; print(numpy.__version__)",
    "pandas":       "import pandas; print(pandas.__version__)",
    "pyproj":       "import pyproj; print(pyproj.__version__)",
    "scipy":        "import scipy; print(scipy.__version__)",
    "scikit-learn": "import sklearn; print(sklearn.__version__)",
    "matplotlib":   "import matplotlib; print(matplotlib.__version__)",
    "rasterio":     "import rasterio; print(rasterio.__version__)",
    "rtree":        "import rtree; print(rtree.__version__)",
    "geopandas":    "import geopandas; print(geopandas.__version__)",
    "tqdm":         "import tqdm; print(tqdm.__version__)",
    "elapid":       "import elapid; print(elapid.__version__)",
}


def verify_venv(
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[bool, str]:
    if not venv_exists():
        return False, "Venv not found"

    python = get_venv_python_path()
    site_packages = get_venv_site_packages()
    # Use the verify-specific environment so native modules (rasterio,
    # pyproj, pyogrio) can locate their bundled PROJ / GDAL data even
    # though the parent QGIS process strips PROJ_DATA via clean_env.
    env    = _get_env_for_verify(site_packages)
    kwargs = _get_subprocess_kwargs()
    total  = len(REQUIRED_PACKAGES)

    for i, (name, _) in enumerate(REQUIRED_PACKAGES):
        if progress_callback:
            progress_callback(
                92 + int(i / total * 7),
                f"Verifying {name}... ({i+1}/{total})",
            )
        code = _VERIFY_CODE.get(name, f"import {name.replace('-','_')}")
        try:
            r = subprocess.run(
                [python, "-c", code],
                capture_output=True, text=True, timeout=60,
                env=env, **kwargs,
            )
            if r.returncode != 0:
                # Show enough of the traceback to actually diagnose the
                # failure. The previous 300-char cap was truncating the
                # final line — the one that names the real cause (e.g.
                # "ImportError: numpy.core.multiarray failed to import"
                # for an ABI mismatch). 1200 chars comfortably covers
                # a multi-frame Python traceback.
                detail = (r.stderr or r.stdout)[:1200]
                _log(f"Verify failed {name}: {detail}", Qgis.Warning)
                return False, f"Package {name} is broken: {detail}"
        except subprocess.TimeoutExpired:
            return False, f"Verification timed out for {name}"
        except Exception as e:
            return False, f"Verification error for {name}: {e}"

    _log("All packages verified", Qgis.Success)
    if progress_callback:
        progress_callback(100, "All dependencies verified")
    return True, "Virtual environment ready"


# ---------------------------------------------------------------------------
# Quick status check (no subprocess — safe for main thread)
# ---------------------------------------------------------------------------

def _quick_check() -> Tuple[bool, str]:
    site_packages = get_venv_site_packages()
    if not os.path.exists(site_packages):
        return False, "site-packages not found"
    for pkg_dir in ("elapid", "rasterio", "geopandas"):
        if not os.path.exists(os.path.join(site_packages, pkg_dir)):
            return False, f"{pkg_dir} not found"
    return True, "packages present"


def get_venv_status() -> Tuple[bool, str]:
    if not venv_exists():
        return False, "Dependencies not installed"
    ok, msg = _quick_check()
    if not ok:
        return False, f"Incomplete installation: {msg}"
    stored  = _read_deps_hash()
    current = _compute_deps_hash()
    if stored and stored != current:
        return False, "Dependencies need updating"
    if not stored:
        _write_deps_hash()
    return True, (
        f"Ready (Python {sys.version_info.major}.{sys.version_info.minor})"
    )


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------

def create_venv_and_install(
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, str]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        test = os.path.join(CACHE_DIR, ".write_test")
        with open(test, "w") as f:
            f.write("test")
        os.remove(test)
    except OSError as e:
        return False, f"Cannot write to {CACHE_DIR}: {e}"

    if not venv_exists():
        ok, msg = create_venv(progress_callback)
        if not ok:
            return False, msg
    else:
        _log("Venv already exists, skipping creation", Qgis.Info)
        if progress_callback:
            progress_callback(12, "Venv already exists")

    if cancel_check and cancel_check():
        return False, "Cancelled"

    ok, msg = install_dependencies(progress_callback, cancel_check)
    if not ok:
        return False, msg

    if cancel_check and cancel_check():
        return False, "Cancelled"

    ok, msg = verify_venv(progress_callback)
    if not ok:
        return False, f"Verification failed: {msg}"

    _write_deps_hash()
    return True, "QMaxent environment ready"


def cleanup_sidelined_venvs() -> int:
    """Best-effort delete of any orphaned ``<base>_orphaned_*`` folders
    sitting beside the active VENV_DIR.

    These are leftovers from a previous remove_venv() call that ran
    while QGIS still held .pyd handles. On a fresh QGIS start the OS
    has released those handles, so a plain rmtree usually succeeds.
    Anything that still can't be removed is silently skipped — it
    will be tried again on the next QGIS start.

    Returns the number of orphaned folders that were fully removed.
    Safe to call from initGui() on every QGIS launch.
    """
    parent = os.path.dirname(VENV_DIR)
    if not os.path.isdir(parent):
        return 0
    base = os.path.basename(VENV_DIR)
    n_removed = 0
    try:
        for entry in os.listdir(parent):
            if not entry.startswith(f"{base}_orphaned_"):
                continue
            full = os.path.join(parent, entry)
            try:
                shutil.rmtree(full)
                n_removed += 1
                _log(f"Cleaned up orphaned venv: {entry}", Qgis.Info)
            except Exception as e:
                # Files still locked — try again next start.
                _log(f"Could not yet clean {entry}: {e}", Qgis.Info)
    except OSError:
        pass
    return n_removed


def remove_venv() -> Tuple[bool, str]:
    """Remove the QMaxent virtual environment, robustly on Windows.

    On Windows, files inside the venv may be locked by the running
    QGIS process — any ``.pyd`` that ``ensure_venv_packages_available``
    caused to be imported (rasterio._base, numpy's _multiarray, etc.)
    is held open until QGIS itself exits. Plain ``shutil.rmtree``
    raises ``PermissionError`` on the first such file and bails out,
    leaving a partially-deleted folder behind. The next install then
    overlays new Python files on top of the stale ``.pyd`` — producing
    the cryptic ``init rasterio._base`` cython ImportError users see
    after Remove → Install while QGIS is still running.

    Strategy:
      1. Try a clean ``shutil.rmtree``. Most common case — succeeds
         when no native module from the venv was ever imported (e.g.
         user installed deps but never opened the analysis dock).
      2. On failure, **rename** VENV_DIR to ``<VENV_DIR>_orphaned_<ts>``.
         Renaming a folder only requires write access to the parent
         directory, which we have, so it succeeds even when files
         inside are locked. After the rename, VENV_DIR no longer
         exists and the next install will create a clean fresh venv
         — no overlay, no jumbled state.
      3. Best-effort delete of the sidelined folder with
         ``ignore_errors=True`` to free what we can; the locked
         ``.pyd``\u2009s stay until QGIS restarts, when
         ``cleanup_sidelined_venvs()`` finishes the job.
    """
    if not os.path.exists(VENV_DIR):
        return True, "Venv does not exist"

    # Step 1: try the clean path first.
    try:
        shutil.rmtree(VENV_DIR)
        _log("Venv removed cleanly", Qgis.Success)
        cleanup_sidelined_venvs()
        return True, "Venv removed"
    except Exception as primary_err:
        _log(
            f"Clean rmtree failed ({primary_err}); falling back to "
            f"sideline-by-rename strategy",
            Qgis.Warning,
        )

    # Step 2: rename the live VENV_DIR out of the way so a subsequent
    # install never overlays a half-deleted tree. We use a microsecond-
    # resolution suffix so multiple Remove→Install cycles in one QGIS
    # session don't collide.
    sidelined = f"{VENV_DIR}_orphaned_{int(time.time() * 1000)}"
    try:
        os.rename(VENV_DIR, sidelined)
        _log(f"Venv sidelined: {VENV_DIR} -> {sidelined}", Qgis.Warning)
    except Exception as rename_err:
        # Rename can fail too if the parent dir is itself locked
        # (rare on Windows, but possible with some antivirus tools).
        # In that case the user has to resolve it manually.
        # Add a plain-language explanation of what the OS-level
        # error code actually means in our context — raw "WinError 5
        # 액세스가 거부되었습니다" is technically correct but tells
        # users nothing actionable. The real meaning here is almost
        # always "Python is still holding .pyd handles inside the
        # folder, which only release when QGIS exits".
        return False, (
            f"Failed to remove the QMaxent virtual environment.\n\n"
            f"Reason: {rename_err}\n"
            f"(In plain terms: Python is still holding open handles "
            f"to native modules inside the folder — typically "
            f"rasterio, numpy, or pyproj that QGIS imported earlier "
            f"in this session. Windows does not allow renaming or "
            f"deleting a folder while any file inside it is open. "
            f"These handles are only released when QGIS itself "
            f"exits.)\n\n"
            f"This typically means QGIS is currently using files in:\n"
            f"  {VENV_DIR}\n\n"
            f"To resolve:\n"
            f"  1. Close QGIS completely.\n"
            f"  2. Delete the folder manually.\n"
            f"  3. Restart QGIS and click "
            f"\"Install / Update Dependencies\"."
        )

    # Step 3: best-effort cleanup. Anything still locked stays under
    # the _orphaned_ name and gets retried on next QGIS start.
    shutil.rmtree(sidelined, ignore_errors=True)
    cleanup_sidelined_venvs()

    if os.path.exists(sidelined):
        # Some locked files survived — be honest with the user.
        return True, (
            "Environment removed.\n\n"
            "Some files are still in use by the running QGIS process "
            "and were sidelined; they will be cleaned up automatically "
            "when you restart QGIS. The next install will start from "
            "a clean folder, so this does not block anything — but "
            "if you want a fully clean slate before reinstalling, "
            "close and reopen QGIS first."
        )
    return True, "Venv removed (after sideline)"
