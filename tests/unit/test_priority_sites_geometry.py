"""Unit tests for the priority-sites geometry pipeline.

Defends the spacing semantics that the manuscript states (§ 2.2 ⑤,
§ 3.4):

  - Discovery mode candidates must respect both a minimum distance
    to existing presence points AND a minimum distance to other
    accepted candidates.
  - Validation mode candidates must be drawn from each of the four
    suitability quartiles (Rhoden et al. 2017 stratification).
  - With ``random_seed`` fixed, repeated calls return identical
    coordinate lists.

Generates a small synthetic GeoTIFF on the fly under ``tmp_path`` so
the test is fully hermetic and adds no fixture bytes to the repo.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.needs_rasterio]


def _write_synthetic_raster(out: Path, size: int = 60, seed: int = 1) -> Path:
    """Write a small EPSG:4326 GeoTIFF with a north-south suitability
    gradient (high north, low south). Returns the path."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    # Smooth gradient so quartile stratification has something to bite
    # into, plus a touch of noise so MaxSSS-style decision boundaries
    # are not perfectly degenerate.
    base = np.linspace(0.0, 1.0, size).reshape(-1, 1)
    field = np.broadcast_to(base, (size, size)).copy()
    field += rng.normal(0, 0.02, size=(size, size))
    field = np.clip(field, 0.0, 1.0).astype("float32")

    path = out / "synthetic_suitability.tif"
    # ~ 0.01° per pixel, north-up.
    transform = from_origin(west=126.0, north=37.0, xsize=0.01, ysize=0.01)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=size, width=size, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(field, 1)
    return path


def test_spacing_constraints_are_respected(tmp_path):
    from bridge.priority_sites import extract_priority_sites

    rpath = _write_synthetic_raster(tmp_path)
    presences = [(126.30, 36.45)]  # somewhere inside the raster
    sites = extract_priority_sites(
        prediction_path=str(rpath),
        presence_xy=presences,
        threshold=0.7,                 # only the northern band qualifies
        n_sites=8,
        min_distance_from_presence_m=1000.0,
        min_distance_between_sites_m=2000.0,
        stratify_by_quartile=False,
        sampling_order="topn",
        random_seed=42,
    )
    assert sites, "no sites returned; threshold may be too aggressive"

    # All sites must be at least 1 km from every presence point.
    # Equirectangular degree-to-metre at 36.5°N is ~ 89 km / degree.
    deg_per_km = 1.0 / 89.0
    min_from_pres_deg = 1.0 * deg_per_km
    for s in sites:
        d = math.hypot(s["lon"] - presences[0][0], s["lat"] - presences[0][1])
        assert d >= min_from_pres_deg * 0.95, "site too close to a presence point"

    # All site pairs must be at least 2 km apart.
    min_between_deg = 2.0 * deg_per_km
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            d = math.hypot(a["lon"] - b["lon"], a["lat"] - b["lat"])
            assert d >= min_between_deg * 0.95, "two sites are too close"


def test_validation_mode_covers_all_four_quartiles(tmp_path):
    """Rhoden et al. (2017) stratification: with stratify_by_quartile=True
    and a target large enough to fill every quartile, the returned sites
    span all four suitability quartiles."""
    from bridge.priority_sites import extract_priority_sites

    rpath = _write_synthetic_raster(tmp_path, size=80, seed=7)

    sites = extract_priority_sites(
        prediction_path=str(rpath),
        presence_xy=[],
        threshold=0.05,                   # keep nearly all suitable cells
        n_sites=12,
        min_distance_from_presence_m=0.0,
        min_distance_between_sites_m=0.0,
        stratify_by_quartile=True,
        sampling_order="topn",
        random_seed=42,
    )
    quartiles = {s["quartile"] for s in sites}
    assert quartiles >= {0, 1, 2, 3}, (
        f"Validation mode missed quartile(s): {set(range(4)) - quartiles}"
    )


def test_deterministic_under_fixed_seed(tmp_path):
    """Reproducibility contract documented in § 2.2 ②: with random_seed
    fixed, two runs must return byte-identical coordinate lists."""
    from bridge.priority_sites import extract_priority_sites

    rpath = _write_synthetic_raster(tmp_path)
    kw = dict(
        prediction_path=str(rpath),
        presence_xy=[(126.30, 36.45)],
        threshold=0.7,
        n_sites=6,
        min_distance_from_presence_m=500.0,
        min_distance_between_sites_m=500.0,
        stratify_by_quartile=False,
        sampling_order="random",
        random_seed=42,
    )
    a = extract_priority_sites(**kw)
    b = extract_priority_sites(**kw)
    assert a == b, "random seed not honoured: two runs returned different sites"
