"""Regression tests for the v0.2.0 fixes (SoftwareX review, Oct 2026).

Covers: local/engineering-CRS detection and the QGIS CRS override in
Harmonize (reviewer 1), the maxnet auto-feature rule (reviewer 3,
comment 2), the coefficient table written next to model.pkl
(reviewer 1), projection with categorical classes unseen in training,
and CSV column detection for presence tables.
"""

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from bridge import raster_bridge as rb
from core.maxent_rules import auto_feature_types


def _plugin_module(dotted: str):
    """Import a plugin module that uses package-relative imports.

    workers/ and dialogs/ use ``from ..i18n import tr``, so they must be
    imported through a package. The repository root is registered as
    the package ``qmaxent`` (the plugin's install name).
    """
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    if "qmaxent" not in sys.modules:
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "qmaxent", root / "__init__.py", submodule_search_locations=[str(root)]
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["qmaxent"] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module("qmaxent." + dotted)


AEA = CRS.from_proj4(
    "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +datum=WGS84 +units=m +no_defs"
)
LOCAL = CRS.from_wkt(
    'LOCAL_CS["NAD83(NSRS2007) / Michigan South",'
    'UNIT["metre",1],AXIS["Easting",EAST],'
    'AXIS["Northing",NORTH]]'
)


def _write(path, crs, origin, size=(40, 50), res=2000.0, value=1.0):
    arr = np.full(size, value, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size[1],
        height=size[0],
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(*origin, res, res),
        nodata=-9999,
    ) as d:
        d.write(arr, 1)
    return str(path)


@pytest.fixture
def rasters(tmp_path):
    # Southern Michigan, in Albers (AEA) and in Michigan South coordinates
    a = _write(tmp_path / "a.tif", AEA, (900000, 2300000))
    b = _write(tmp_path / "b.tif", AEA, (900000, 2300000), value=2)
    c = _write(tmp_path / "c_local.tif", LOCAL, (4000000, 250000), value=3)
    return a, b, c


def test_local_crs_is_reported_with_epsg_hint(rasters):
    r = rb.check_raster_consistency(list(rasters))
    assert not r["is_consistent"]
    assert [p for p, _ in r["crs_problems"]] == [rasters[2]]
    assert "EPSG:3592" in r["error"]
    assert "Set Layer CRS" in r["error"]


def test_harmonize_raises_friendly_error(rasters, tmp_path):
    with pytest.raises(rb.RasterCRSError) as e:
        rb.harmonize_rasters(list(rasters), output_dir=str(tmp_path / "o"))
    assert "projjson" not in str(e.value).lower()
    assert "c_local.tif" in str(e.value)


def test_qgis_crs_override_allows_harmonize(rasters, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    new, _ = rb.harmonize_rasters(
        list(rasters), output_dir=str(out), crs_overrides={2: "EPSG:3592"}
    )
    assert rb.check_raster_consistency(new)["is_consistent"]


def test_override_never_touches_healthy_raster(rasters):
    r = rb.check_raster_consistency(list(rasters[:2]), crs_overrides={0: "EPSG:4326"})
    assert r["is_consistent"]


@pytest.mark.parametrize(
    "n,expected",
    [
        (5, ["linear"]),
        (12, ["linear", "quadratic"]),
        (40, ["linear", "quadratic", "hinge"]),
        (80, ["linear", "quadratic", "product", "hinge"]),
    ],
)
def test_auto_feature_rule_is_maxnet_default_v020(n, expected):
    assert auto_feature_types(n) == expected


def test_coefficient_table_matches_elapid_layout():
    elapid = pytest.importorskip("elapid")
    MaxentWorker = _plugin_module("workers.maxent_worker").MaxentWorker
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    x[:, 2] = rng.integers(0, 3, 300)
    y = (x[:, 0] > 0.3).astype(int)
    m = elapid.MaxentModel(n_hinge_features=5, n_threshold_features=4)
    m.fit(x, y, categorical=[2], labels=["a", "b", "c"])
    rows = MaxentWorker._coefficient_rows(
        m, ["a", "b", "c"], m.transformer.feature_names_, len(m.beta_scores_)
    )
    assert [r[0] for r in rows] == list(m.transformer.feature_names_)
    feats = m.transformer.transform(x)
    col = {"a": 0, "b": 1}
    for i, (ftype, var, detail, knot) in enumerate(rows):
        if ftype == "threshold":
            assert np.array_equal(feats[:, i], (x[:, col[var]] > knot))
        if ftype == "hinge" and detail.startswith("left"):
            assert np.all(feats[x[:, col[var]] <= knot, i] == 0)
        if ftype == "hinge" and detail.startswith("right"):
            assert np.allclose(feats[x[:, col[var]] >= knot, i], 1)


def test_csv_column_detection(tmp_path):
    m = _plugin_module("dialogs.csv_presence_dialog")
    guess_xy, sniff_table = m.guess_xy, m.sniff_table
    p = tmp_path / "gbif.csv"
    p.write_text(
        "gbifID;species;decimalLongitude;decimalLatitude\n1;x;-70.1;18.9\n", encoding="utf-8"
    )
    delim, header = sniff_table(str(p))
    assert delim == ";"
    assert guess_xy(header) == ("decimalLongitude", "decimalLatitude")


def test_projection_tolerates_unseen_category():
    elapid = pytest.importorskip("elapid")
    from bridge.elapid_bridge import tolerate_unknown_categories

    rng = np.random.default_rng(1)
    x = np.c_[rng.normal(size=200), rng.integers(0, 3, 200)]
    y = (x[:, 0] > 0).astype(int)
    m = elapid.MaxentModel(feature_types=["linear"])
    m.fit(x, y, categorical=[1])
    new = np.array([[0.1, 7.0]])  # class 7 never seen in training
    with pytest.raises(ValueError):
        m.predict(new)
    assert tolerate_unknown_categories(m) == 1
    assert np.isfinite(m.predict(new)).all()
