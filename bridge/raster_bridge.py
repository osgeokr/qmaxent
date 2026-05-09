"""Bridge: QGIS raster layers ↔ rasterio file paths."""

import os
import tempfile

from qgis.core import (
    QgsRasterLayer,
    QgsRasterFileWriter,
    QgsRasterPipe,
    QgsMessageLog,
    Qgis,
)


def layer_to_path(layer: QgsRasterLayer) -> str:
    """Resolve a raster layer to a file path usable by rasterio.

    For file-backed layers the source path is returned directly.
    For in-memory or virtual layers a temporary GeoTIFF is written.

    Args:
        layer: A QGIS raster layer.

    Returns:
        Absolute file path to a GeoTIFF readable by rasterio.
    """
    source = layer.source()

    # File-backed layer: return path directly if it exists
    if os.path.isfile(source.split("|")[0]):
        return source.split("|")[0]

    # Memory / virtual layer: export to temp GeoTIFF
    QgsMessageLog.logMessage(
        f"Layer '{layer.name()}' is not file-backed; exporting to temp GeoTIFF",
        "QMaxent", Qgis.Info,
    )
    # Race-free temp filename via mkstemp; close the fd immediately so
    # QGIS's QgsRasterFileWriter can open the path for writing itself.
    fd, tmp = tempfile.mkstemp(suffix=".tif", prefix="qmaxent_ras_")
    os.close(fd)
    provider = layer.dataProvider()
    pipe = QgsRasterPipe()
    if not pipe.set(provider.clone()):
        raise RuntimeError(f"Cannot clone provider for layer '{layer.name()}'")

    writer = QgsRasterFileWriter(tmp)
    writer.setOutputFormat("GTiff")
    error = writer.writeRaster(
        pipe,
        provider.xSize(),
        provider.ySize(),
        provider.extent(),
        provider.crs(),
    )
    if error != QgsRasterFileWriter.NoError:
        raise RuntimeError(f"Failed to export raster layer '{layer.name()}'")

    return tmp


def layers_to_paths(layers: list) -> list:
    """Convert a list of QgsRasterLayer objects to file paths.

    Args:
        layers: List of QgsRasterLayer objects.

    Returns:
        List of file path strings.
    """
    return [layer_to_path(layer) for layer in layers]


def get_raster_crs(path: str):
    """Return a rasterio CRS for the raster at `path`, or None on failure.

    Used by the worker to verify CRS consistency between presence points
    and environmental rasters before training. Spatial cross-validation
    strategies (Geographic K-Fold, Buffered LOO) operate in the units of
    the presence-layer CRS, so a CRS mismatch silently corrupts those
    distance-based splits (Roberts et al. 2017).
    """
    import rasterio
    try:
        with rasterio.open(path) as src:
            return src.crs
    except Exception:
        return None


# ── Raster consistency check & harmonization ──────────────────────────────
#
# Maxent's mathematical formulation assumes that all environmental
# covariates are evaluated on the same spatial grid: same CRS, same
# extent, same resolution. When the input rasters do not share these
# properties, predictions are affected by silent reprojection errors,
# extrapolation outside the smallest common extent, and resolution-
# dependent variable importance distortions.
#
# Established SDM packages handle this in two ways: maxent.jar/dismo
# *require* the user to harmonize beforehand; biomod2 and SDMSelect
# expose explicit pre-harmonization helpers (e.g. SDMSelect's
# Prepare_r_multi). We follow the latter pattern, with rasterio.warp
# as the engine. Continuous rasters are resampled bilinearly; rasters
# that the user has tagged as categorical use nearest-neighbor
# resampling so that class labels are preserved (resampling a class
# raster with bilinear or cubic produces meaningless intermediate
# values like 2.7 between class 2 and class 3).

def check_raster_consistency(raster_paths: list) -> dict:
    """Inspect a set of rasters and report whether they share a grid.

    Returns a dict with the following keys:
        crs_uniform        bool — all rasters share one CRS
        extent_uniform     bool — all rasters share bounding boxes (within a
                                  small numerical tolerance)
        resolution_uniform bool — all rasters share pixel sizes
        rasters            list of per-raster (path, crs, bounds, res) tuples
        is_consistent      bool — convenience: True iff all three uniform

    On read failure the returned dict has is_consistent=False and an
    'error' key describing the first failure.
    """
    import rasterio
    info = []
    try:
        for p in raster_paths:
            with rasterio.open(p) as src:
                info.append({
                    "path":   p,
                    "crs":    src.crs,
                    "bounds": tuple(src.bounds),
                    "res":    tuple(src.res),
                })
    except Exception as e:
        return {
            "is_consistent": False,
            "error": f"Could not read raster: {e}",
            "crs_uniform": False,
            "extent_uniform": False,
            "resolution_uniform": False,
            "rasters": info,
        }

    if not info:
        return {
            "is_consistent": False,
            "error": "No rasters provided.",
            "crs_uniform": False,
            "extent_uniform": False,
            "resolution_uniform": False,
            "rasters": [],
        }

    # CRS comparison — use string form to side-step minor object differences
    crs_strings = {str(r["crs"]) if r["crs"] is not None else "" for r in info}
    crs_uniform = len(crs_strings) == 1 and "" not in crs_strings

    # Bounds: tolerance of 1e-6 in CRS units handles float roundoff
    def _bounds_close(a, b):
        return all(abs(x - y) < 1e-6 for x, y in zip(a, b))
    first_bounds = info[0]["bounds"]
    extent_uniform = all(_bounds_close(r["bounds"], first_bounds) for r in info)

    # Resolution: tolerance of 1e-9 of CRS units
    def _res_close(a, b):
        return all(abs(x - y) < 1e-9 for x, y in zip(a, b))
    first_res = info[0]["res"]
    resolution_uniform = all(_res_close(r["res"], first_res) for r in info)

    return {
        "is_consistent":      crs_uniform and extent_uniform and resolution_uniform,
        "crs_uniform":        crs_uniform,
        "extent_uniform":     extent_uniform,
        "resolution_uniform": resolution_uniform,
        "rasters":            info,
    }


def _intersection_bounds_in_crs(infos: list, target_crs):
    """Return the bounds (l, b, r, t) of the intersection of all rasters
    expressed in `target_crs`. Each raster's source bounds are
    transformed into target_crs first via rasterio.warp.transform_bounds
    so that an intersection is well-defined across heterogeneous CRSes.
    """
    from rasterio.warp import transform_bounds
    transformed = []
    for r in infos:
        if r["crs"] is None:
            # No CRS metadata; assume same as target (a common-but-fragile
            # convention for asc/grd files written without projection info).
            transformed.append(r["bounds"])
        else:
            transformed.append(
                transform_bounds(r["crs"], target_crs, *r["bounds"])
            )
    # Intersection: max of left/bottom, min of right/top
    lefts   = [b[0] for b in transformed]
    bottoms = [b[1] for b in transformed]
    rights  = [b[2] for b in transformed]
    tops    = [b[3] for b in transformed]
    return (max(lefts), max(bottoms), min(rights), min(tops))


def harmonize_rasters(
    raster_paths: list,
    categorical_indices: list = None,
    template_idx: int = 0,
    output_dir: str = None,
    progress_callback=None,
    cancel_check=None,
):
    """Reproject, resample, and clip every raster to a common grid.

    Produces a list of new raster paths whose CRS, resolution, and extent
    all match the *template* raster's grid (resolution + CRS) clipped to
    the intersection of the input rasters' extents. The intersection
    convention follows the standard SDM practice of training only where
    every covariate is defined (Hijmans 2024; SDMSelect Prepare_r_multi).

    Args:
        raster_paths: input raster file paths.
        categorical_indices: indices of rasters that should be resampled
            with nearest-neighbor instead of bilinear, to preserve class
            labels. None ↔ all continuous.
        template_idx: index of the raster whose CRS and pixel size define
            the output grid.
        output_dir: directory for harmonized copies. If None, a fresh
            temporary directory is created.
        progress_callback: optional callable(int, str) for percent and
            status messages, invoked once per raster.
        cancel_check: optional callable() -> bool. Polled at the start of
            each per-raster iteration; when it returns True the loop
            aborts cleanly and a RuntimeError("Cancelled") is raised.
            The harmonize worker passes a flag-checker here so the
            "Cancel" button on the modal progress dialog actually
            interrupts the work in progress.

    Returns:
        Tuple (new_paths, output_dir). The caller is responsible for
        cleaning up output_dir when the workflow ends; the worker uses
        a try/finally for that.
    """
    import os
    import tempfile
    import rasterio
    from rasterio.warp import (
        calculate_default_transform, reproject, Resampling, transform_bounds,
    )

    if not raster_paths:
        return [], None
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="qmaxent_harmonized_")

    cat_set = set(categorical_indices or [])

    # 1. Read template grid (CRS, native resolution).
    with rasterio.open(raster_paths[template_idx]) as tmpl:
        target_crs = tmpl.crs
        target_res = tmpl.res
        if target_crs is None:
            raise RuntimeError(
                "Template raster has no CRS metadata; cannot harmonize. "
                "Define the CRS on the template raster first."
            )

    # 2. Compute intersection extent in target CRS.
    info = []
    for p in raster_paths:
        with rasterio.open(p) as src:
            info.append({"path": p, "crs": src.crs,
                         "bounds": tuple(src.bounds)})
    bounds = _intersection_bounds_in_crs(info, target_crs)
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise RuntimeError(
            f"Rasters have no spatial overlap in the template CRS "
            f"({target_crs}); harmonization aborted. Check that input "
            f"rasters cover a common region."
        )

    # 3. Compute the destination transform on the template's grid, snapped
    #    to the intersection bounds. We use calculate_default_transform
    #    twice — first to obtain a snapped origin from the template,
    #    then we manually clip to the intersection by adjusting the
    #    transform and dimensions.
    from rasterio.transform import from_origin
    px_w, px_h = target_res
    # Snap intersection origin to the template's pixel grid so output
    # rasters are pixel-aligned to the template (the -tap behavior of
    # gdalwarp). We anchor on the template's upper-left and walk in
    # whole-pixel increments.
    with rasterio.open(raster_paths[template_idx]) as tmpl:
        # Reproject template bounds into target_crs (no-op if same CRS)
        tmpl_l, tmpl_b, tmpl_r, tmpl_t = transform_bounds(
            tmpl.crs, target_crs, *tmpl.bounds
        )
    # Find the smallest-step pixel snap from the template's upper-left
    snap_x = tmpl_l + round((bounds[0] - tmpl_l) / px_w) * px_w
    snap_y = tmpl_t - round((tmpl_t - bounds[3]) / px_h) * px_h
    width  = max(1, int(round((bounds[2] - snap_x) / px_w)))
    height = max(1, int(round((snap_y - bounds[1]) / px_h)))
    dst_transform = from_origin(snap_x, snap_y, px_w, px_h)

    # 4. Reproject every raster to this destination grid.
    new_paths = []
    n = len(raster_paths)
    for i, src_path in enumerate(raster_paths):
        # Honour user cancel before doing any work for this raster.
        # Outputs already written to output_dir are left in place; the
        # caller (HarmonizeWorker) removes the directory on cancel.
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Cancelled")
        if progress_callback:
            progress_callback(
                int(((i) / n) * 100),
                f"Harmonizing {os.path.basename(src_path)} ({i+1}/{n})",
            )
        out_path = os.path.join(
            output_dir, f"{i:02d}_{os.path.basename(src_path)}"
        )
        # Force GeoTIFF output regardless of input driver — output paths
        # are temporary and need to be readable by elapid downstream.
        if not out_path.lower().endswith((".tif", ".tiff")):
            out_path = os.path.splitext(out_path)[0] + ".tif"

        resampling = (
            Resampling.nearest if i in cat_set else Resampling.bilinear
        )

        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            profile.update({
                "driver":    "GTiff",
                "crs":       target_crs,
                "transform": dst_transform,
                "width":     width,
                "height":    height,
                "compress":  "lzw",
            })
            # Preserve the source raster's NoData value when present;
            # otherwise rasterio will fill reprojected gaps with 0,
            # which contaminates the model with spurious zero values.
            if src.nodata is not None:
                profile["nodata"] = src.nodata

            with rasterio.open(out_path, "w", **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=target_crs,
                        resampling=resampling,
                    )
                    # Carry over the GDAL "thematic" hint and category
                    # names so QMaxent's categorical detector still
                    # recognises the harmonized copy.
                    if i in cat_set:
                        dst.update_tags(LAYER_TYPE="thematic")
                        try:
                            names = src.tags(band_idx).get("CATEGORY_NAMES")
                            if names:
                                dst.update_tags(band_idx, CATEGORY_NAMES=names)
                        except Exception:
                            pass
        new_paths.append(out_path)

    if progress_callback:
        progress_callback(100, "Harmonization complete")

    return new_paths, output_dir


def load_raster_to_qgis(path: str, name: str, transform: str = "cloglog"):
    """Load a suitability raster and apply a green-to-red colour ramp.

    Args:
        path: Path to the GeoTIFF file.
        name: Display name for the layer.
        transform: Maxent transform used ('cloglog', 'logistic', or 'raw').
                   cloglog/logistic outputs are bounded [0, 1] so we use a
                   fixed range; raw uses actual raster statistics.

    Returns:
        QgsRasterLayer added to the current project.
    """
    from qgis.core import (
        QgsProject,
        QgsRasterLayer,
        QgsColorRampShader,
        QgsSingleBandPseudoColorRenderer,
        QgsRasterShader,
        QgsRasterBandStats,
    )
    from qgis.PyQt.QtGui import QColor

    layer = QgsRasterLayer(path, name)
    if not layer.isValid():
        raise RuntimeError(f"Output raster is not valid: {path}")

    # Add to project FIRST so QGIS finishes initialisation before we
    # set a custom renderer (otherwise QGIS resets it on first render).
    QgsProject.instance().addMapLayer(layer)

    # ── Value range ───────────────────────────────────────────────────────
    if transform in ("cloglog", "logistic"):
        vmin, vmax = 0.0, 1.0
    else:
        stats = layer.dataProvider().bandStatistics(
            1, QgsRasterBandStats.Min | QgsRasterBandStats.Max,
        )
        vmin = stats.minimumValue
        vmax = stats.maximumValue
        if vmax <= vmin:
            vmin, vmax = 0.0, 1.0

    # ── GnBu colour ramp (elapid default) ────────────────────────────────
    # ColorBrewer's GnBu (Green→Blue) sequential ramp — the colormap
    # elapid uses by default for its prediction outputs. Switching to
    # match elapid's convention keeps QMaxent's QGIS rendering visually
    # consistent with the matplotlib previews researchers see when
    # using elapid directly. Like viridis, GnBu is perceptually
    # ordered (lightness increases monotonically toward high values)
    # and color-blind friendly. 7 stops sampled evenly from
    # matplotlib's GnBu_r-equivalent so QGIS reproduces the same
    # gradient without depending on an optional cpt-city ramp.
    gnbu_stops = [
        (0.00, QColor(247, 252, 240)),  # palest green
        (0.17, QColor(224, 243, 219)),
        (0.33, QColor(204, 235, 197)),
        (0.50, QColor(168, 221, 181)),
        (0.67, QColor(123, 204, 196)),
        (0.83, QColor( 67, 162, 202)),
        (1.00, QColor(  8,  64, 129)),  # deepest blue
    ]
    span = vmax - vmin
    ramp_items = [
        QgsColorRampShader.ColorRampItem(vmin + t * span, color,
                                          f"{vmin + t * span:.3f}")
        for t, color in gnbu_stops
    ]
    color_ramp = QgsColorRampShader()
    color_ramp.setColorRampType(QgsColorRampShader.Interpolated)
    color_ramp.setColorRampItemList(ramp_items)
    color_ramp.setMinimumValue(vmin)
    color_ramp.setMaximumValue(vmax)

    shader = QgsRasterShader()
    shader.setMinimumValue(vmin)
    shader.setMaximumValue(vmax)
    shader.setRasterShaderFunction(color_ramp)

    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    renderer.setClassificationMin(vmin)
    renderer.setClassificationMax(vmax)
    # NOTE: QgsSingleBandPseudoColorRenderer does NOT have setContrastEnhancement.
    # That method belongs to QgsSingleBandGrayRenderer only.
    # The classification range above fully controls the colour mapping.

    layer.setRenderer(renderer)
    layer.triggerRepaint()
    return layer


# ── Categorical detection ──────────────────────────────────────────────────
#
# Following the convention of dismo, ENMeval, SDMtune, and maxent.jar, the
# user — not the plugin — declares which rasters are categorical. This
# helper reads ONLY explicit metadata hints (GDAL RAT thematic type,
# LAYER_TYPE='thematic', explicit color tables, GDAL category names); it
# never infers categorical status from data type or unique-value counts
# because those signals produce false positives on rasters like elevation
# that are stored as integers but are continuous.
def detect_categorical_hint(path: str) -> bool:
    """Return True if the GeoTIFF metadata explicitly declares the raster
    as categorical/thematic.

    Detection rules (all conservative — explicit declarations only):

    1. GDAL Raster Attribute Table (RAT) of thematic type. This is the
       official GDAL standard for categorical rasters and is what R's
       ``gdalraster`` package writes via ``setDefaultRAT(table_type =
       "thematic")``. See https://gdal.org for ``GDALRATTableType`` and
       ``GRTT_THEMATIC``.
    2. GDAL metadata key ``LAYER_TYPE='thematic'`` (ESRI/ArcGIS
       convention; widely recognised though not strictly part of the
       GDAL spec).
    3. The first band exposes non-empty ``CATEGORY_NAMES`` (GDAL
       standard ``GetCategoryNames``).
    4. The first band has an attached colour table that is paletted
       (``ColorInterp.palette``).

    Returns False on any I/O error; the caller then leaves the raster
    flagged as continuous, which is the safe default.
    """
    try:
        import rasterio
        from rasterio.enums import ColorInterp
        with rasterio.open(path) as src:
            # Rule 1: GDAL RAT thematic — the official categorical standard.
            # rasterio doesn't expose RAT directly so we drop to the GDAL
            # dataset handle. This is a no-op on drivers that don't carry
            # RAT (e.g. plain GeoTIFFs without one).
            try:
                from osgeo import gdal as _gdal  # noqa: F401
                ds = _gdal.Open(path)
                if ds is not None:
                    band = ds.GetRasterBand(1)
                    if band is not None:
                        rat = band.GetDefaultRAT()
                        if rat is not None:
                            # GRTT_THEMATIC == 0 ; GRTT_ATHEMATIC == 1
                            try:
                                if rat.GetTableType() == _gdal.GRTT_THEMATIC:
                                    return True
                            except Exception:
                                pass
            except ImportError:
                # osgeo.gdal is normally available wherever rasterio is,
                # but we tolerate its absence and fall through to the
                # other detection rules below.
                pass

            # Rule 2: LAYER_TYPE='thematic' (ESRI/ArcGIS convention)
            tags = {k.lower(): v for k, v in (src.tags() or {}).items()}
            if str(tags.get("layer_type", "")).strip().lower() == "thematic":
                return True
            band_tags = {k.lower(): v for k, v in (src.tags(1) or {}).items()}
            if str(band_tags.get("layer_type", "")).strip().lower() == "thematic":
                return True

            # Rule 3: GDAL category names on band 1
            if band_tags.get("category_names"):
                return True

            # Rule 4: paletted color interpretation
            try:
                if src.colorinterp[0] == ColorInterp.palette:
                    return True
            except Exception:
                pass
    except Exception as e:
        QgsMessageLog.logMessage(
            f"detect_categorical_hint: {e}", "QMaxent", Qgis.Warning
        )
        return False
    return False
