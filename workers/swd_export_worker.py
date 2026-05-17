"""Background worker: export data in formats consumable by maxent.jar.

Two output modes are supported, selected by the user via the
``format`` field of the worker configuration:

  "swd"             — Samples-With-Data CSV pair (presence.csv +
                       background.csv), each conforming to the schema
                       documented in maxent.jar's density/SampleSet.java
                       v3.4.4 (lines 35-68):

                           Species,Longitude,Latitude,<env1>,...,<envN>

                       Best for cross-checking the elapid backend against
                       maxent.jar on exactly the same point sample (no
                       re-sampling, no NoData ambiguity, no projection
                       raster output without an explicit projectionlayers
                       argument).

  "raster_samples"  — A samples CSV (Species,Longitude,Latitude — three
                       columns only) plus one ESRI ASCII Grid (.asc) per
                       environmental raster, written into a layers/
                       subfolder. This is the natural input format for
                       maxent.jar's main mode: the user passes
                       environmentallayers=<folder>/layers and maxent.jar
                       both fits the model and produces a projection
                       raster at the input grid resolution without
                       requiring a separate projectionlayers argument.

Both modes write a README.txt containing a generic, dataset-agnostic
maxent.jar command line using the maxent.jar defaults — no study-specific
recipes. The intent is that any QGIS user can invoke this export on
their own data and obtain a ready-to-run maxent.jar input directory.

Reference:
    Phillips, S.J., Anderson, R.P., Dudík, M., Schapire, R.E., & Blair,
        M.E. (2017). Opening the black box: an open-source release of
        Maxent. Ecography, 40, 887-893.
"""

import os
import traceback

import numpy as np
import pandas as pd
from qgis.PyQt.QtCore import QThread, pyqtSignal

from ..i18n import tr


class SWDExportWorker(QThread):
    progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str, dict)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._cfg = config
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _check(self, pct: int, msg: str):
        if self._cancelled:
            raise InterruptedError("cancelled")
        self.progress.emit(pct, msg)
        self.log.emit(msg)

    def run(self):
        try:
            result = self._execute()
            self.finished.emit(True, "Done", result)
        except InterruptedError:
            self.finished.emit(False, "Cancelled", {})
        except Exception as e:
            self.finished.emit(False, f"{e}\n{traceback.format_exc()}", {})

    # -----------------------------------------------------------------------
    # Main execution — dispatches on cfg["format"]
    # -----------------------------------------------------------------------

    def _execute(self) -> dict:
        fmt = self._cfg.get("format", "swd")
        if fmt == "swd":
            return self._execute_swd()
        elif fmt == "raster_samples":
            return self._execute_raster_samples()
        else:
            raise ValueError(f"Unknown export format '{fmt}'. Expected 'swd' or 'raster_samples'.")

    # -----------------------------------------------------------------------
    # Mode 1: SWD (Samples-With-Data CSV pair)
    # -----------------------------------------------------------------------

    def _execute_swd(self) -> dict:
        from ..bridge import elapid_bridge as eb

        cfg = self._cfg
        presence_gdf = cfg["presence_gdf"]
        raster_paths = cfg["raster_paths"]
        feature_names = cfg["feature_names"]
        categorical_indices = list(cfg.get("categorical_indices", []) or [])
        n_background = int(cfg.get("n_background", 10000))
        output_dir = cfg["output_dir"]
        species_name = cfg.get("species_name") or "species"
        bias_path = cfg.get("bias_path")

        species_name = _normalize_species_name(species_name)
        os.makedirs(output_dir, exist_ok=True)

        # ── Raster consistency check (fail-fast) ─────────────────────────
        self._check(5, "Checking raster consistency...")
        _ensure_raster_consistency(raster_paths)

        # ── Reproject presences to raster CRS if needed ──────────────────
        presence_gdf = _reproject_to_raster_crs(presence_gdf, raster_paths, log=self.log.emit)

        # ── Background sampling ──────────────────────────────────────────
        self._check(15, f"Sampling {n_background:,} background points...")
        if bias_path:
            bg_gs = eb.sample_bias_file(bias_path, count=n_background)
            self.log.emit("  → Bias-weighted sampling (Phillips 2009)")
        else:
            bg_gs = eb.sample_raster(raster_paths[0], count=n_background)
        self.log.emit(f"  → {len(bg_gs):,} background points sampled")

        # ── Annotation ───────────────────────────────────────────────────
        self._check(40, "Extracting raster values at presence points...")
        presence_ann = eb.annotate(presence_gdf, raster_paths, labels=feature_names, quiet=True)
        self._check(60, "Extracting raster values at background points...")
        background_ann = eb.annotate(bg_gs, raster_paths, labels=feature_names, quiet=True)

        # Drop NoData rows the same way the training pipeline does.
        presence_clean = presence_ann.dropna(subset=feature_names).reset_index(drop=True)
        background_clean = background_ann.dropna(subset=feature_names).reset_index(drop=True)
        n_pres = len(presence_clean)
        n_bg = len(background_clean)
        self.log.emit(f"  → After NoData removal: {n_pres} presence, {n_bg:,} background")
        if n_pres == 0:
            raise RuntimeError(
                tr(
                    "No valid presence points after removing rows with NoData "
                    "in the environmental rasters. Check that your presence "
                    "layer overlaps the raster extents."
                )
            )

        # ── Write SWD CSV files ──────────────────────────────────────────
        self._check(80, "Writing SWD files...")
        presence_path = os.path.join(output_dir, "presence.csv")
        background_path = os.path.join(output_dir, "background.csv")

        n_pres_written = _write_swd(
            presence_clean,
            presence_path,
            species_name,
            feature_names,
            categorical_indices,
        )
        n_bg_written = _write_swd(
            background_clean,
            background_path,
            "background",
            feature_names,
            categorical_indices,
        )

        # ── README with generic command line ─────────────────────────────
        self._check(95, "Writing README.txt...")
        readme_path = os.path.join(output_dir, "README.txt")
        _write_swd_readme(
            readme_path,
            n_presence=n_pres_written,
            n_background=n_bg_written,
            feature_names=feature_names,
            categorical_indices=categorical_indices,
        )

        self._check(100, "Done")
        return {
            "format": "swd",
            "presence_path": presence_path,
            "background_path": background_path,
            "readme_path": readme_path,
            "n_presence": n_pres_written,
            "n_background": n_bg_written,
            "output_dir": output_dir,
        }

    # -----------------------------------------------------------------------
    # Mode 2: Samples + Raster (samples CSV + per-variable .asc files)
    # -----------------------------------------------------------------------

    def _execute_raster_samples(self) -> dict:
        cfg = self._cfg
        presence_gdf = cfg["presence_gdf"]
        raster_paths = cfg["raster_paths"]
        feature_names = cfg["feature_names"]
        categorical_indices = list(cfg.get("categorical_indices", []) or [])
        output_dir = cfg["output_dir"]
        species_name = cfg.get("species_name") or "species"

        species_name = _normalize_species_name(species_name)
        os.makedirs(output_dir, exist_ok=True)
        samples_dir = os.path.join(output_dir, "samples")
        layers_dir = os.path.join(output_dir, "layers")
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(layers_dir, exist_ok=True)

        # ── Raster consistency check ─────────────────────────────────────
        self._check(5, "Checking raster consistency...")
        _ensure_raster_consistency(raster_paths)

        # ── Reproject presences to raster CRS ────────────────────────────
        presence_gdf = _reproject_to_raster_crs(presence_gdf, raster_paths, log=self.log.emit)

        # ── Write samples CSV (Species, Longitude, Latitude) ─────────────
        self._check(15, "Writing samples CSV...")
        samples_path = os.path.join(samples_dir, f"{species_name}.csv")
        n_pres = _write_samples_csv(presence_gdf, samples_path, species_name)
        self.log.emit(f"  → {n_pres} presence points written")
        if n_pres == 0:
            raise RuntimeError(
                tr(
                    "No valid presence points to export. Check that your "
                    "presence layer has features."
                )
            )

        # ── Convert each raster to ESRI ASCII Grid (.asc) ────────────────
        # maxent.jar accepts .asc as its native environmental layer
        # format. We write one .asc per input raster into layers/. The
        # files share a common grid (already verified above), so
        # maxent.jar can treat the layers/ folder as a self-consistent
        # stack.
        n_rasters = len(raster_paths)
        for i, (rpath, fname) in enumerate(zip(raster_paths, feature_names)):
            pct = 20 + int(75 * i / n_rasters)
            self._check(pct, f"Converting {fname} → ASCII Grid...")
            asc_path = os.path.join(layers_dir, f"{fname}.asc")
            _write_raster_as_asc(rpath, asc_path, is_categorical=(i in categorical_indices))

        # ── README with generic command line ─────────────────────────────
        self._check(96, "Writing README.txt...")
        readme_path = os.path.join(output_dir, "README.txt")
        _write_raster_samples_readme(
            readme_path,
            samples_subpath=os.path.relpath(samples_path, output_dir),
            layers_subpath=os.path.relpath(layers_dir, output_dir),
            n_presence=n_pres,
            feature_names=feature_names,
            categorical_indices=categorical_indices,
        )

        self._check(100, "Done")
        return {
            "format": "raster_samples",
            "samples_path": samples_path,
            "layers_dir": layers_dir,
            "readme_path": readme_path,
            "n_presence": n_pres,
            "n_rasters": n_rasters,
            "output_dir": output_dir,
        }


# ---------------------------------------------------------------------------
# Module-level helpers (kept outside the class for unit-testability)
# ---------------------------------------------------------------------------


def _normalize_species_name(name: str) -> str:
    """Strip whitespace, replace commas, collapse internal whitespace.

    maxent.jar's SampleSet.read() splits on commas, and a species name
    with embedded spaces also confuses ENMeval-style downstream tooling
    that uses the name as a filename root.
    """
    name = (name or "species").strip().replace(",", "_")
    return "_".join(name.split())


def _ensure_raster_consistency(raster_paths):
    """Raise RuntimeError if the raster stack does not share a grid."""
    from ..bridge.raster_bridge import check_raster_consistency

    consistency = check_raster_consistency(raster_paths)
    if consistency.get("is_consistent", False):
        return
    mismatches = []
    if not consistency["crs_uniform"]:
        mismatches.append("CRS")
    if not consistency["extent_uniform"]:
        mismatches.append("extent")
    if not consistency["resolution_uniform"]:
        mismatches.append("resolution")
    raise RuntimeError(
        tr(
            "Environmental rasters do not share a common grid "
            '({mismatches} differ). Run "Check Raster Consistency" and '
            '"Harmonize to Folder…" in the ① Data tab before exporting.'
        ).format(mismatches=", ".join(mismatches))
    )


def _reproject_to_raster_crs(presence_gdf, raster_paths, log=None):
    """Reproject presences to the raster CRS in place; return updated gdf.

    This matches what the training pipeline does — we use the raster
    CRS as the canonical coordinate space for the export so that the
    samples and the (eventual) ASCII grid headers agree.
    """
    from ..bridge.raster_bridge import get_raster_crs

    raster_crs = get_raster_crs(raster_paths[0])
    if raster_crs is None or presence_gdf.crs is None:
        return presence_gdf
    try:
        same = presence_gdf.crs.to_string() == str(raster_crs)
    except Exception:
        same = False
    if same:
        return presence_gdf
    if log is not None:
        log(
            f"  ⚠ CRS mismatch — presence={presence_gdf.crs}, "
            f"raster={raster_crs}; reprojecting presences."
        )
    return presence_gdf.to_crs(raster_crs)


def _write_swd(gdf, path: str, species: str, feature_names: list, categorical_indices: list) -> int:
    """Write a SWD CSV. Returns the number of rows written."""
    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values

    out = pd.DataFrame(
        {
            "Species": [species] * len(gdf),
            "Longitude": xs,
            "Latitude": ys,
        }
    )
    for i, nm in enumerate(feature_names):
        col = gdf[nm].values
        if i in categorical_indices:
            # Round before cast — float-typed rasters (uint8 read as
            # float32 by rasterio) collapse back to their original
            # integer codes here.
            out[nm] = np.rint(col).astype(np.int64)
        else:
            out[nm] = col

    out.to_csv(path, index=False, float_format="%.6f")
    return len(out)


def _write_samples_csv(presence_gdf, path: str, species_name: str) -> int:
    """Write the Species,Longitude,Latitude CSV for raster-mode export.

    Unlike SWD, no environmental values are extracted here — maxent.jar
    will resolve them from the layers/ folder at training time using
    its own bilinear sampling.
    """
    xs = presence_gdf.geometry.x.values
    ys = presence_gdf.geometry.y.values
    df = pd.DataFrame(
        {
            "Species": [species_name] * len(presence_gdf),
            "Longitude": xs,
            "Latitude": ys,
        }
    )
    df.to_csv(path, index=False, float_format="%.6f")
    return len(df)


def _write_raster_as_asc(src_path: str, dst_path: str, is_categorical: bool) -> None:
    """Convert a raster to ESRI ASCII Grid (.asc).

    ASC format requires a header followed by row-major values:

        ncols         <n>
        nrows         <n>
        xllcorner     <float>          # lower-left X (cell corner)
        yllcorner     <float>          # lower-left Y (cell corner)
        cellsize      <float>          # uniform cell size
        NODATA_value  <value>
        <data rows, top to bottom>

    maxent.jar expects square cells. The transform from the source
    raster must be axis-aligned with positive pixel width — we verify
    this and raise rather than producing a silently-wrong header.
    Categorical layers are written as integers (with -9999 NODATA);
    continuous as floats (with -9999.0 NODATA).
    """
    import rasterio

    with rasterio.open(src_path) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata

        # Verify the transform is axis-aligned (no rotation/shear) and
        # has uniform positive cell size. maxent.jar will silently
        # misalign features if the .asc header doesn't match the
        # actual pixel layout.
        a, b = transform.a, transform.b
        c, d = transform.c, transform.d
        e, f = transform.e, transform.f
        if b != 0 or d != 0:
            raise RuntimeError(
                f"Cannot export {src_path}: raster transform is not "
                f"axis-aligned (shear/rotation present). Reproject first."
            )
        if abs(abs(a) - abs(e)) > 1e-6 * abs(a):
            raise RuntimeError(
                f"Cannot export {src_path}: raster has non-square cells "
                f"(dx={a}, dy={e}). Resample first."
            )

        cellsize = float(abs(a))
        ncols = int(src.width)
        nrows = int(src.height)
        # rasterio's transform: c = X of upper-left pixel CORNER,
        # f = Y of upper-left pixel CORNER (for the typical
        # negative-e north-up case). xllcorner / yllcorner are the
        # lower-left CORNER of the lower-left pixel.
        xllcorner = float(c)
        if e < 0:  # north-up (typical)
            yllcorner = float(f + e * nrows)  # e is negative → subtraction
        else:
            yllcorner = float(f)

    # NoData sentinel — ASC convention is -9999 for both int and float,
    # and we apply that whether or not the source raster has a NoData
    # set (some uint8 categorical rasters don't carry one).
    NODATA_INT = -9999
    NODATA_FLOAT = -9999.0

    if is_categorical:
        if data.dtype.kind == "f":
            mask_nan = np.isnan(data)
        else:
            mask_nan = np.zeros(data.shape, dtype=bool)
        out = np.where(mask_nan, NODATA_INT, np.rint(data).astype(np.int64))
        if nodata is not None:
            try:
                mask_nd = data == nodata
                out = np.where(mask_nd, NODATA_INT, out)
            except Exception:
                pass
        nd_str = str(NODATA_INT)
        fmt = "%d"
    else:
        arr = data.astype(np.float64)
        mask = np.isnan(arr)
        if nodata is not None:
            try:
                mask = mask | (arr == nodata)
            except Exception:
                pass
        out = np.where(mask, NODATA_FLOAT, arr)
        nd_str = "-9999"
        fmt = "%.6g"

    # Write header + data. We write line-by-line to keep memory low
    # on large rasters (e.g. country-scale at 30 m).
    with open(dst_path, "w", encoding="ascii") as fp:
        fp.write(f"ncols         {ncols}\n")
        fp.write(f"nrows         {nrows}\n")
        fp.write(f"xllcorner     {xllcorner:.6f}\n")
        fp.write(f"yllcorner     {yllcorner:.6f}\n")
        fp.write(f"cellsize      {cellsize:.6f}\n")
        fp.write(f"NODATA_value  {nd_str}\n")
        for row in out:
            fp.write(" ".join(fmt % v for v in row))
            fp.write("\n")


# ---------------------------------------------------------------------------
# README writers
# ---------------------------------------------------------------------------


def _toggle_block(feature_names, categorical_indices) -> str:
    """Render the togglelayertype= flags for the README command line."""
    cats = [feature_names[i] for i in categorical_indices]
    if not cats:
        return ""
    block = " \\\n    ".join(f"togglelayertype={c}" for c in cats)
    return "    " + block + " \\\n    "


def _write_swd_readme(path, n_presence, n_background, feature_names, categorical_indices):
    """Generic README for the SWD output (no study-specific recipe)."""
    cat_str = ", ".join(feature_names[i] for i in categorical_indices) or "(none)"
    toggle = _toggle_block(feature_names, categorical_indices)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            _SWD_README.format(
                n_presence=n_presence,
                n_background=n_background,
                n_features=len(feature_names),
                feature_list=", ".join(feature_names),
                categorical_list=cat_str,
                toggle_block=toggle,
            )
        )


def _write_raster_samples_readme(
    path, samples_subpath, layers_subpath, n_presence, feature_names, categorical_indices
):
    """Generic README for the samples + raster output."""
    cat_str = ", ".join(feature_names[i] for i in categorical_indices) or "(none)"
    toggle = _toggle_block(feature_names, categorical_indices)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            _RASTER_README.format(
                samples_subpath=samples_subpath.replace("\\", "/"),
                layers_subpath=layers_subpath.replace("\\", "/"),
                n_presence=n_presence,
                n_features=len(feature_names),
                feature_list=", ".join(feature_names),
                categorical_list=cat_str,
                toggle_block=toggle,
            )
        )


# ---------------------------------------------------------------------------
# README templates — generic, dataset-agnostic
# ---------------------------------------------------------------------------

_SWD_README = """\
QMaxent export — SWD format for maxent.jar
==========================================

These files were generated by QMaxent
(https://github.com/osgeokr/qmaxent) for use with the standalone
maxent.jar implementation (Phillips et al. 2017).

Files in this directory:
  presence.csv     ({n_presence} rows; SWD format)
  background.csv   ({n_background} rows; SWD format)
  README.txt       (this file)

Schema (matches maxent.jar density/SampleSet.java:35-68 of v3.4.4):
  Species,Longitude,Latitude,<env1>,<env2>,...,<envN>

Environmental variables ({n_features}):
  {feature_list}

Categorical variables (must be flagged via togglelayertype):
  {categorical_list}

----------------------------------------------------------------------
Run maxent.jar with default settings
----------------------------------------------------------------------

  java -mx2048m -jar maxent.jar \\
    samplesfile=presence.csv \\
    environmentallayers=background.csv \\
    outputdirectory=maxent_output \\
{toggle_block}autorun=true visible=false

This uses maxent.jar's defaults: autofeature=true (sample-size based
feature selection), betamultiplier=1, outputformat=Cloglog,
replicates=1. The `samplesfile=` / `environmentallayers=` pair is
how maxent.jar recognises SWD mode (Phillips et al. 2017).

To override defaults, append parameters to the command line, e.g.:

  betamultiplier=2 outputformat=Logistic replicates=10
  replicatetype=Subsample randomtestpoints=25 jackknife=true
  responsecurves=true

----------------------------------------------------------------------
Notes
----------------------------------------------------------------------

* Projection raster: SWD-mode runs do NOT produce a habitat
  suitability raster by default. The background.csv file contains
  point-by-point values, not a continuous surface. To get a raster
  output from this command line, append:

      projectionlayers=<path/to/raster/folder>

  pointing to the directory containing the source rasters
  (in ESRI ASCII Grid or GeoTIFF format).

  If you want a projection raster without managing the
  projectionlayers argument yourself, QMaxent's "Samples + Raster"
  export format (the default) produces .asc layers in a single step
  and maxent.jar projects automatically.

* Coordinates are in the raster's native CRS. The Species/Longitude/
  Latitude header is a maxent.jar naming convention; the values are
  not interpreted as geographic degrees unless the input rasters are
  in EPSG:4326.

* For deterministic runs, add randomseed=false (or fix a specific
  seed via Java's RNG settings).

Reference:
  Phillips, S.J., Anderson, R.P., Dudík, M., Schapire, R.E., & Blair,
  M.E. (2017). Opening the black box: an open-source release of
  Maxent. Ecography, 40, 887-893.
"""


_RASTER_README = """\
QMaxent export — Samples + Raster for maxent.jar
================================================

These files were generated by QMaxent
(https://github.com/osgeokr/qmaxent) for use with the standalone
maxent.jar implementation (Phillips et al. 2017).

Files in this directory:
  {samples_subpath}     (presence locations: Species, Longitude, Latitude)
  {layers_subpath}/<var>.asc   ({n_features} ESRI ASCII Grid files,
                                one per environmental variable)
  README.txt            (this file)

Each .asc file shares the same grid (extent, resolution, CRS) — this
is verified by QMaxent before export. Categorical variables are
written as integer codes; continuous variables as floats. NoData
cells use the value -9999 (ASC convention).

Environmental variables ({n_features}):
  {feature_list}

Categorical variables (must be flagged via togglelayertype):
  {categorical_list}

----------------------------------------------------------------------
Run maxent.jar with default settings
----------------------------------------------------------------------

  java -mx2048m -jar maxent.jar \\
    samplesfile={samples_subpath} \\
    environmentallayers={layers_subpath} \\
    outputdirectory=maxent_output \\
{toggle_block}autorun=true visible=false

In this mode (samples + environmentallayers folder), maxent.jar
both fits the model AND produces a projection raster covering the
extent of the input layers — no separate projectionlayers argument
is needed. Output: maxent_output/<species>.asc plus the standard
HTML report and statistics.

To override defaults, append parameters:

  betamultiplier=2 outputformat=Logistic replicates=10
  replicatetype=Subsample randomtestpoints=25 jackknife=true
  responsecurves=true

----------------------------------------------------------------------
Notes
----------------------------------------------------------------------

* Background sample: by default maxent.jar will draw 10,000 random
  background points from the valid extent of the layers folder.
  This is the standard SDM convention. To control it, add:

      maximumbackground=<N>

* Coordinates in the samples CSV are in the rasters' native CRS.

* The .asc files are larger than the original rasters (ASCII text
  vs binary). If disk space matters, you can also place GeoTIFFs in
  the layers folder — recent maxent.jar versions accept .tif as well.

* For deterministic runs, add randomseed=false.

Reference:
  Phillips, S.J., Anderson, R.P., Dudík, M., Schapire, R.E., & Blair,
  M.E. (2017). Opening the black box: an open-source release of
  Maxent. Ecography, 40, 887-893.
"""
