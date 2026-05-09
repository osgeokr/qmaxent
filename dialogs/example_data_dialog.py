"""Example dataset download dialog.

Downloads a small canonical SDM example dataset directly from its
canonical URL into a local folder. After successful download the
dialog adds the rasters and presence layer to the active QGIS
project so the user can immediately train a model.

Following the convention of dismo, SDMtune, ENMeval, and biomod2 —
all of which ship example data with their R packages — this gives
QMaxent users a working example without any external data hunting.

Datasets are described in DATASET_REGISTRY below; adding a new one
is a matter of declaring the URLs, file names, and presence-layer
setup. We deliberately do NOT route through elapid.download_sample_data
because that function in elapid 1.0.3 silently no-ops for the
``"bradypus"`` name (the function only defines fnames inside the
``"ariolimax"`` branch and then iterates the undefined variable);
direct downloads also let us host the canonical Bradypus data at
its de-facto archival home (the cran/dismo GitHub mirror, which
mirrors CRAN's dismo package contents).
"""

import os
import traceback
import urllib.request

from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton,
    QButtonGroup, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QProgressBar, QGroupBox,
)

from ..i18n import tr, tooltip


# ─── Dataset registry ────────────────────────────────────────────────────
#
# Each entry describes a canonical SDM example dataset. To add a new
# dataset (e.g. a Korean SDM example for v0.2), append another dict
# here. The download worker iterates `files` in order and reports
# progress per file; `presence_file` is the vector that will be
# loaded as the presence layer in QGIS, and `categorical_files` are
# raster filenames that should be flagged as categorical when the
# user adds them to the QMaxent raster list (currently informational
# only — the user still toggles the categorical/continuous switch
# manually in the ① Data tab, matching dismo/SDMtune convention).

DATASET_REGISTRY = {
    "bradypus": {
        "label":    "Bradypus variegatus (Phillips et al. 2006 standard)",
        "tooltip":  (
            "Brown-throated three-toed sloth occurrence + 8 bioclimatic "
            "rasters and 1 categorical biome raster covering South "
            "America. The canonical Maxent benchmark dataset, mirrored "
            "from the CRAN dismo R package."
        ),
        # Source: cran/dismo GitHub mirror. Stable, public, MIT-style
        # data redistribution (dismo is GPL-3 and ships these as
        # examples).
        "base_url": (
            "https://raw.githubusercontent.com/cran/dismo/master/inst/ex"
        ),
        "files": [
            "bradypus.csv",
            "bio1.grd", "bio1.gri",
            "bio5.grd", "bio5.gri",
            "bio6.grd", "bio6.gri",
            "bio7.grd", "bio7.gri",
            "bio8.grd", "bio8.gri",
            "bio12.grd", "bio12.gri",
            "bio16.grd", "bio16.gri",
            "bio17.grd", "bio17.gri",
            "biome.grd", "biome.gri",
        ],
        "presence_file":     "bradypus.csv",
        "presence_x_field":  "lon",
        "presence_y_field":  "lat",
        "presence_crs":      "EPSG:4326",
        # Raster file names (without sidecar) that represent categorical
        # variables. QMaxent uses the GDAL "thematic" tag for automatic
        # detection; this list is informational only.
        "categorical_files": ["biome.grd"],
    },
    "ariolimax": {
        "label":    "Ariolimax (banana slug, elapid default)",
        "tooltip":  (
            "Banana slug occurrence + 6 cloud cover, leaf area index, "
            "and surface temperature rasters covering California. "
            "elapid's built-in example dataset."
        ),
        # Source: elapid's own example data hosting (Christopher Anderson's
        # GitHub Pages site).
        "base_url": (
            "https://earth-chris.github.io/images/research"
        ),
        "files": [
            "ariolimax-ca.gpkg",
            "ca-cloudcover-mean.tif",
            "ca-cloudcover-stdv.tif",
            "ca-leafareaindex-mean.tif",
            "ca-leafareaindex-stdv.tif",
            "ca-surfacetemp-mean.tif",
            "ca-surfacetemp-stdv.tif",
        ],
        "presence_file":     "ariolimax-ca.gpkg",
        "presence_x_field":  None,    # geometry column, no x/y fields
        "presence_y_field":  None,
        "presence_crs":      None,    # embedded in the gpkg
        "categorical_files": [],
    },
}


# ─── Background download worker ──────────────────────────────────────────

class _DownloadWorker(QThread):
    """Sequentially downloads each file in a dataset spec.

    Signals:
        progress(int, str): 0-100% across the whole dataset, plus the
            data unit currently being fetched.
        finished(bool, str, list): success flag, message, and the list
            of *actual* on-disk file basenames after any post-download
            conversion. Empty list on failure or cancel.

    The worker reports progress in *logical data units* rather than
    raw HTTP files: a multi-file raster like ``bio1.grd`` + ``bio1.gri``
    counts as one unit (a single raster) so the progress bar matches
    the user's mental model (9 rasters + 1 vector = 10 units, not the
    19 HTTP downloads happening underneath). The ``unit_groups`` arg
    expresses this grouping; if omitted, every file is its own unit
    (the right default for datasets that ship single-file rasters
    like Ariolimax's GeoTIFFs).

    After download, ``.grd`` rasters are post-processed to GeoTIFF if
    rasterio is importable. Conversion is best-effort — if rasterio
    isn't available (e.g. the user hasn't installed plugin dependencies
    yet), we log a warning and leave the originals in place; the
    plugin can still read them via the RRASTER GDAL driver.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, list)

    def __init__(self, base_url: str, files: list, dst_dir: str,
                 unit_groups: list = None,
                 categorical_files: list = None,
                 parent=None):
        super().__init__(parent)
        self._base_url = base_url
        self._files    = list(files)
        self._dst_dir  = dst_dir
        # unit_groups: list of (label, [filenames]) describing the
        # progress-bar unit each download belongs to. When None, build
        # a 1:1 mapping so every file is its own unit.
        if unit_groups is None:
            self._unit_groups = [(f, [f]) for f in self._files]
        else:
            self._unit_groups = list(unit_groups)
        # Filenames the registry tagged as categorical. After .grd →
        # .tif conversion we tag the matching .tif with the GDAL
        # "thematic" metadata so QMaxent's categorical detector picks
        # them up automatically — same as if the user supplied a
        # purpose-built categorical GeoTIFF.
        self._categorical_files = set(categorical_files or [])
        self._cancel   = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            n_units = len(self._unit_groups)
            for unit_idx, (label, fnames) in enumerate(self._unit_groups):
                if self._cancel:
                    self.finished.emit(False, "Cancelled", [])
                    return
                pct_before = int((unit_idx / n_units) * 100)
                self.progress.emit(
                    pct_before,
                    f"Downloading {label} ({unit_idx + 1}/{n_units})",
                )
                # Fetch every file in the unit. Sidecar files (.gri,
                # .aux.xml, .prj …) are downloaded silently here —
                # the user only sees the unit-level message above.
                for fname in fnames:
                    if self._cancel:
                        self.finished.emit(False, "Cancelled", [])
                        return
                    url = f"{self._base_url}/{fname}"
                    dst = os.path.join(self._dst_dir, fname)
                    # Use a User-Agent because some hosts (notably the
                    # raw.githubusercontent.com CDN) reject default
                    # urllib clients.
                    req = urllib.request.Request(
                        url,
                        headers={"User-Agent": "qmaxent/0.1.0"},
                    )
                    with urllib.request.urlopen(req, timeout=30) as r, \
                         open(dst, "wb") as f:
                        while True:
                            if self._cancel:
                                self.finished.emit(False, "Cancelled", [])
                                return
                            chunk = r.read(64 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)

            # ── Post-download: convert .grd → .tif when possible ────────
            self.progress.emit(100, "Converting rasters to GeoTIFF...")
            final_files = self._maybe_convert_grd_to_tiff()

            self.progress.emit(100, "Download complete")
            self.finished.emit(True, "Done", final_files)
        except Exception as e:
            self.finished.emit(
                False, f"{e}\n{traceback.format_exc()}", []
            )

    def _maybe_convert_grd_to_tiff(self) -> list:
        """Convert every downloaded .grd raster to a GeoTIFF in place.

        Returns the list of final file basenames the dialog should add
        to the QGIS project — with .grd entries replaced by their .tif
        equivalents and the .gri sidecars dropped from the list.

        Falls back gracefully (returns the original file list) when
        rasterio cannot be imported, which happens on a fresh install
        before the user has run the dependency installer. The original
        .grd + .gri pair stays on disk and remains usable through
        GDAL's RRASTER driver.
        """
        try:
            import rasterio
            from rasterio.enums import Resampling  # noqa: F401  (probe import)
        except ImportError:
            # Convert step is purely cosmetic — we keep going with
            # the .grd originals so the user can still load the
            # dataset. The dialog surfaces an info message about
            # this in _on_finished().
            return list(self._files)

        out_files = []
        for fname in self._files:
            if not fname.lower().endswith(".grd"):
                # .gri sidecars are dropped from the user-visible
                # list because they'll be deleted alongside their
                # parent .grd. Other extensions (csv, gpkg, tif)
                # pass through.
                if fname.lower().endswith(".gri"):
                    continue
                out_files.append(fname)
                continue

            grd_path = os.path.join(self._dst_dir, fname)
            tif_name = fname[:-4] + ".tif"
            tif_path = os.path.join(self._dst_dir, tif_name)
            gri_path = grd_path[:-4] + ".gri"
            is_categorical = fname in self._categorical_files
            try:
                with rasterio.open(grd_path) as src:
                    profile = src.profile.copy()
                    profile.update(
                        driver="GTiff",
                        compress="lzw",      # standard SDM convention
                        tiled=False,         # small example rasters
                    )
                    data = src.read()
                    src_tags = src.tags()
                    band_tags = src.tags(1)
                with rasterio.open(tif_path, "w", **profile) as dst:
                    dst.write(data)
                    # Preserve any tags the source had — including
                    # GDAL's LAYER_TYPE=thematic on .grd categorical
                    # rasters like biome.grd.
                    if src_tags:
                        dst.update_tags(**src_tags)
                    if band_tags:
                        dst.update_tags(1, **band_tags)
                    # If the registry told us this is categorical
                    # but the source .grd didn't carry the
                    # thematic flag (often the case for dismo's
                    # biome.grd, which uses a colour-table hint
                    # instead), set it explicitly so QMaxent's
                    # auto-detector recognises the .tif.
                    if is_categorical:
                        dst.update_tags(LAYER_TYPE="thematic")

                # Original .grd + .gri are now redundant; remove them
                # so the user's data folder doesn't accumulate two
                # copies of every raster. Best-effort — failure here
                # leaves harmless duplicates.
                for old in (grd_path, gri_path):
                    try:
                        if os.path.exists(old):
                            os.remove(old)
                    except OSError:
                        pass
                out_files.append(tif_name)
            except Exception:
                # Conversion of a single raster failed — keep the
                # .grd in place so the user still has the data, and
                # surface the original name so the dialog loads it
                # via the RRASTER driver.
                out_files.append(fname)

        return out_files


# Sidecar file extensions that *belong to* their main raster — they
# should be grouped under the same progress unit so the user sees
# "9 rasters" rather than "9 rasters + 9 invisible companion files".
# .gri  : R raster binary data (paired with .grd)
# .aux.xml : GDAL auxiliary metadata (CRS, statistics, RAT)
# .prj  : ESRI projection (paired with .shp / .asc)
# .ovr  : pyramids / overviews
_SIDECAR_EXTS = (".gri", ".aux.xml", ".prj", ".ovr")


def _group_files_into_units(files: list) -> list:
    """Group sidecar files under their main file for progress reporting.

    Returns a list of ``(label, [filenames])`` tuples where each unit
    represents one logical data item (one raster, one vector, one
    table). For a Bradypus-style .grd + .gri pair, the .grd becomes
    the main file and .gri is folded into the same unit; for single-
    file rasters like Ariolimax's GeoTIFFs each file is its own unit.

    The label is the main file's stem (no extension), which is what
    QGIS shows as the layer name and matches the user's mental model.
    """
    units = []  # list of [main_stem, [files]]
    for f in files:
        # Detect compound sidecar extensions like ".aux.xml" first.
        lower = f.lower()
        sidecar_ext = next(
            (e for e in _SIDECAR_EXTS if lower.endswith(e)),
            None,
        )
        if sidecar_ext is not None:
            # Strip the sidecar suffix to recover the main file's
            # stem. For .aux.xml this leaves the full main filename
            # (e.g. "dem.tif.aux.xml" → "dem.tif"); we then strip
            # one more extension so it matches whatever the registry
            # called the main file.
            without_sidecar = f[:-len(sidecar_ext)]
            stem = os.path.splitext(without_sidecar)[0]
            # Also try the un-stripped name as a fallback (handles
            # cases like .gri where the sidecar ext is the whole
            # extension and there's nothing more to strip).
            candidate_stems = {stem, without_sidecar}
            attached = False
            for u in units:
                if u[0] in candidate_stems:
                    u[1].append(f)
                    attached = True
                    break
            if attached:
                continue
            # Sidecar with no main file yet — keep it as its own unit
            # (better visible orphan than silently swallowed).
        # Main file: new unit. Use stem (filename minus extension) as label.
        stem = os.path.splitext(f)[0]
        units.append([stem, [f]])
    # Convert inner lists to tuples for immutability downstream
    return [(label, list(fs)) for label, fs in units]


# ─── Dialog ──────────────────────────────────────────────────────────────

class ExampleDataDialog(QDialog):
    """Modal dialog: pick a dataset, choose a destination, download.

    The download runs in a QThread so the dialog stays responsive and
    the user can cancel mid-download. After completion the dataset's
    rasters and presence layer are added to the active QGIS project.
    """

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self._iface = iface
        self._worker = None

        self.setWindowTitle(tr("Download Example Dataset"))
        self.setModal(True)
        self.setMinimumWidth(540)

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 14); v.setSpacing(10)

        intro = QLabel(tr(
            "Downloads a small canonical SDM example dataset directly "
            "from its archival URL. After download, layers will be "
            "added to the current QGIS project automatically."
        ))
        intro.setWordWrap(True)
        v.addWidget(intro)

        # Dataset choice -------------------------------------------------
        ds_grp = QGroupBox(tr("Dataset"))
        dsv = QVBoxLayout(ds_grp)

        self._radio_group = QButtonGroup(self)
        self._radios = {}
        for i, (key, spec) in enumerate(DATASET_REGISTRY.items()):
            r = QRadioButton(tr(spec["label"]))
            r.setToolTip(tooltip(tr(spec["tooltip"])))
            if i == 0:
                r.setChecked(True)
            dsv.addWidget(r)
            self._radio_group.addButton(r, i)
            self._radios[key] = r

        v.addWidget(ds_grp)

        # Destination directory ------------------------------------------
        dst_row = QHBoxLayout()
        dst_row.addWidget(QLabel(tr("Save to:")))
        self._dst_edit = QLineEdit()
        # Default: ~/qmaxent_examples
        default_dir = os.path.join(
            os.path.expanduser("~"), "qmaxent_examples"
        )
        self._dst_edit.setText(default_dir)
        dst_row.addWidget(self._dst_edit, stretch=1)
        browse_btn = QPushButton("…"); browse_btn.setMaximumWidth(34)
        browse_btn.clicked.connect(self._on_browse)
        dst_row.addWidget(browse_btn)
        v.addLayout(dst_row)

        # Progress + status ----------------------------------------------
        # Determinate progress bar — the worker reports 0-100% across the
        # full file list, so the user sees real progress instead of a
        # spinning placeholder.
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.hide()
        v.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setStyleSheet("color: gray;")
        self._status.setWordWrap(True)
        v.addWidget(self._status)

        # Buttons --------------------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._cancel_btn = QPushButton(tr("Cancel"))
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)
        self._download_btn = QPushButton(tr("Download"))
        self._download_btn.setDefault(True)
        self._download_btn.clicked.connect(self._on_download)
        btn_row.addWidget(self._download_btn)
        v.addLayout(btn_row)

    # ── Slots ────────────────────────────────────────────────────────────
    def _on_browse(self):
        d = QFileDialog.getExistingDirectory(
            self, tr("Select destination directory"), self._dst_edit.text()
        )
        if d:
            self._dst_edit.setText(d)

    def _selected_key(self):
        for key, radio in self._radios.items():
            if radio.isChecked():
                return key
        return None

    def _on_download(self):
        key = self._selected_key()
        if key is None or key not in DATASET_REGISTRY:
            QMessageBox.warning(
                self, tr("Download Example Dataset"),
                tr("Please select a dataset.")
            )
            return
        spec = DATASET_REGISTRY[key]

        dst_root = self._dst_edit.text().strip()
        if not dst_root:
            QMessageBox.warning(
                self, tr("Download Example Dataset"),
                tr("Please choose a destination directory.")
            )
            return

        # Each dataset gets its own subfolder under the chosen root —
        # mirrors elapid.download_sample_data's layout so users coming
        # from elapid see a familiar directory structure.
        dst_dir = os.path.join(dst_root, key)
        try:
            os.makedirs(dst_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(
                self, tr("Download Example Dataset"),
                tr("Could not create destination directory:\n{e}").format(e=e)
            )
            return

        self._progress.setValue(0); self._progress.show()
        self._status.setText(tr("Starting download..."))
        self._download_btn.setEnabled(False)
        # Cancel button stays enabled and is now wired to abort the worker.

        # Group multi-file rasters (.grd + .gri pair) into a single
        # progress unit so the bar tracks data items, not byte-stream
        # sidecar files.
        unit_groups = _group_files_into_units(spec["files"])

        self._worker = _DownloadWorker(
            base_url=spec["base_url"],
            files=spec["files"],
            dst_dir=dst_dir,
            unit_groups=unit_groups,
            categorical_files=spec.get("categorical_files", []),
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(
            lambda ok, msg, files: self._on_finished(
                ok, msg, files, key, dst_dir
            )
        )
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        self._status.setText(msg)

    def _on_cancel(self):
        # Two roles for Cancel: abort an in-flight download, or close
        # the dialog when nothing is running.
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._status.setText(tr("Cancelling..."))
        else:
            self.reject()

    def _on_finished(self, ok: bool, msg: str, final_files: list,
                     key: str, dst_dir: str):
        self._worker = None
        self._download_btn.setEnabled(True)

        if not ok:
            self._progress.hide()
            self._status.setText(tr("Download failed."))
            QMessageBox.critical(
                self, tr("Download Example Dataset"),
                tr("Download failed:\n{msg}").format(msg=msg)
            )
            return

        # Detect whether the conversion step ran. If `final_files`
        # still contains .grd entries, rasterio wasn't available so
        # the worker left the originals in place. We surface this in
        # an info message rather than failing — the data is fully
        # usable through GDAL's RRASTER driver.
        had_grd_input = any(
            f.lower().endswith(".grd") for f in DATASET_REGISTRY[key]["files"]
        )
        kept_grd = any(
            f.lower().endswith(".grd") for f in final_files
        )
        conversion_skipped = had_grd_input and kept_grd

        # Add layers to the active project ------------------------------
        try:
            # Build a transient spec that reflects the on-disk reality
            # after conversion (e.g., bio1.grd → bio1.tif).
            effective_spec = dict(DATASET_REGISTRY[key])
            effective_spec["files"] = final_files
            self._add_layers_to_project(dst_dir, effective_spec)
        except Exception as e:
            QMessageBox.warning(
                self, tr("Download Example Dataset"),
                tr(
                    "Files were downloaded but could not be added to "
                    "the project automatically:\n{e}"
                ).format(e=e)
            )

        if conversion_skipped:
            QMessageBox.information(
                self, tr("Download Example Dataset"),
                tr(
                    "Example dataset '{ds}' is ready in:\n{path}\n\n"
                    "Note: rasters were left in their original .grd "
                    "format because rasterio is not available yet. "
                    "Install plugin dependencies (Plugins → QMaxent → "
                    "QMaxent Dependencies) and re-download to get "
                    "GeoTIFF copies. The .grd files work fine for now."
                ).format(ds=key, path=dst_dir)
            )
        else:
            QMessageBox.information(
                self, tr("Download Example Dataset"),
                tr(
                    "Example dataset '{ds}' is ready in:\n{path}\n\n"
                    "Open the QMaxent Analysis dock to start training."
                ).format(ds=key, path=dst_dir)
            )
        self.accept()

    # ── Layer loading ────────────────────────────────────────────────────
    def _add_layers_to_project(self, dst_dir: str, spec: dict):
        """Load the dataset's rasters and presence layer into QGIS.

        Raster files are added as QgsRasterLayer; the presence file is
        added either as an OGR vector (.gpkg, .shp, .geojson) or as a
        delimited-text point layer (.csv with explicit lon/lat fields).
        Sidecar files (.gri for .grd, .aux.xml, etc.) are skipped — they
        are loaded transparently by GDAL when the main raster opens.
        """
        from qgis.core import (
            QgsProject, QgsRasterLayer, QgsVectorLayer, QgsSymbol,
        )
        from qgis.PyQt.QtGui import QColor

        if not os.path.isdir(dst_dir):
            return

        # 1) Rasters. We iterate the spec's `files` list (rather than
        #    os.listdir) so the order in QGIS matches the order on
        #    disk — useful when the user wants the bio* layers to
        #    appear in numeric order.
        raster_exts = (".tif", ".tiff", ".grd", ".img", ".asc", ".vrt")
        for fname in spec["files"]:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in raster_exts:
                continue
            full = os.path.join(dst_dir, fname)
            if not os.path.isfile(full):
                continue
            display_name = os.path.splitext(fname)[0]
            lyr = QgsRasterLayer(full, display_name)
            if lyr.isValid():
                QgsProject.instance().addMapLayer(lyr)

        # 2) Presence layer.
        #
        # We convert presence CSVs to GeoPackage on first download,
        # then load every example dataset through the same OGR path.
        # That removes the OGR delimited-text URI's Windows-specific
        # quirks (the file:/// vs /C:/ path discrepancy that
        # occasionally lost the loader) and gives users a layer that
        # already lives in a GIS-standard, editable format — they can
        # save attribute edits without re-export. The conversion runs
        # exactly once per dataset folder; subsequent loads reuse the
        # cached .gpkg.
        pres_fname = spec.get("presence_file")
        if not pres_fname:
            return
        pres_path = os.path.join(dst_dir, pres_fname)
        if not os.path.isfile(pres_path):
            return
        pres_ext = os.path.splitext(pres_fname)[1].lower()
        pres_name = os.path.splitext(pres_fname)[0]

        if pres_ext == ".csv":
            gpkg_path = os.path.join(dst_dir, f"{pres_name}.gpkg")
            if not os.path.isfile(gpkg_path):
                try:
                    self._csv_to_gpkg(
                        csv_path=pres_path,
                        gpkg_path=gpkg_path,
                        layer_name=pres_name,
                        x_field=spec.get("presence_x_field") or "lon",
                        y_field=spec.get("presence_y_field") or "lat",
                        crs=spec.get("presence_crs") or "EPSG:4326",
                    )
                except Exception:
                    # If conversion fails for any reason, bail out
                    # of the auto-load step rather than fall back to
                    # the legacy CSV path — keeping a single code
                    # path is the whole point of the change.
                    return
            pres_path = pres_name + ".gpkg"
            pres_path = os.path.join(dst_dir, pres_path)
        elif pres_ext not in (".gpkg", ".geojson", ".shp"):
            return

        lyr = QgsVectorLayer(pres_path, pres_name, "ogr")
        if lyr.isValid():
            # Apply fixed presence-marker styling so example datasets
            # render the same way every time, regardless of QGIS's
            # random default colour. The deep blue (#2C5F8D) reads
            # cleanly on basemaps and contrasts with the red
            # priority-site marker (#cb181d) — the two can be
            # displayed together without confusion.
            try:
                sym = QgsSymbol.defaultSymbol(lyr.geometryType())
                if sym is not None:
                    sym.setColor(QColor("#2C5F8D"))
                    try:
                        sym.symbolLayer(0).setSize(2.5)
                    except Exception:
                        pass
                    lyr.renderer().setSymbol(sym.clone())
            except Exception:
                pass
            QgsProject.instance().addMapLayer(lyr)

    # ------------------------------------------------------------------
    def _csv_to_gpkg(self, csv_path: str, gpkg_path: str,
                     layer_name: str, x_field: str, y_field: str,
                     crs: str):
        """Convert a points CSV to a single-layer GeoPackage.

        Reads the CSV with the stdlib `csv` module (so we don't depend
        on pandas at example-loading time), constructs a Point layer
        in the supplied CRS, and writes it via QgsVectorFileWriter.
        Non-numeric x/y rows are skipped. All non-coord columns are
        kept as String fields so the original `species` / metadata
        columns survive into the GeoPackage.
        """
        import csv as _csv
        from qgis.core import (
            QgsCoordinateReferenceSystem, QgsField, QgsFields,
            QgsFeature, QgsGeometry, QgsPointXY,
            QgsProject as _QP, QgsVectorFileWriter, QgsWkbTypes,
        )
        from qgis.PyQt.QtCore import QVariant

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = _csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        fields = QgsFields()
        for name in fieldnames:
            fields.append(QgsField(name, QVariant.String))

        crs_obj = QgsCoordinateReferenceSystem(crs)
        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = "GPKG"
        opts.layerName = layer_name
        ctx = _QP.instance().transformContext()
        writer = QgsVectorFileWriter.create(
            gpkg_path, fields, QgsWkbTypes.Point, crs_obj, ctx, opts
        )
        if writer is None:
            raise RuntimeError(f"Could not create {gpkg_path}")
        if writer.hasError() != QgsVectorFileWriter.NoError:
            err = writer.errorMessage()
            del writer
            raise RuntimeError(f"GPKG writer error: {err}")

        try:
            for row in rows:
                try:
                    x = float(row.get(x_field, ""))
                    y = float(row.get(y_field, ""))
                except (TypeError, ValueError):
                    continue
                feat = QgsFeature(fields)
                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                feat.setAttributes([row.get(name, "") for name in fieldnames])
                writer.addFeature(feat)
        finally:
            del writer
