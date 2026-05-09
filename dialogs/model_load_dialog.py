"""Modal dialog: map QMaxent model variables to QGIS raster layers.

When the user loads a previously saved .pkl model, the model expects
its environmental rasters in a specific order. elapid's
apply_model_to_rasters uses raster *position*, not name, so a wrong
order produces silently incorrect predictions. This dialog forces an
explicit name-based mapping so the user can never project a model
against the wrong rasters.

Auto-matching is case-insensitive on the QGIS layer name. Manual
selection via per-row drop-downs covers cases where names differ or
where the user wants to deliberately substitute a covariate.
"""

from qgis.core import QgsProject, QgsRasterLayer
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QGridLayout, QLabel,
    QVBoxLayout,
)

from ..i18n import tr


class ModelLoadDialog(QDialog):
    """Match model variables to QGIS raster layers by name, then by user choice.

    Usage:
        dlg = ModelLoadDialog(parent, feature_names=["temp", "precip"])
        if dlg.exec_() == QDialog.Accepted:
            ordered_layers = dlg.selected_layers()   # list[QgsRasterLayer]

    The returned list has length len(feature_names) and is in the SAME order
    as the model's training-time covariates. The caller is responsible for
    setting the dock's raster list from this ordering.
    """

    def __init__(self, parent, feature_names: list, categorical_indices=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Map model variables to rasters"))
        self.setModal(True)

        self._feature_names = list(feature_names)
        # 0-based indices of variables that were categorical at training
        # time. Shown as a small "[cat]" tag next to the variable name so
        # the user knows which inputs need to be matched to a categorical
        # raster (e.g. land cover, biome, soil type).
        self._categorical_indices = set(categorical_indices or [])
        self._combos: list[QComboBox] = []  # one per feature, in order

        self._project_layers = self._collect_raster_layers()
        self._build_ui()
        self._auto_match()
        self._update_ok_button()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        v = QVBoxLayout(self)

        intro = QLabel(tr(
            "This model was trained on {n} variables in a specific order. "
            "Map each model variable to a QGIS raster layer below. "
            "Order matters: predictions are computed by raster position, "
            "not by name."
        ).format(n=len(self._feature_names)))
        intro.setWordWrap(True)
        v.addWidget(intro)

        # Pickle-format security advisory. .pkl files are Python pickle
        # objects, which can execute arbitrary code on load. We surface
        # this risk so the user opens models from trusted sources only —
        # ideally only models the user (or a trusted collaborator)
        # produced with this plugin. A short note here is the standard
        # academic disclosure for any tool that distributes pickled
        # models (sklearn, joblib, elapid all carry similar warnings in
        # their own documentation).
        sec_note = QLabel(tr(
            "⚠ Note: .pkl is a Python pickle file. Only load models from "
            "sources you trust (typically models you produced with this "
            "plugin); a malicious .pkl can execute arbitrary code on load."
        ))
        sec_note.setWordWrap(True)
        sec_note.setStyleSheet("color:#7A6A2A; font-size: 11px;")
        v.addWidget(sec_note)

        if not self._project_layers:
            warn = QLabel(tr(
                "⚠ No raster layers in the QGIS project. Add the required "
                "rasters to the project first, then load the model again."
            ))
            warn.setWordWrap(True)
            warn.setStyleSheet("color:#B23A2A;")
            v.addWidget(warn)

        # Header row + one row per model variable
        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 2)
        grid.addWidget(self._header_label(tr("Model variable:")), 0, 0)
        grid.addWidget(self._header_label(tr("QGIS raster layer:")), 0, 1)
        grid.addWidget(self._header_label(""), 0, 2)

        for row, name in enumerate(self._feature_names, start=1):
            display = name
            if (row - 1) in self._categorical_indices:
                # row is 1-indexed in the grid; subtract 1 for 0-based
                # categorical index lookup. Translate the [categorical]
                # tag so Korean users see the same text they see in the
                # ① Data tab toggle.
                display = f"{name}  {tr('[categorical]')}"
            name_lbl = QLabel(display)
            name_lbl.setStyleSheet("font-family: monospace;")
            grid.addWidget(name_lbl, row, 0)

            combo = QComboBox()
            combo.addItem(tr("— Select layer —"), userData=None)
            for lyr in self._project_layers:
                combo.addItem(lyr.name(), userData=lyr)
            combo.currentIndexChanged.connect(self._update_ok_button)
            grid.addWidget(combo, row, 1)
            self._combos.append(combo)

            # Per-row status indicator (e.g. ✓/⚠ glyph populated by
            # _update_ok_button when matched/unmatched).
            slot = QLabel("")
            slot.setObjectName(f"status_{row}")
            grid.addWidget(slot, row, 2)

        v.addLayout(grid)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        v.addWidget(self._summary)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.button(QDialogButtonBox.Ok).setText(tr("Load model"))
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        v.addWidget(self._buttons)

        self.resize(600, 80 + 36 * len(self._feature_names))

    def _header_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight:bold;")
        return lbl

    # --------------------------------------------------- raster collection
    def _collect_raster_layers(self) -> list:
        """Return all single-band-capable raster layers in the project."""
        layers = []
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsRasterLayer) and lyr.isValid():
                layers.append(lyr)
        # Sort by display name for predictable dropdown order.
        layers.sort(key=lambda l: l.name().lower())
        return layers

    # --------------------------------------------------- auto-matching
    def _auto_match(self):
        """Pre-select layers by case-insensitive name match."""
        by_name = {lyr.name().lower(): lyr for lyr in self._project_layers}
        for combo, fname in zip(self._combos, self._feature_names):
            match = by_name.get(fname.lower())
            if match is None:
                # Try also matching against layer name without extension
                # (some workflows name layers like "temp.tif")
                for key, lyr in by_name.items():
                    base = key.rsplit(".", 1)[0]
                    if base == fname.lower():
                        match = lyr
                        break
            if match is not None:
                idx = combo.findData(match)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    # --------------------------------------------------- OK gating
    def _update_ok_button(self):
        layers = [c.currentData() for c in self._combos]
        all_set = all(l is not None for l in layers)
        # Also check no duplicate selections
        non_none = [l for l in layers if l is not None]
        no_dupes = len(non_none) == len(set(id(l) for l in non_none))

        ok_btn = self._buttons.button(QDialogButtonBox.Ok)
        ok_btn.setEnabled(all_set and no_dupes)

        # Update the summary label
        n_total   = len(self._combos)
        n_matched = sum(1 for l in layers if l is not None)
        if not no_dupes:
            self._summary.setText(tr(
                "⚠ The same raster is assigned to two or more variables. "
                "Each model variable needs its own raster."
            ))
            self._summary.setStyleSheet("color:#B23A2A;")
        elif all_set:
            self._summary.setText(tr(
                "✓ All {n} variables matched."
            ).format(n=n_total))
            self._summary.setStyleSheet("color:#0F6E56;")
        else:
            self._summary.setText(tr(
                "Auto-matched {n}/{total} by name. Pick layers for the rest."
            ).format(n=n_matched, total=n_total))
            self._summary.setStyleSheet("color:#444;")

    # --------------------------------------------------- result
    def selected_layers(self) -> list:
        """Return the user's chosen layers, in model-variable order."""
        return [c.currentData() for c in self._combos]
