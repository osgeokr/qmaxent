"""QMaxent Main Dock — single panel with 4 tabs.

Fixes applied:
  ① Projection progress: deterministic % bar via tqdm patching in ProjectionWorker
  ② Matplotlib English-only: avoids Korean font encoding issues cross-platform
  ③ ROC curve: shown in Variable Importance tab
  ④ PNG export: response curves + ROC + jackknife saved alongside output raster
  ⑤ Parameters tab: Transform separated from Regularization; grid_size added
"""

import os

import numpy as np
from qgis.core import (
    Qgis,
    QgsMapLayerProxyModel,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
)
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QDoubleValidator, QFont, QIntValidator
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tooltip, tr

# ---------------------------------------------------------------------------
# Default output directory
# ---------------------------------------------------------------------------
# Every Output Files field across the dock (model.pkl, results.xlsx,
# prediction.tif, priority_sites.gpkg) shows an absolute path rooted at
# the same folder so the user always knows where artefacts will land.
# The previous version put bare filenames like "prediction.tif" in the
# fields; those resolved against QGIS's current working directory,
# which differs by OS / launcher / shell and is impossible for an end
# user to predict.
#
# Resolution order:
#   1. QMAXENT_OUTPUT_DIR environment variable, if set
#   2. ~/qmaxent_output (auto-created on first use)
#
# The folder is created on first call, so the directory always exists
# by the time any file dialog opens or any worker writes there.


def _default_output_dir() -> str:
    """Return the folder where QMaxent writes its outputs.

    Override with ``QMAXENT_OUTPUT_DIR`` if the user wants outputs in
    a project-specific location (e.g. one folder per study). The
    environment variable is read fresh on every call so a user who
    sets it via QGIS's environment-variable settings sees the change
    without needing to reload the plugin.
    """
    d = os.environ.get("QMAXENT_OUTPUT_DIR") or os.path.join(
        os.path.expanduser("~"), "qmaxent_output"
    )
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        # makedirs failure is rare (only when ~ itself isn't writable).
        # Fall through and let the eventual save attempt surface the
        # real error to the user — silently swallowing it here would
        # leave the path in the LineEdit looking valid when it isn't.
        pass
    return d


class QMaxentMainDock(QDockWidget):
    def __init__(self, iface, parent=None):
        super().__init__(tr("QMaxent — Analysis"), parent)
        self.setObjectName("QMaxentMainDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        # Minimum size: width holds the widest control comfortably (the
        # CV/threshold rows). Height is small enough to fit on 13″ MBP /
        # 1366-line laptop screens — the per-tab QScrollArea handles
        # long content. Without a height floor here, Qt's default sizeHint
        # comes from the *content* and forces the dock taller than the
        # screen on the first show, which is the very glitch we're
        # fixing (user had to title-bar-double-click to re-dock).
        self.setMinimumWidth(380)
        self.setMinimumHeight(500)

        self._iface = iface
        self._worker = None
        self._proj_worker = None
        self._model = None
        self._meta = None
        self._results = None

        self._build_ui()

    def sizeHint(self):
        """Suggest a starting size that fits common laptop screens.

        Without overriding sizeHint, Qt uses the layout's preferred
        size, which for QMaxent ends up being ~1100px tall (long
        Data tab + harmonize panel + buttons). On smaller screens
        QGIS clips the bottom — the user then has to grab the dock
        edge or title-bar-double-click to re-fit it. Returning a
        screen-aware default avoids that on first show.
        """
        from qgis.PyQt.QtCore import QSize

        try:
            screen = self._iface.mainWindow().screen()
            avail = screen.availableGeometry()
            # Use 70% of screen height as a reasonable default; never
            # below the minimum, never taller than the screen.
            h = max(500, min(int(avail.height() * 0.70), avail.height() - 100))
            return QSize(420, h)
        except Exception:
            return QSize(420, 720)

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self):
        root = QWidget()
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        self.tabs = QTabWidget()

        # Wrap each tab in a QScrollArea so the dock fits any screen
        # size: long tab content (e.g. Data tab with raster list +
        # categorical controls + harmonize panel) used to push the
        # dock past the bottom of the screen, hiding the "Run Maxent"
        # button and forcing the user to title-bar-double-click the
        # dock to dock it back into the QGIS shell. With per-tab
        # scrolling the dock's height stays bounded by the screen and
        # users scroll within the tab when content is long.
        def _scrolled(widget):
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QScrollArea.NoFrame)  # no extra border
            sa.setWidget(widget)
            return sa

        self.tabs.addTab(_scrolled(self._build_data_tab()), tr("① Data"))
        # _build_params_tab already returns a QScrollArea internally,
        # so wrapping it again would add a second nested scrollbar.
        # Pass it through unwrapped.
        self.tabs.addTab(self._build_params_tab(), tr("② Parameters"))
        self.tabs.addTab(_scrolled(self._build_train_tab()), tr("③ Training"))
        self.tabs.addTab(_scrolled(self._build_results_tab()), tr("④ Results"))
        self.tabs.addTab(_scrolled(self._build_priority_tab()), tr("⑤ Priority Sites for Survey"))
        vbox.addWidget(self.tabs, stretch=1)

        bot = QHBoxLayout()
        self._status_lbl = QLabel(tr("Model not yet trained."))
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bot.addWidget(self._status_lbl, stretch=1)

        self._run_btn = QPushButton(tr("▶  Run Maxent"))
        bold = QFont()
        bold.setBold(True)
        self._run_btn.setFont(bold)
        self._run_btn.setMinimumWidth(130)
        self._run_btn.setMinimumHeight(36)
        self._run_btn.clicked.connect(self._run_maxent)
        bot.addWidget(self._run_btn)
        vbox.addLayout(bot)

        self.setWidget(root)

    # ─── Tab 1: Data ─────────────────────────────────────────────────────────

    def _build_data_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)

        # Reuse trained model — caption + button + divider.
        # Option C layout: a question-style caption explains who this row is
        # for, the button sits at the right, and a thin divider separates
        # the "reuse" path from the "set up new training" path that
        # follows. The button opens a variable-mapping dialog before
        # adopting the model, protecting against the silent-failure mode
        # where wrong raster order yields nonsense projections.
        load_row = QHBoxLayout()
        load_caption = QLabel(tr("Already trained a model?"))
        load_caption.setStyleSheet("color: gray;")
        load_row.addWidget(load_caption)
        load_row.addStretch()
        load_btn = QPushButton(tr("Load existing model (.pkl)..."))
        load_btn.setToolTip(
            tooltip(
                tr(
                    "Load a previously saved QMaxent model and project it to "
                    "rasters in the current QGIS project. You will be asked to "
                    "match the model's variables to your raster layers."
                )
            )
        )
        load_btn.clicked.connect(self._on_load_model_clicked)
        load_row.addWidget(load_btn)
        v.addLayout(load_row)

        # Thin horizontal divider separating the reuse row from the new
        # training inputs below.
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        v.addWidget(divider)

        pres_grp = QGroupBox(tr("Presence Points Layer"))
        pg = QVBoxLayout(pres_grp)
        try:
            self._pres_combo = QgsMapLayerComboBox()
            self._pres_combo.setFilters(QgsMapLayerProxyModel.PointLayer)
            pg.addWidget(self._pres_combo)
            self._pres_combo.layerChanged.connect(self._on_presence_changed)
        except Exception:
            self._pres_combo = QComboBox()
            pg.addWidget(self._pres_combo)
        self._pres_info = QLabel(tr("Select a layer to see point count."))
        self._pres_info.setStyleSheet("color: gray; font-size: 11px;")
        pg.addWidget(self._pres_info)
        v.addWidget(pres_grp)

        rast_grp = QGroupBox(tr("Environmental Rasters"))
        rg = QVBoxLayout(rast_grp)
        self._raster_list = QListWidget()
        self._raster_list.setAlternatingRowColors(True)
        self._raster_list.setSelectionMode(QListWidget.ExtendedSelection)
        self._raster_list.setMinimumHeight(120)
        rg.addWidget(self._raster_list)
        btn_row = QHBoxLayout()
        add_btn = QPushButton(tr("Add from project"))
        add_btn.clicked.connect(self._add_raster_from_project)
        btn_row.addWidget(add_btn)
        rm_btn = QPushButton(tr("Remove selected"))
        rm_btn.clicked.connect(self._remove_selected_rasters)
        btn_row.addWidget(rm_btn)
        up_btn = QPushButton("▲")
        up_btn.setMaximumWidth(30)
        up_btn.clicked.connect(lambda: self._move_raster(-1))
        btn_row.addWidget(up_btn)
        dn_btn = QPushButton("▼")
        dn_btn.setMaximumWidth(30)
        dn_btn.clicked.connect(lambda: self._move_raster(1))
        btn_row.addWidget(dn_btn)
        rg.addLayout(btn_row)

        # ── Explicit raster-grid consistency check ────────────────────────
        # Following biomod2's BIOMOD_FormatingData and SDMSelect's
        # Prepare_r_multi, raster harmonization is an explicit pre-
        # training step rather than an opaque automatic one. The user
        # presses "Check Raster Consistency" to inspect the inputs;
        # when grids disagree, a "Harmonize to Folder..." button
        # appears for the user to write aligned copies to a folder of
        # their choosing. This trades a couple of extra clicks for
        # transparency, reproducibility, and inspectable output.
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        rg.addWidget(sep)

        check_row = QHBoxLayout()
        self._check_consistency_btn = QPushButton(tr("Check Raster Consistency"))
        self._check_consistency_btn.setToolTip(
            tooltip(
                tr(
                    "Inspect every raster's CRS, extent, and resolution. "
                    "If any of them disagree, a 'Harmonize to Folder…' button "
                    "will appear so you can write aligned copies to a folder "
                    "of your choosing (Hijmans 2024; SDMSelect Prepare_r_multi)."
                )
            )
        )
        self._check_consistency_btn.clicked.connect(self._on_check_consistency)
        check_row.addWidget(self._check_consistency_btn)
        self._harmonize_btn = QPushButton(tr("Harmonize to Folder..."))
        self._harmonize_btn.setVisible(False)  # hidden until mismatch found
        self._harmonize_btn.clicked.connect(self._on_harmonize_clicked)
        check_row.addWidget(self._harmonize_btn)
        check_row.addStretch()
        rg.addLayout(check_row)

        self._consistency_lbl = QLabel(tr("Status: not checked yet."))
        self._consistency_lbl.setWordWrap(True)
        self._consistency_lbl.setStyleSheet("color: gray;")
        rg.addWidget(self._consistency_lbl)

        v.addWidget(rast_grp)

        bg_grp = QGroupBox(tr("Background Points"))
        bgg = QHBoxLayout(bg_grp)
        bgg.addWidget(QLabel(tr("Sample count:")))
        self._bg_spin = QSpinBox()
        self._bg_spin.setRange(100, 100000)
        self._bg_spin.setValue(10000)
        self._bg_spin.setSingleStep(1000)
        bgg.addWidget(self._bg_spin)
        bgg.addStretch()
        v.addWidget(bg_grp)

        # ── Export for external Maxent (v0.1.3, dual-format in v0.1.4) ───
        # Exports the current presence + raster selection in one of two
        # formats consumable by the standalone maxent.jar implementation
        # (Phillips et al. 2017). Two formats are offered:
        #
        #   "SWD" — Samples-With-Data CSV pair. Best for byte-level
        #     cross-checking the elapid backend against maxent.jar on
        #     exactly the same point sample. Does NOT yield a
        #     projection raster on its own.
        #
        #   "Samples + Raster" — A short samples CSV plus one ESRI
        #     ASCII Grid (.asc) per environmental raster. This is the
        #     natural maxent.jar input format: model fit AND a
        #     projection raster are produced in a single command-line
        #     invocation. Files are larger than SWD because the
        #     environment is stored as full surfaces, not just the
        #     extracted point values.
        #
        # Both formats write a README.txt with a generic maxent.jar
        # default-settings command line, so any QGIS user can take the
        # output and run maxent.jar without further configuration.
        swd_grp = QGroupBox(tr("Export for external Maxent"))
        swdv = QVBoxLayout(swd_grp)

        swd_info = QLabel(
            tr(
                "Export the current presence and environmental raster "
                "selection in a format consumable by the standalone "
                "maxent.jar implementation. Useful for running the "
                "original Java Maxent on the same data, or for sharing "
                "a ready-to-use Maxent input directory."
            )
        )
        swd_info.setWordWrap(True)
        swd_info.setStyleSheet("color: #555; font-size: 9pt;")
        swdv.addWidget(swd_info)

        # Format radio selector. We use plain QRadioButton instead of
        # a QComboBox so the trade-off between the two formats is
        # visible side-by-side without having to open a dropdown. The
        # default option (Samples + Raster) is listed first to match
        # the platform convention that the default sits at the top of
        # a radio group (Microsoft Fluent Design, Apple HIG, and QGIS
        # itself follow this).
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel(tr("Format:")))
        self._swd_fmt_raster = QRadioButton(tr("Samples + Raster (samples CSV + .asc layers)"))
        self._swd_fmt_raster.setToolTip(
            tooltip(
                tr(
                    "A short samples CSV (Species, Longitude, Latitude only) "
                    "plus one ESRI ASCII Grid (.asc) per environmental raster "
                    "in a layers/ folder. maxent.jar fits the model AND "
                    "produces a projection raster in a single run. Larger "
                    "files but ready to use without additional arguments."
                )
            )
        )
        self._swd_fmt_swd = QRadioButton(tr("SWD (CSV pair, extracted values)"))
        self._swd_fmt_swd.setToolTip(
            tooltip(
                tr(
                    "Two CSVs (presence.csv + background.csv) with environmental "
                    "values pre-extracted at each point. maxent.jar will use "
                    "these exact points and values — no further sampling. "
                    "Smaller files, but produces no projection raster unless "
                    "you also pass projectionlayers= to maxent.jar."
                )
            )
        )
        # Default to the format that matches typical maxent.jar usage —
        # samples + raster — so a first-time user gets a usable
        # projection raster out of the box.
        self._swd_fmt_raster.setChecked(True)
        fmt_row.addWidget(self._swd_fmt_raster)
        fmt_row.addWidget(self._swd_fmt_swd)
        fmt_row.addStretch()
        swdv.addLayout(fmt_row)

        swd_row = QHBoxLayout()
        swd_row.addWidget(QLabel(tr("Output folder:")))
        self._swd_dir_edit = QLineEdit()
        self._swd_dir_edit.setText(os.path.join(_default_output_dir(), "maxent_export"))
        self._swd_dir_edit.setToolTip(self._swd_dir_edit.text())
        self._swd_dir_edit.textChanged.connect(self._swd_dir_edit.setToolTip)
        swd_row.addWidget(self._swd_dir_edit, stretch=1)
        swd_browse = QPushButton("...")
        swd_browse.setMaximumWidth(30)
        swd_browse.clicked.connect(self._browse_swd_dir)
        swd_row.addWidget(swd_browse)
        swdv.addLayout(swd_row)

        self._swd_export_btn = QPushButton(tr("Export"))
        self._swd_export_btn.clicked.connect(self._on_swd_export_clicked)
        swdv.addWidget(self._swd_export_btn)

        v.addWidget(swd_grp)

        v.addStretch()
        return w

    # ── Raster list helpers ─────────────────────────────────────────────────
    #
    # Each item in self._raster_list represents one environmental raster.
    # We track two pieces of metadata per item:
    #   Qt.UserRole       — QGIS layer id (string)
    #   Qt.UserRole + 1   — bool: whether this raster is categorical
    #
    # The categorical flag drives both the visible "[continuous]/[categorical]"
    # toggle on the right of each row AND elapid's MaxentModel.fit(...,
    # categorical=[indices, ...]) call. Following the dismo, ENMeval, SDMtune
    # and maxent.jar convention, the user — not the plugin — declares
    # categorical variables. The plugin only auto-checks when the GeoTIFF
    # carries an explicit "categorical" or "PIXELTYPE=SIGNEDBYTE" metadata
    # hint (see _detect_categorical_hint); raw data types and value counts
    # are intentionally NOT used as auto-detection signals because they
    # produce false positives on rasters like elevation that are stored as
    # integers but are continuous.

    _ROLE_LAYER_ID = Qt.UserRole
    _ROLE_IS_CATEG = Qt.UserRole + 1

    def _add_raster_item(
        self, layer_name: str, layer_id: str, is_categorical: bool, display_text: str = None
    ):
        """Append a row to the raster list with a categorical toggle button."""
        item = QListWidgetItem("")
        item.setData(self._ROLE_LAYER_ID, layer_id)
        item.setData(self._ROLE_IS_CATEG, bool(is_categorical))
        self._raster_list.addItem(item)

        # Custom row widget: name on the left, toggle on the right.
        # Transparent background so the QListWidgetItem's native
        # selection highlight (blue when the row is selected, white
        # when not) shows through the row widget. Without this the
        # row widget paints its own opaque background and selected
        # rows look unselected — a known Qt gotcha when combining
        # setItemWidget with ExtendedSelection.
        #
        # The stylesheet uses an objectName-scoped selector
        # (#qmaxentRasterRow) so the transparent rule applies ONLY
        # to this QWidget and does NOT cascade to its descendants.
        # An unscoped "background: transparent;" would propagate to
        # the categorical toggle QPushButton and strip its native
        # button styling — leaving the user with a row of black
        # rectangles on the Windows native theme.
        row_w = QWidget()
        row_w.setObjectName("qmaxentRasterRow")
        row_w.setAttribute(Qt.WA_TranslucentBackground)
        row_w.setAutoFillBackground(False)
        row_w.setStyleSheet("QWidget#qmaxentRasterRow { background: transparent; }")
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(6, 2, 6, 2)
        row_l.setSpacing(8)

        name_lbl = QLabel(display_text or layer_name)
        # The label needs its own transparent background for the
        # same reason — and it's safe to scope this one with
        # objectName too, since the QLabel has no descendants we
        # care about styling.
        name_lbl.setObjectName("qmaxentRasterRowLabel")
        name_lbl.setStyleSheet("QLabel#qmaxentRasterRowLabel { background: transparent; }")
        row_l.addWidget(name_lbl, stretch=1)

        toggle = QPushButton()
        toggle.setCheckable(True)
        toggle.setChecked(bool(is_categorical))
        toggle.setMinimumWidth(96)
        toggle.setMaximumWidth(120)
        toggle.setFocusPolicy(Qt.NoFocus)
        self._sync_categorical_toggle(toggle)
        toggle.toggled.connect(
            lambda checked, it=item, btn=toggle: self._on_categorical_toggled(it, btn, checked)
        )
        row_l.addWidget(toggle)

        # The QListWidget owns the row widget; size hints follow the widget.
        item.setSizeHint(row_w.sizeHint())
        self._raster_list.setItemWidget(item, row_w)
        return item

    def _sync_categorical_toggle(self, btn):
        """Update the toggle button's label and tooltip to match its state."""
        if btn.isChecked():
            btn.setText(tr("[categorical]"))
            btn.setToolTip(
                tooltip(
                    tr(
                        "Treated as categorical: one-hot encoded inside the model "
                        "(e.g. land cover, biome, soil type)."
                    )
                )
            )
        else:
            btn.setText(tr("[continuous]"))
            btn.setToolTip(
                tooltip(
                    tr(
                        "Treated as a continuous numeric variable "
                        "(e.g. temperature, precipitation, elevation)."
                    )
                )
            )

    def _on_categorical_toggled(self, item, btn, checked: bool):
        item.setData(self._ROLE_IS_CATEG, bool(checked))
        self._sync_categorical_toggle(btn)

    # ── Tab 2: Parameters ───────────────────────────────────────────────────

    def _build_params_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)

        # Feature Types
        ft_grp = QGroupBox(tr("Feature Types"))
        ftv = QVBoxLayout(ft_grp)
        self._ft_auto_radio = QRadioButton(tr("Auto (sample-size based, Phillips et al. 2017)"))
        self._ft_manual_radio = QRadioButton(tr("Manual selection"))
        self._ft_auto_radio.setChecked(True)
        ftv.addWidget(self._ft_auto_radio)
        ftv.addWidget(self._ft_manual_radio)
        ft_row = QHBoxLayout()
        self._ft_checks = {}
        for code, label in [
            ("linear", "Linear"),
            ("quadratic", "Quadratic"),
            ("hinge", "Hinge"),
            ("product", "Product"),
            ("threshold", "Threshold"),
        ]:
            cb = QCheckBox(label)
            # Default: all five checked (LQPHT) — consistent with the
            # maxnet auto rule for n >= 80 (Phillips et al. 2017).
            cb.setChecked(True)
            cb.setEnabled(False)
            self._ft_checks[code] = cb
            ft_row.addWidget(cb)
        ft_row.addStretch()
        ftv.addLayout(ft_row)
        self._ft_auto_radio.toggled.connect(
            lambda on: [cb.setEnabled(not on) for cb in self._ft_checks.values()]
        )
        v.addWidget(ft_grp)

        # Regularization — Beta only (Transform is NOT regularization)
        reg_grp = QGroupBox(tr("Regularization"))
        regv = QHBoxLayout(reg_grp)
        regv.addWidget(QLabel(tr("Regularization multiplier:")))
        self._beta_spin = QDoubleSpinBox()
        self._beta_spin.setRange(0.1, 10.0)
        self._beta_spin.setSingleStep(0.1)
        # Default 1.0 follows the maxent.jar / maxnet learning standard
        # (Phillips et al. 2017; ENMeval default), NOT elapid's own
        # MaxentModel default of 1.5. We deliberately track the published
        # Maxent learning convention so that QMaxent results are directly
        # comparable to maxent.jar and ENMeval out of the box.
        self._beta_spin.setValue(1.0)
        self._beta_spin.setToolTip(
            tooltip(
                tr(
                    "Regularization multiplier (Phillips et al. 2017). "
                    "Higher values produce smoother, more conservative response "
                    "curves; lower values fit training data more tightly. "
                    "Default 1.0 follows the maxent.jar / maxnet standard."
                )
            )
        )
        regv.addWidget(self._beta_spin)
        regv.addStretch()
        v.addWidget(reg_grp)

        # Advanced
        adv_grp = QGroupBox(tr("Advanced"))
        advv = QVBoxLayout(adv_grp)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel(tr("Hinge knots:")))
        self._hinge_spin = QSpinBox()
        self._hinge_spin.setRange(5, 200)
        self._hinge_spin.setValue(50)
        self._hinge_spin.setToolTip(
            tooltip(tr("Each variable contributes 2*nknots-2 hinge features (maxnet default: 50)."))
        )
        row1.addWidget(self._hinge_spin)
        row1.addSpacing(20)
        row1.addWidget(QLabel(tr("Threshold knots:")))
        self._thresh_spin = QSpinBox()
        self._thresh_spin.setRange(5, 200)
        self._thresh_spin.setValue(50)
        self._thresh_spin.setToolTip(
            tooltip(
                tr("Each variable contributes 2*nknots-2 threshold features (maxnet default: 50).")
            )
        )
        row1.addWidget(self._thresh_spin)
        row1.addStretch()
        advv.addLayout(row1)
        # addsamplestobackground — keep the user-facing label clean and put
        # the maxnet identifier in the tooltip (where developers expect it).
        self._addtobg_chk = QCheckBox(tr("Add presences to background"))
        self._addtobg_chk.setChecked(True)
        self._addtobg_chk.setToolTip(
            tooltip(
                tr(
                    "Add presences to the background sample so that the fitted "
                    "density is consistent over the full study area "
                    "(addsamplestobackground; Phillips et al. 2006, 2017)."
                )
            )
        )
        advv.addWidget(self._addtobg_chk)

        # distance_weights — opt-in spatial bias correction.
        #
        # Mainstream SDM tools (maxent.jar, ENMeval, biomod2, spThin,
        # GeoThinneR) all leave bias correction OFF by default and
        # require the user to opt in. The reasoning is that bias
        # correction is data-dependent: it is appropriate for
        # opportunistic data with known sampling bias (GBIF, citizen
        # science) but inappropriate for systematic surveys or for
        # genuinely clustered species. Following Phillips et al. (2009)
        # — whose recommendation is conditional ("when sampling bias is
        # present") rather than universal — we keep this checkbox
        # unchecked by default. When enabled, elapid's distance_weights
        # function down-weights spatially clustered presence points so
        # the fitted model reflects habitat suitability rather than
        # sampling effort.
        self._distweights_chk = QCheckBox(tr("Down-weight spatially clustered points"))
        self._distweights_chk.setChecked(False)
        self._distweights_chk.setToolTip(
            tooltip(
                tr(
                    "Reduce the influence of spatially clustered presences to "
                    "correct for sample selection bias (Phillips et al. 2009; "
                    "elapid distance_weights). Recommended when occurrence data "
                    "come from opportunistic sources (e.g. GBIF, citizen "
                    "science). Not recommended for systematic surveys or when "
                    "clustering reflects genuine habitat preference."
                )
            )
        )
        advv.addWidget(self._distweights_chk)
        v.addWidget(adv_grp)

        # Spatial evaluation (cross-validation strategies that account for
        # spatial autocorrelation; Roberts et al. 2017)
        cv_grp = QGroupBox(tr("Spatial evaluation"))
        cvv = QVBoxLayout(cv_grp)
        row_cv = QHBoxLayout()
        row_cv.addWidget(QLabel(tr("Method:")))
        self._cv_combo = QComboBox()
        self._cv_combo.addItems(
            [
                tr("None (no cross-validation; training AUC only)"),
                tr("Geographic K-Fold (Anderson 2023)"),
                tr("Random K-Fold (Phillips 2006)"),
                tr("Checkerboard (single spatial split; Muscarella 2014)"),
                tr("Buffered LOO (Pearson 2007; Ploton 2020)"),
            ]
        )
        self._cv_combo.setCurrentIndex(1)  # Geographic K-Fold (default)
        self._cv_combo.setToolTip(
            tr(
                "Geographic K-Fold (default) provides spatially-independent "
                "folds and is the recommended choice for unbiased "
                "generalization assessment when presence points show any "
                "spatial clustering (Roberts 2017).\n\n"
                "Random K-Fold is included for direct comparability with "
                "maxent.jar / ENMeval / dismo workflows; note it tends to "
                "inflate AUC estimates relative to spatial methods when "
                "presences are spatially autocorrelated."
            )
        )
        self._cv_combo.currentIndexChanged.connect(self._on_cv_method_changed)
        row_cv.addWidget(self._cv_combo, stretch=1)
        cvv.addLayout(row_cv)

        row_folds = QHBoxLayout()
        # Keep references to the labels so _on_cv_method_changed can grey
        # them out together with the spin boxes when the active CV method
        # does not use a given parameter.
        self._folds_lbl = QLabel(tr("Folds:"))
        row_folds.addWidget(self._folds_lbl)
        self._folds_spin = QSpinBox()
        self._folds_spin.setRange(2, 20)
        self._folds_spin.setValue(5)
        row_folds.addWidget(self._folds_spin)
        row_folds.addSpacing(16)

        # Grid size — for Checkerboard (was missing)
        self._grid_lbl = QLabel(tr("Grid size (m):"))
        row_folds.addWidget(self._grid_lbl)
        self._grid_spin = QDoubleSpinBox()
        # Range: 1m to 10,000km. 50000 (50 km) is a reasonable
        # default for continent-wide SDMs; users with finer-grained
        # study areas can drop it to a few kilometres.
        self._grid_spin.setRange(1.0, 1e7)
        self._grid_spin.setValue(50000.0)
        self._grid_spin.setSingleStep(10000.0)
        self._grid_spin.setDecimals(0)
        # v0.1.7: Grid size is always in metres regardless of the
        # presence-layer CRS. Internally the worker reprojects to
        # EPSG:6933 (area-preserving metric) before handing the
        # GeoDataFrame to elapid's checkerboard_split.
        self._grid_spin.setToolTip(
            tooltip(
                tr(
                    "Checkerboard cell size in metres. Choose a value "
                    "matching the environmental autocorrelation length of "
                    "the study area (Muscarella et al. 2014, ENMeval). "
                    "Internally reprojected to EPSG:6933 so this value is "
                    "metres regardless of the presence layer's CRS."
                )
            )
        )
        row_folds.addWidget(self._grid_spin)
        row_folds.addSpacing(16)

        self._buffer_lbl = QLabel(tr("Buffer (m):"))
        row_folds.addWidget(self._buffer_lbl)
        self._buffer_spin = QDoubleSpinBox()
        self._buffer_spin.setRange(0, 1e7)
        self._buffer_spin.setValue(50000)
        self._buffer_spin.setSingleStep(10000)
        self._buffer_spin.setDecimals(0)
        # v0.1.7: Buffer is always in metres regardless of the
        # presence-layer CRS — same reprojection treatment as
        # Grid size above.
        self._buffer_spin.setToolTip(
            tooltip(
                tr(
                    "Buffer distance in metres. Choose a value appropriate "
                    "to the species' dispersal range (Roberts et al. 2017; "
                    "Ploton et al. 2020). Internally reprojected to "
                    "EPSG:6933 so this value is metres regardless of the "
                    "presence layer's CRS."
                )
            )
        )
        row_folds.addWidget(self._buffer_spin)
        row_folds.addStretch()
        cvv.addLayout(row_folds)

        # Random seed — controls every stochastic operation in the
        # plugin (Random K-Fold partitions, hold-out splits used by
        # the jackknife when no spatial CV is configured, the priority
        # sites RNG used for shuffle-mode sampling, and elapid's
        # internal background draw).
        #
        # UI layout: a checkbox toggles deterministic mode (default
        # on); the spinbox carries the actual seed value, which is
        # written into the Overview sheet of results.xlsx so the
        # exact seed value used in a paper is recoverable. This
        # mirrors the maxent.jar "random seed" UI convention — both
        # a switch and an explicit value, since journal Methods
        # sections typically need to cite the seed value, not just
        # "fixed/not fixed".
        seed_row = QHBoxLayout()
        self._seed_check = QCheckBox(tr("Fix random seed:"))
        self._seed_check.setChecked(True)
        self._seed_check.setToolTip(
            tr(
                "On (default): use the seed value below for every "
                "stochastic operation (CV fold partitions, hold-out "
                "splits, priority-site shuffling, background draws). "
                "Same seed → identical results across runs.\n\n"
                "Off: a fresh random seed is drawn from the OS each "
                "run; results will vary slightly between runs. Useful "
                "when checking robustness to fold assignment without "
                "having to manually try multiple values."
            )
        )
        seed_row.addWidget(self._seed_check)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2_147_483_647)
        # 42 is the de-facto ML community convention for a "reasonable
        # arbitrary integer seed" (Adams, 1979). It carries no special
        # statistical properties — any fixed integer is equally valid
        # for reproducibility — but using a deliberately-chosen value
        # signals to reviewers that the seed is intentional rather than
        # an unreset default. The actual value is written into the
        # Overview sheet of results.xlsx for citation in the paper's
        # Methods section.
        self._seed_spin.setValue(42)
        self._seed_spin.setToolTip(
            tr(
                "The seed value. The Overview sheet of results.xlsx "
                "records this value (or 'random (not fixed)' when the "
                "checkbox is off) so the run is fully reproducible."
            )
        )
        seed_row.addWidget(self._seed_spin)
        seed_row.addStretch()
        cvv.addLayout(seed_row)

        # Wire the checkbox to grey out the spinbox when off.
        self._seed_check.toggled.connect(self._seed_spin.setEnabled)

        v.addWidget(cv_grp)
        # Apply the initial enable/disable state to match the default
        # method (Geographic K-Fold, index 1).
        self._on_cv_method_changed(self._cv_combo.currentIndex())

        # Jackknife is a variable-importance analysis, not a spatial CV
        # strategy — it sits in its own row so the UI structure mirrors
        # what the worker actually does.
        self._jackknife_chk = QCheckBox(tr("Jackknife variable importance"))
        self._jackknife_chk.setChecked(True)
        self._jackknife_chk.setToolTip(
            tooltip(
                tr(
                    "Train models with each variable alone and with each "
                    "variable removed; report train and held-out test AUC. "
                    "Robust to multicollinearity but slower (requires "
                    "retraining N×2 times)."
                )
            )
        )
        v.addWidget(self._jackknife_chk)

        # Permutation importance (added v0.1.3) — the same metric
        # reported by maxent.jar as "permutation importance" in its
        # HTML output. Computed by shuffling each variable's values
        # in the fitted model and measuring the AUC drop; reported
        # both as raw AUC drop and as a percentage of total
        # (matching maxent.jar's convention). Enables direct ranking
        # comparison with published Maxent SDM studies.
        pi_row = QHBoxLayout()
        self._permutation_chk = QCheckBox(tr("Permutation importance"))
        self._permutation_chk.setChecked(True)
        self._permutation_chk.setToolTip(
            tooltip(
                tr(
                    "Shuffle each variable's values in the fitted model and "
                    "measure the AUC drop (sklearn permutation_importance). "
                    "Same metric reported by maxent.jar; enables direct "
                    "ranking comparison with maxent.jar-based SDM studies. "
                    "Fast (no retraining) but can underestimate the "
                    "importance of correlated variables — interpret "
                    "alongside jackknife rather than alone."
                )
            )
        )
        pi_row.addWidget(self._permutation_chk)
        pi_row.addWidget(QLabel(tr("Repeats:")))
        self._permutation_repeats_spin = QSpinBox()
        self._permutation_repeats_spin.setRange(2, 100)
        self._permutation_repeats_spin.setValue(10)
        self._permutation_repeats_spin.setToolTip(
            tooltip(
                tr(
                    "Number of independent shuffles per variable. The "
                    "default of 10 matches sklearn and maxent.jar; higher "
                    "values reduce noise in the std column at linear cost."
                )
            )
        )
        pi_row.addWidget(self._permutation_repeats_spin)
        pi_row.addStretch()
        # Enable/disable the spin box alongside the checkbox so the UI
        # state never lies about what will actually be computed.
        self._permutation_chk.toggled.connect(self._permutation_repeats_spin.setEnabled)
        v.addLayout(pi_row)

        # Output Files
        out_grp = QGroupBox(tr("Output Files"))
        outg = QVBoxLayout(out_grp)

        def _path_row(label_text, filter_str, default_filename=None):
            hl = QHBoxLayout()
            hl.addWidget(QLabel(label_text))
            le = QLineEdit()
            le.setPlaceholderText(tr("Enter path or click [...]"))
            # Pre-fill with an absolute path under the default output
            # folder so the user can read the on-screen field and
            # know exactly where the file will land — no more guessing
            # against QGIS's current working directory. When the
            # caller passes only a basename we prepend the default
            # output dir; absolute paths from the caller are honoured
            # verbatim.
            if default_filename:
                if not os.path.isabs(default_filename):
                    default_filename = os.path.join(_default_output_dir(), default_filename)
                le.setText(default_filename)
            # Show the full path on hover so a narrow dock that
            # visually truncates the line edit doesn't hide where the
            # file is going. Auto-sync the tooltip when the user types.
            le.setToolTip(le.text())
            le.textChanged.connect(le.setToolTip)
            hl.addWidget(le, stretch=1)
            btn = QPushButton("...")
            btn.setMaximumWidth(30)
            btn.clicked.connect(lambda: self._browse_save(le, filter_str))
            hl.addWidget(btn)
            return hl, le

        row_m, self._model_path = _path_row(
            tr("Model (.pkl):"),
            tr("Pickle files (*.pkl)"),
            default_filename="model.pkl",
        )
        outg.addLayout(row_m)
        row_c, self._xlsx_path = _path_row(
            tr("Results XLSX:"),
            tr("Excel files (*.xlsx)"),
            default_filename="results.xlsx",
        )
        outg.addLayout(row_c)

        # v0.1.7: the explicit "Save training log alongside prediction
        # raster" checkbox is removed. training_log.txt is now written
        # automatically next to model.pkl at the end of ▶ Run Maxent —
        # the natural moment a researcher would want the log — rather
        # than gating it on the optional ▶ Run Spatial Projection step.
        # The log captures the full Run Maxent narrative (categorical
        # mask decisions, fold AUCs, jackknife outliers, dependency
        # versions) which the results.xlsx Overview sheet only
        # summarises, and is small enough (a few KB) that opt-in
        # friction would be net-negative.
        v.addWidget(out_grp)
        v.addStretch()
        scroll.setWidget(inner)
        return scroll

    def _on_cv_method_changed(self, idx: int):
        """Enable only the parameters that the active CV method actually uses.

        Index map (matches `_cv_combo` order):
            0 — None / random hold-out → no parameters
            1 — Geographic K-Fold      → folds
            2 — Random K-Fold          → folds
            3 — Checkerboard           → grid size
            4 — Buffered LOO           → buffer distance

        Greying out the irrelevant rows prevents the silent confusion
        where a user changes, say, Buffer while running K-Fold and
        wonders why the result didn't change.
        """
        # Both Geographic K-Fold and Random K-Fold consume the "folds"
        # parameter; the other rows stay greyed out for them.
        uses_folds = idx == 1 or idx == 2
        uses_grid = idx == 3
        uses_buffer = idx == 4

        for w in (self._folds_lbl, self._folds_spin):
            w.setEnabled(uses_folds)
        for w in (self._grid_lbl, self._grid_spin):
            w.setEnabled(uses_grid)
        for w in (self._buffer_lbl, self._buffer_spin):
            w.setEnabled(uses_buffer)

    # ─── Tab 3: Training ─────────────────────────────────────────────────────

    def _build_train_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)
        self._train_progress = QProgressBar()
        self._train_progress.setRange(0, 100)
        self._train_progress.setValue(0)
        self._train_progress.setTextVisible(True)
        v.addWidget(self._train_progress)
        self._train_status = QLabel(tr("Waiting..."))
        self._train_status.setStyleSheet("font-weight: bold;")
        v.addWidget(self._train_status)
        self._train_log = QPlainTextEdit()
        self._train_log.setReadOnly(True)
        self._train_log.setMaximumBlockCount(500)
        self._train_log.setFont(QFont("Courier New", 9))
        v.addWidget(self._train_log, stretch=1)
        # Three-button row: clear / copy / save-as. "Copy log" and
        # "Save log as…" give the user a manual escape hatch separate
        # from the auto-save (which lands next to model.pkl at the end
        # of ▶ Run Maxent in v0.1.7), useful when (a) the run is still
        # in progress, (b) the user wants a quick snapshot at a
        # specific moment, or (c) the log needs to go into a slack /
        # email / GitHub issue. The buttons grab whatever is currently
        # in the QPlainTextEdit — same single source of truth as the
        # auto-save path.
        log_btn_row = QHBoxLayout()
        clr_btn = QPushButton(tr("Clear log"))
        clr_btn.clicked.connect(self._train_log.clear)
        copy_btn = QPushButton(tr("Copy log"))
        copy_btn.setToolTip(tr("Copy the entire training log to the clipboard."))
        copy_btn.clicked.connect(self._copy_train_log_to_clipboard)
        save_btn = QPushButton(tr("Save log as..."))
        save_btn.setToolTip(
            tr(
                "Save the entire training log to a text file at a "
                "location of your choice. This is independent of the "
                "auto-save next to model.pkl."
            )
        )
        save_btn.clicked.connect(self._save_train_log_as)
        log_btn_row.addWidget(clr_btn)
        log_btn_row.addWidget(copy_btn)
        log_btn_row.addWidget(save_btn)
        log_btn_row.addStretch()
        v.addLayout(log_btn_row)
        return w

    # ─── Tab 4: Results ──────────────────────────────────────────────────────

    def _build_results_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)
        self._result_tabs = QTabWidget()
        self._result_tabs.addTab(self._build_response_tab(), tr("Response Curves"))
        self._result_tabs.addTab(self._build_importance_tab(), tr("Jackknife Importance"))
        self._result_tabs.addTab(self._build_permutation_tab(), tr("Permutation Importance"))
        self._result_tabs.addTab(self._build_project_tab(), tr("Spatial Projection"))
        v.addWidget(self._result_tabs)
        return w

    def _build_response_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        # v0.1.7: one-paragraph explanation matching the pattern used
        # on the Jackknife and Permutation tabs. Tells the user what
        # a response curve actually shows so the Y-axis transform
        # selector and the per-variable dropdown make sense without
        # prior Maxent familiarity. The "holding other variables at
        # training-set means" caveat is spelled out because it's the
        # single most common misreading — users intuitively want to
        # read the curve as an absolute probability map.
        info = QLabel(
            tr(
                "Marginal effect: how the predicted suitability changes "
                "across the range of one variable while the other "
                "variables are held at their training-set means. The "
                "Y-axis reads as relative habitat suitability (cloglog: "
                "0 to 1, saturates near 1 in the most suitable region). "
                "The shaded green band marks the variable's training-data "
                "range — curves outside it are extrapolation and should "
                "be interpreted with caution; the dotted line marks the "
                "training-data mean. Useful for inspecting whether the "
                "model has learned ecologically plausible shapes (e.g. "
                "unimodal temperature optima) and for spotting variables "
                "whose curves are nearly flat — candidates for removal "
                "in a parsimonious model."
            )
        )
        info.setWordWrap(True)
        info.setStyleSheet("QLabel { padding: 6px; font-size: 9pt; }")
        v.addWidget(info)

        bar = QHBoxLayout()
        bar.addWidget(QLabel(tr("Variable:")))
        self._response_var_combo = QComboBox()
        self._response_var_combo.currentIndexChanged.connect(self._plot_response)
        bar.addWidget(self._response_var_combo, stretch=1)
        v.addLayout(bar)
        self._response_canvas_widget = QWidget()
        self._response_canvas_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(self._response_canvas_widget, stretch=1)
        return w

    def _build_importance_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        # v0.1.7: explanation strip matching the other two
        # variable-analysis tabs. Distinguishes Jackknife (retrain
        # with each variable alone / removed) from Permutation
        # (shuffle in the fitted model) so users know why both are
        # offered and which one to cite for which question.
        info = QLabel(
            tr(
                "Retraining test: trains the model with each variable "
                "alone (per-variable AUC) and with each variable removed "
                "(per-variable drop in AUC), then averages across CV "
                "folds. Robust to correlated variables — unlike "
                "permutation importance — because each retrained model "
                "sees a different variable set. Slower (requires N×2 "
                "retrains) but the standard Maxent variable-importance "
                "diagnostic since Phillips et al. (2006)."
            )
        )
        info.setWordWrap(True)
        info.setStyleSheet("QLabel { padding: 6px; font-size: 9pt; }")
        v.addWidget(info)

        self._importance_canvas_widget = QWidget()
        self._importance_canvas_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(self._importance_canvas_widget, stretch=1)
        return w

    def _build_permutation_tab(self) -> QWidget:
        """Permutation importance Results sub-tab (added v0.1.3).

        Shows a horizontal bar chart of normalized permutation
        importance (percent of total, matching maxent.jar's output
        convention) with ±std error bars. The metric measures how
        much the fitted model's AUC drops when each variable's values
        are randomly shuffled — a direct measure of model reliance on
        each variable.
        """
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        # Short explanation strip so users know how this differs from
        # the Jackknife tab they just saw. Calling out the
        # correlation-underestimation caveat (Strobl et al. 2007)
        # up-front prevents the "but TWI shows lower here than in
        # Jackknife" confusion.
        info = QLabel(
            tr(
                "Model reliance: how much the trained model's predictions "
                "depend on each variable, measured by the AUC drop after "
                "shuffling that variable's values. Same metric reported "
                'by maxent.jar as "permutation importance". Fast to '
                "compute but can underestimate the importance of "
                "variables correlated with other variables in the model "
                "(Strobl et al. 2007); read alongside the Jackknife tab "
                "rather than alone."
            )
        )
        info.setWordWrap(True)
        # v0.1.7: grey background removed for visual consistency with
        # the Response Curves and Jackknife Importance tabs (which
        # also carry one-paragraph explanations now). Padding and
        # font-size kept so the strip reads as a description rather
        # than body prose.
        info.setStyleSheet("QLabel { padding: 6px; font-size: 9pt; }")
        v.addWidget(info)

        self._permutation_canvas_widget = QWidget()
        self._permutation_canvas_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(self._permutation_canvas_widget, stretch=1)

        # Action row: save the chart as PNG / data as CSV.
        btn_row = QHBoxLayout()
        self._permutation_save_png_btn = QPushButton(tr("Save PNG..."))
        self._permutation_save_png_btn.clicked.connect(self._on_save_permutation_png)
        self._permutation_save_png_btn.setEnabled(False)
        btn_row.addWidget(self._permutation_save_png_btn)

        self._permutation_save_csv_btn = QPushButton(tr("Save CSV..."))
        self._permutation_save_csv_btn.clicked.connect(self._on_save_permutation_csv)
        self._permutation_save_csv_btn.setEnabled(False)
        btn_row.addWidget(self._permutation_save_csv_btn)
        btn_row.addStretch()
        v.addLayout(btn_row)

        return w

    def _build_project_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)
        info = QLabel(
            tr(
                "Applies the trained model to environmental rasters to produce a "
                "habitat suitability map.\nUses the layers set in the ① Data tab."
            )
        )
        info.setWordWrap(True)
        v.addWidget(info)

        tr_row = QHBoxLayout()
        tr_row.addWidget(QLabel(tr("Output transform:")))
        self._proj_transform = QComboBox()
        self._proj_transform.addItems(["cloglog", "logistic", "raw"])
        # Changing the transform should immediately redraw the response
        # curve to match (ROC is rank-only and unaffected by monotonic
        # transforms, so we don't need to redraw it).
        self._proj_transform.currentIndexChanged.connect(self._on_transform_changed)
        tr_row.addWidget(self._proj_transform)
        tr_row.addStretch()
        v.addLayout(tr_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel(tr("Output raster:")))
        self._proj_path = QLineEdit()
        self._proj_path.setPlaceholderText(tr("Output path (.tif)"))
        # Default to an absolute path under the unified output folder,
        # so the user can read the field and know exactly where the
        # GeoTIFF will be saved. Same rationale as the Output Files
        # rows in the ② Parameters tab — see _default_output_dir.
        self._proj_path.setText(os.path.join(_default_output_dir(), "prediction.tif"))
        # Hover tooltip mirrors the field in case it gets truncated
        # in a narrow dock; kept in sync as the user edits.
        self._proj_path.setToolTip(self._proj_path.text())
        self._proj_path.textChanged.connect(self._proj_path.setToolTip)
        out_row.addWidget(self._proj_path, stretch=1)
        pbr = QPushButton("...")
        pbr.setMaximumWidth(30)
        pbr.clicked.connect(lambda: self._browse_save(self._proj_path, tr("GeoTIFF (*.tif)")))
        out_row.addWidget(pbr)
        v.addLayout(out_row)

        self._proj_load_chk = QCheckBox(tr("Auto-load result as QGIS layer"))
        self._proj_load_chk.setChecked(True)
        v.addWidget(self._proj_load_chk)

        # Auto-save analysis charts (Response Curves, ROC, Jackknife) as
        # PNG files alongside the prediction raster. Surfaced as a
        # checkbox so users discover the feature instead of having to
        # find PNG files in the output folder by accident. Default-on:
        # the files are small (~hundreds of KB) and the workflow that
        # produces them is exactly the moment a researcher would want
        # the figures, so opt-in friction would be net-negative for
        # most users.
        self._proj_save_charts_chk = QCheckBox(
            tr("Save analysis charts as PNG (Response Curves, ROC, Jackknife, Permutation)")
        )
        self._proj_save_charts_chk.setChecked(True)
        self._proj_save_charts_chk.setToolTip(
            tr(
                "When the projection finishes, four sets of PNG files are "
                "written next to the GeoTIFF:\n"
                "  • <name>_response_curves.png — one image with all response curves\n"
                "  • <name>_roc.png — ROC panel (training + CV folds + mean)\n"
                "  • <name>_jackknife.png — variable-importance bars (jackknife)\n"
                "  • <name>_permutation.png — variable-importance bars (permutation)\n"
                "Uncheck to skip this step entirely."
            )
        )
        v.addWidget(self._proj_save_charts_chk)

        self._proj_btn = QPushButton(tr("▶  Run Spatial Projection"))
        self._proj_btn.setMinimumHeight(34)
        self._proj_btn.clicked.connect(self._project_model)
        self._proj_btn.setEnabled(False)
        v.addWidget(self._proj_btn)

        # Deterministic progress bar (⑤ fix)
        self._proj_progress = QProgressBar()
        self._proj_progress.setRange(0, 100)
        self._proj_progress.setValue(0)
        self._proj_progress.setTextVisible(True)
        self._proj_progress.hide()
        v.addWidget(self._proj_progress)

        self._proj_status = QLabel("")
        self._proj_status.setWordWrap(True)
        v.addWidget(self._proj_status)
        v.addStretch()
        return w

    # =========================================================================
    # Results → Priority Sites for Survey
    # =========================================================================
    #
    # Post-prediction sampling of new candidate occurrence sites for
    # field surveys. Suitable cells (above a Pearson 2007 / Liu 2013
    # threshold) are filtered to be geographically distinct from
    # existing presences and from each other, then sampled in one of
    # two modes:
    #   • Discovery (default): pick the highest-suitability cells
    #     first — focus survey effort on the most likely habitat.
    #   • Validation (optional, Rhoden et al. 2017): pick equal
    #     numbers from each suitability quartile — evaluates the
    #     model across its full predicted gradient.
    # Results are reverse-geocoded via OpenStreetMap Nominatim and
    # added to QGIS as a styled vector layer.
    #
    # All academic decisions exposed here are referenced in the
    # accompanying manuscript's "Software description" section,
    # sub-section 2.3.

    def _build_priority_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)

        # ── Survey purpose (mode selection) ───────────────────────────────
        # Two distinct workflows that share the same downstream code path
        # but differ in how the candidate pool is shaped:
        #   • Discovery — find new populations: filter cells to a
        #     high-suitability band (≥ user-defined floor), then sample
        #     N sites within that band, either at random (default,
        #     statistically unbiased within the band) or by picking
        #     the top-N (greedy on suitability).
        #   • Validation — Rhoden et al. 2017: split cells above a
        #     threshold (T10/MTP/MaxSSS/Custom) into 4 quartiles and
        #     pick equal numbers from each, deliberately including
        #     lower-suitability cells so the survey can evaluate the
        #     model across its full predicted gradient.
        purpose_grp = QGroupBox(tr("Survey purpose"))
        pg = QVBoxLayout(purpose_grp)
        self._priority_mode_discovery = QRadioButton(
            tr("Discovery mode — find new populations in unsurveyed areas")
        )
        self._priority_mode_validation = QRadioButton(
            tr("Validation mode — test the suitability gradient")
        )
        self._priority_mode_discovery.setChecked(True)
        self._priority_mode_discovery.setToolTip(
            tr(
                "Sample sites within a high-suitability band, focused on "
                "the most likely habitat for the species."
            )
        )
        self._priority_mode_validation.setToolTip(
            tr(
                "Sample equal numbers from four suitability quartiles "
                "(Rhoden 2017). Will include lower-suitability sites by "
                "design — useful for evaluating the model gradient."
            )
        )
        self._priority_mode_discovery.toggled.connect(self._on_priority_mode_changed)
        pg.addWidget(self._priority_mode_discovery)
        pg.addWidget(self._priority_mode_validation)
        v.addWidget(purpose_grp)

        # ── Discovery settings ────────────────────────────────────────────
        # Visible only in Discovery mode. Two controls:
        #   1. Minimum suitability — a numeric floor for the candidate
        #      pool. Auto-filled to (raster_max × 0.9) when the
        #      prediction raster is loaded, so the default adapts to
        #      datasets whose suitability distribution doesn't reach
        #      1.0. The user can override.
        #   2. Sampling order — Random (statistically unbiased within
        #      the band) or Top-N (greedy on suitability).
        self._priority_discovery_grp = QGroupBox(tr("Discovery settings"))
        dg = QVBoxLayout(self._priority_discovery_grp)

        sf_row = QHBoxLayout()
        sf_row.addWidget(QLabel(tr("Minimum suitability:")))
        self._priority_min_suit = QLineEdit()
        suit_validator = QDoubleValidator(0.0, 1.0, 4)
        suit_validator.setNotation(QDoubleValidator.StandardNotation)
        self._priority_min_suit.setValidator(suit_validator)
        self._priority_min_suit.setText("0.9000")
        self._priority_min_suit.setMaximumWidth(120)
        self._priority_min_suit.setToolTip(
            tr(
                "Cells with suitability ≥ this value form the candidate "
                "pool. Default is auto-filled to (raster max × 0.9) when "
                "the prediction raster is selected."
            )
        )
        sf_row.addWidget(self._priority_min_suit)
        sf_row.addStretch()
        dg.addLayout(sf_row)

        order_row = QHBoxLayout()
        order_row.addWidget(QLabel(tr("Sampling order:")))
        self._priority_order_random = QRadioButton(tr("Random"))
        self._priority_order_topn = QRadioButton(tr("Top-N (highest first)"))
        self._priority_order_random.setChecked(True)
        order_row.addWidget(self._priority_order_random)
        order_row.addWidget(self._priority_order_topn)
        order_row.addStretch()
        dg.addLayout(order_row)
        v.addWidget(self._priority_discovery_grp)

        # ── Validation settings ───────────────────────────────────────────
        # Visible only in Validation mode. Reuses the existing
        # Pearson 2007 / Liu 2013 threshold options because they
        # legitimately define the lower bound of Q1 (the lowest
        # suitability quartile sampled).
        self._priority_validation_grp = QGroupBox(tr("Validation settings"))
        vg = QVBoxLayout(self._priority_validation_grp)

        tr_row = QHBoxLayout()
        tr_row.addWidget(QLabel(tr("Threshold method:")))
        self._priority_thr_method = QComboBox()
        self._priority_thr_method.addItems(
            [
                tr("10th percentile training presence (T10; Pearson 2007)"),
                tr("Minimum training presence (MTP; Pearson 2007)"),
                tr("Maximum sum of sensitivity + specificity (MaxSSS; Liu 2013)"),
                tr("Custom value..."),
            ]
        )
        self._priority_thr_method.currentIndexChanged.connect(
            self._on_priority_threshold_method_changed
        )
        tr_row.addWidget(self._priority_thr_method, stretch=1)
        vg.addLayout(tr_row)

        val_row = QHBoxLayout()
        val_row.addWidget(QLabel(tr("Computed threshold value:")))
        self._priority_thr_value = QLineEdit()
        thr_validator = QDoubleValidator(0.0, 1.0, 4)
        thr_validator.setNotation(QDoubleValidator.StandardNotation)
        self._priority_thr_value.setValidator(thr_validator)
        self._priority_thr_value.setText("0.0000")
        self._priority_thr_value.setMaximumWidth(120)
        self._priority_thr_value.setReadOnly(True)
        self._priority_thr_value.setEnabled(False)
        val_row.addWidget(self._priority_thr_value)
        val_row.addStretch()
        vg.addLayout(val_row)
        v.addWidget(self._priority_validation_grp)
        # Validation group starts hidden (Discovery is the default mode).
        self._priority_validation_grp.hide()

        # ── Sampling strategy (shared between modes) ──────────────────────
        sm_grp = QGroupBox(tr("Sampling strategy"))
        sm = QVBoxLayout(sm_grp)

        n_row = QHBoxLayout()
        n_row.addWidget(QLabel(tr("Number of priority sites:")))
        self._priority_n = QLineEdit("20")
        self._priority_n.setValidator(QIntValidator(1, 500))
        self._priority_n.setMaximumWidth(80)
        n_row.addWidget(self._priority_n)
        n_row.addStretch()
        sm.addLayout(n_row)

        d1_row = QHBoxLayout()
        d1_row.addWidget(QLabel(tr("Min. distance from existing presences (m):")))
        self._priority_d_pres = QLineEdit("1000")
        self._priority_d_pres.setValidator(QIntValidator(0, 1_000_000))
        self._priority_d_pres.setMaximumWidth(120)
        d1_row.addWidget(self._priority_d_pres)
        d1_row.addStretch()
        sm.addLayout(d1_row)

        d2_row = QHBoxLayout()
        d2_row.addWidget(QLabel(tr("Min. distance between sites (m):")))
        self._priority_d_site = QLineEdit("500")
        self._priority_d_site.setValidator(QIntValidator(0, 1_000_000))
        self._priority_d_site.setMaximumWidth(120)
        d2_row.addWidget(self._priority_d_site)
        d2_row.addStretch()
        sm.addLayout(d2_row)
        v.addWidget(sm_grp)

        # ── Reverse geocoding ─────────────────────────────────────────────
        gc_grp = QGroupBox(tr("Reverse geocoding"))
        gc = QVBoxLayout(gc_grp)
        self._priority_geocode = QCheckBox(
            tr("Add administrative address (province/city/district)")
        )
        self._priority_geocode.setChecked(True)
        self._priority_geocode.setToolTip(
            tr(
                "Uses OpenStreetMap Nominatim. No API key required. "
                "Rate limit 1 req/sec is applied automatically. "
                "Results carry © OpenStreetMap contributors attribution."
            )
        )
        gc.addWidget(self._priority_geocode)
        v.addWidget(gc_grp)

        # ── Output ────────────────────────────────────────────────────────
        out_grp = QGroupBox(tr("Output"))
        og = QVBoxLayout(out_grp)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel(tr("Output vector layer:")))
        self._priority_path = QLineEdit()
        self._priority_path.setPlaceholderText(tr("Output path (.gpkg)"))
        # Absolute path under the unified output folder for the same
        # reason as the other Output fields — make the destination
        # visible on screen instead of leaving it implicit.
        self._priority_path.setText(os.path.join(_default_output_dir(), "priority_sites.gpkg"))
        self._priority_path.setToolTip(self._priority_path.text())
        self._priority_path.textChanged.connect(self._priority_path.setToolTip)
        out_row.addWidget(self._priority_path, stretch=1)
        browse = QPushButton("...")
        browse.setMaximumWidth(30)
        browse.clicked.connect(
            lambda: self._browse_save(self._priority_path, tr("GeoPackage (*.gpkg)"))
        )
        out_row.addWidget(browse)
        og.addLayout(out_row)
        self._priority_autoload = QCheckBox(tr("Auto-add to QGIS project"))
        self._priority_autoload.setChecked(True)
        og.addWidget(self._priority_autoload)
        v.addWidget(out_grp)

        # Run button + status + progress
        self._priority_btn = QPushButton(tr("▶  Extract Priority Sites"))
        self._priority_btn.clicked.connect(self._run_priority_extraction)
        self._priority_btn.setEnabled(False)
        v.addWidget(self._priority_btn)

        self._priority_progress = QProgressBar()
        self._priority_progress.setRange(0, 100)
        self._priority_progress.setValue(0)
        self._priority_progress.setTextVisible(True)
        self._priority_progress.hide()
        v.addWidget(self._priority_progress)

        self._priority_status = QLabel("")
        self._priority_status.setWordWrap(True)
        v.addWidget(self._priority_status)
        v.addStretch()
        return w

    # ── Mode switch handler ──────────────────────────────────────────────
    def _on_priority_mode_changed(self, _checked: bool):
        """Show/hide the Discovery and Validation parameter groups so
        the user only sees the controls that affect their chosen
        sampling mode. Also refreshes the auto-filled defaults — when
        the user flips into Discovery mode the minimum-suitability
        floor is recomputed from the current prediction raster
        (raster_max × 0.9), and when they flip into Validation mode
        the threshold value is refreshed from the active method
        dropdown.
        """
        is_discovery = self._priority_mode_discovery.isChecked()
        self._priority_discovery_grp.setVisible(is_discovery)
        self._priority_validation_grp.setVisible(not is_discovery)
        if is_discovery:
            self._refresh_discovery_floor()
        else:
            self._on_priority_threshold_method_changed(self._priority_thr_method.currentIndex())

    def _refresh_discovery_floor(self):
        """Auto-fill the Discovery minimum-suitability floor to
        (raster_max × 0.9). This makes the default adapt to datasets
        whose suitability distribution doesn't reach 1.0 — a flat
        constant like 0.9 would leave such datasets with zero
        candidates. We read raster_max once and cache it on the dock
        instance so flipping back and forth doesn't re-open the file.
        """
        try:
            proj_path = self._proj_path.text().strip()
            if not proj_path or not os.path.isfile(proj_path):
                return
            cached_path = getattr(self, "_priority_raster_cached_path", None)
            if cached_path != proj_path:
                import numpy as np
                import rasterio

                with rasterio.open(proj_path) as src:
                    arr = src.read(1, masked=True)
                arr_valid = arr.compressed() if hasattr(arr, "compressed") else arr
                arr_valid = np.asarray(arr_valid, dtype=float)
                arr_valid = arr_valid[np.isfinite(arr_valid)]
                if arr_valid.size == 0:
                    return
                self._priority_raster_max = float(arr_valid.max())
                self._priority_raster_cached_path = proj_path
            floor = round(self._priority_raster_max * 0.9, 4)
            # Only overwrite when the field still holds the previous
            # auto-filled value or the placeholder default — never
            # clobber a value the user has typed.
            current = self._priority_min_suit.text().strip()
            previous_auto = getattr(self, "_priority_floor_last_auto", "0.9000")
            if current in ("", "0.9000", previous_auto):
                self._priority_min_suit.setText(f"{floor:.4f}")
                self._priority_floor_last_auto = f"{floor:.4f}"
        except Exception:
            # Reading the raster can fail (file lock, broken header,
            # missing rasterio). Falling back to the static 0.9 is
            # safe — it just means the user adjusts manually.
            pass

    # ── Threshold method handler ─────────────────────────────────────────
    def _on_priority_threshold_method_changed(self, idx: int):
        """Auto-fill the threshold value from training meta, or unlock
        the spin box when the user picks Custom."""
        # Map combo index → meta key. Order must match _build_priority_tab.
        method_keys = ("T10", "MTP", "MaxSSS", "Custom")
        key = method_keys[idx] if 0 <= idx < len(method_keys) else "T10"
        if key == "Custom":
            self._priority_thr_value.setReadOnly(False)
            self._priority_thr_value.setEnabled(True)
            return
        self._priority_thr_value.setReadOnly(True)
        self._priority_thr_value.setEnabled(False)
        meta = (
            getattr(self._model, "_qmaxent_meta", None) if getattr(self, "_model", None) else None
        )
        if meta is not None:
            thresholds = meta.get("thresholds", {}) or {}
            if key in thresholds:
                self._priority_thr_value.setText(f"{float(thresholds[key]):.4f}")
                return
        # Fallback: leave the previous value, the user can pick Custom.
        self._priority_thr_value.setText("0.0000")

    # ── Run handler ──────────────────────────────────────────────────────
    def _run_priority_extraction(self):
        """Validate inputs and start the PrioritySitesWorker."""
        try:
            self._run_priority_extraction_impl()
        except Exception as e:
            # Catch-all so any unexpected failure surfaces as a dialog
            # instead of bringing down QGIS. The full traceback goes to
            # the QGIS message log (View → Panels → Log Messages →
            # QMaxent) so users can paste it into a bug report.
            import traceback

            tb = traceback.format_exc()
            QgsMessageLog.logMessage(
                f"Priority Sites extraction failed:\n{tb}",
                "QMaxent",
                Qgis.Critical,
            )
            try:
                QMessageBox.critical(
                    self,
                    tr("Priority Sites"),
                    tr(
                        "Priority site extraction failed:\n{e}\n\n"
                        "Full traceback in: View → Panels → Log "
                        "Messages → QMaxent."
                    ).format(e=str(e)[:500]),
                )
            except Exception:
                pass
            # Best-effort UI cleanup so the user can try again.
            try:
                self._priority_btn.setEnabled(True)
                self._priority_progress.hide()
                self._priority_status.setText(tr("Failed."))
            except Exception:
                pass

    def _run_priority_extraction_impl(self):
        """Real run handler — wrapped by _run_priority_extraction so
        any unhandled exception goes to the log instead of crashing
        QGIS."""
        # 1. We need a trained model with metadata.
        if not getattr(self, "_model", None):
            QMessageBox.warning(
                self, tr("Priority Sites"), tr("Train a model first (or load an existing .pkl).")
            )
            return
        # 2. We need a prediction raster — the Spatial Projection tab
        #    must have been run, OR the user can point at a .tif they
        #    saved earlier. We use the projection-tab path as the
        #    canonical input.
        proj_path = self._proj_path.text().strip()
        if not proj_path or not os.path.isfile(proj_path):
            QMessageBox.warning(
                self,
                tr("Priority Sites"),
                tr(
                    "Run a spatial projection first — the priority "
                    "sampling needs a prediction raster.\n\n"
                    "Tip: ④ Results → Spatial Projection → ▶ Run Spatial Projection."
                ),
            )
            return
        # 3. Presence layer must still be set in the ① Data tab.
        pres_layer = (
            self._pres_combo.currentLayer() if hasattr(self._pres_combo, "currentLayer") else None
        )
        if pres_layer is None:
            QMessageBox.warning(
                self, tr("Priority Sites"), tr("No presence layer selected (① Data tab).")
            )
            return
        # Convert presence layer to a list of (lon, lat) — the layer's
        # geometry is reprojected to EPSG:4326 to match the lat/lon
        # convention used inside priority_sites.py.
        try:
            from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform
            from qgis.core import QgsProject as _QP

            wgs = QgsCoordinateReferenceSystem("EPSG:4326")
            xform = QgsCoordinateTransform(pres_layer.crs(), wgs, _QP.instance())
            presence_xy = []
            for feat in pres_layer.getFeatures():
                geom = feat.geometry()
                if geom is None or geom.isEmpty():
                    continue
                # Handle both Point and MultiPoint layers — users
                # sometimes import MultiPoint from GBIF or GPS data.
                # asPoint() raises on multi-geometries, so we branch
                # on the actual type.
                try:
                    if geom.isMultipart():
                        pts = geom.asMultiPoint()
                        if not pts:
                            continue
                        pt = pts[0]  # use the first part
                    else:
                        pt = geom.asPoint()
                except Exception:
                    continue
                p = xform.transform(pt)
                presence_xy.append((float(p.x()), float(p.y())))
        except Exception as e:
            QMessageBox.critical(
                self, tr("Priority Sites"), tr("Could not read presence layer:\n{e}").format(e=e)
            )
            return
        if not presence_xy:
            QMessageBox.warning(
                self, tr("Priority Sites"), tr("Presence layer has no point features.")
            )
            return

        # Threshold value & method label.
        # The numeric inputs are now QLineEdits, so we parse defensively
        # — empty text or a half-typed value would raise ValueError on
        # float() / int() and the run handler would die mid-validation.
        # We catch those here and surface a clear message instead.
        def _parse_float(le, default):
            try:
                t = le.text().strip()
                return float(t) if t else default
            except Exception:
                return default

        def _parse_int(le, default):
            try:
                t = le.text().strip()
                return int(float(t)) if t else default
            except Exception:
                return default

        # Resolve mode-dependent parameters: threshold (= candidate
        # floor) and stratify_by_quartile flag.
        is_discovery = self._priority_mode_discovery.isChecked()
        if is_discovery:
            method = "MinSuit"
            threshold = _parse_float(self._priority_min_suit, 0.9)
            stratify = False
            sampling_order = "random" if self._priority_order_random.isChecked() else "topn"
        else:
            method_keys = ("T10", "MTP", "MaxSSS", "Custom")
            method = method_keys[self._priority_thr_method.currentIndex()]
            threshold = _parse_float(self._priority_thr_value, 0.0)
            stratify = True
            sampling_order = "topn"  # ignored when stratify=True

        if threshold <= 0.0:
            QMessageBox.warning(
                self,
                tr("Priority Sites"),
                tr(
                    "Threshold value is 0 or unset. Enter a value > 0 "
                    "or pick a different threshold method."
                ),
            )
            return

        n = max(1, _parse_int(self._priority_n, 20))
        d_pres = _parse_float(self._priority_d_pres, 1000.0)
        d_site = _parse_float(self._priority_d_site, 500.0)
        do_geocode = bool(self._priority_geocode.isChecked())
        # Tell the user how long Nominatim will take so they don't
        # think the plugin froze. The 1 req/sec policy is hard-coded.
        if do_geocode and n >= 30:
            secs = n * 1.05
            mins = int(secs // 60)
            rem = int(secs - mins * 60)
            QMessageBox.information(
                self,
                tr("Priority Sites"),
                tr(
                    "Reverse geocoding {n} sites at 1 request/second "
                    "will take about {mins}m {rem}s. The progress bar "
                    "will report each step."
                ).format(n=n, mins=mins, rem=rem),
            )

        from ..workers.priority_sites_worker import PrioritySitesWorker

        self._priority_worker = PrioritySitesWorker(
            prediction_path=proj_path,
            presence_xy=presence_xy,
            threshold=threshold,
            threshold_method=method,
            n_sites=n,
            min_dist_from_presence_m=d_pres,
            min_dist_between_sites_m=d_site,
            stratify_by_quartile=stratify,
            sampling_order=sampling_order,
            do_geocode=do_geocode,
            random_seed=(self._seed_spin.value() if self._seed_check.isChecked() else None),
            parent=self,
        )
        self._priority_worker.progress.connect(self._on_priority_progress)
        self._priority_worker.log.connect(self._log_append)
        self._priority_worker.finished.connect(
            lambda ok, msg, rows: self._on_priority_finished(ok, msg, rows, method, threshold)
        )

        self._priority_btn.setEnabled(False)
        self._priority_progress.setValue(0)
        self._priority_progress.show()
        self._priority_status.setText(tr("Starting..."))
        self._priority_worker.start()

    def _on_priority_progress(self, pct: int, msg: str):
        self._priority_progress.setValue(pct)
        self._priority_status.setText(msg)

    def _on_priority_finished(self, ok: bool, msg: str, rows: list, method: str, threshold: float):
        self._priority_worker = None
        if not ok:
            self._priority_btn.setEnabled(True)
            self._priority_progress.hide()
            self._priority_status.setText(tr("Failed."))
            QMessageBox.critical(
                self,
                tr("Priority Sites"),
                tr("Priority site extraction failed:\n{msg}").format(msg=msg),
            )
            return

        n = len(rows)
        if n == 0:
            self._priority_btn.setEnabled(True)
            self._priority_progress.hide()
            self._priority_status.setText(tr("No sites extracted."))
            return

        # Decide whether to geocode now (main-thread QTimer) or skip
        # straight to saving. Geocoding deliberately runs on the main
        # thread instead of inside the worker because Python 3.12 +
        # Windows + QThread + urllib repeatedly produced a native
        # access-violation crash inside socket/SSL cleanup at the
        # 19/20-th HTTP call. Driving geocoding from QTimer.singleShot
        # keeps Qt's own event loop in charge of timing, native HTTP
        # stays on the main thread, and the GUI stays responsive
        # because each step is a single 1-second-or-less blocking
        # call between event-loop ticks.
        if self._priority_geocode.isChecked():
            self._geocode_pending = list(rows)
            self._geocode_done = []
            self._geocode_total = len(rows)
            self._geocode_method = method
            self._geocode_threshold = threshold
            # Repurpose the same progress bar and Run button.
            self._priority_progress.setValue(0)
            self._priority_progress.show()
            self._priority_status.setText(
                tr("Reverse geocoding 0/{n}…").format(n=self._geocode_total)
            )
            # _priority_btn stays disabled while geocoding runs.
            self._geocode_one_step()
        else:
            # No geocoding asked — save immediately.
            self._priority_btn.setEnabled(True)
            self._save_and_finalise_priority(rows, method, threshold)

    def _geocode_one_step(self):
        """Reverse-geocode one queued site, then re-arm via QTimer.

        Runs on the main thread so the native HTTP / SSL path stays
        out of the QThread machinery that was crashing on Windows.
        Each call is up to ~1 second of blocking I/O, but Qt's event
        loop ticks between calls (we re-enter via QTimer.singleShot),
        so the GUI stays responsive — buttons remain clickable and
        the progress bar updates.
        """
        from qgis.PyQt.QtCore import QTimer

        if not getattr(self, "_geocode_pending", None):
            # Done — finalise.
            rows = list(getattr(self, "_geocode_done", []))
            method = getattr(self, "_geocode_method", "")
            threshold = float(getattr(self, "_geocode_threshold", 0.0))
            n_geocoded = sum(1 for r in rows if r.get("display_name"))
            self._log_append(
                f"Reverse geocoding complete — {n_geocoded}/{len(rows)} resolved (Nominatim)"
            )
            # Free the queue state so a second run starts clean.
            self._geocode_pending = None
            self._geocode_done = None
            self._priority_btn.setEnabled(True)
            self._save_and_finalise_priority(rows, method, threshold)
            return

        site = self._geocode_pending.pop(0)
        try:
            from ..bridge.priority_sites import reverse_geocode_one

            addr = reverse_geocode_one(site["lat"], site["lon"])
        except Exception:
            # Best-effort; never fatal. The site row is kept with
            # empty address fields so the GeoPackage schema is intact.
            addr = {
                "country": "",
                "province": "",
                "city_county": "",
                "district": "",
                "display_name": "",
            }
        # Merge: site already has empty address fields from the worker;
        # the spread updates them in place.
        self._geocode_done.append({**site, **addr})

        completed = len(self._geocode_done)
        total = self._geocode_total
        pct = int(completed / max(total, 1) * 100)
        self._priority_progress.setValue(pct)
        self._priority_status.setText(tr("Reverse geocoding {i}/{n}…").format(i=completed, n=total))

        # Re-arm. 1050 ms ≈ Nominatim's 1 req/sec policy with a small
        # safety margin. Using QTimer.singleShot rather than time.sleep
        # is the whole point — Qt drains its event queue between ticks
        # instead of being held off by a sleeping thread.
        if self._geocode_pending:
            QTimer.singleShot(1050, self._geocode_one_step)
        else:
            # Last item already processed; finalise on the next tick
            # so the UI gets one final paint at 100%.
            QTimer.singleShot(0, self._geocode_one_step)

    def _save_and_finalise_priority(self, rows: list, method: str, threshold: float):
        """Write GeoPackage, optionally add to project, update status."""
        out_path = self._priority_path.text().strip() or "priority_sites.gpkg"
        if not os.path.isabs(out_path):
            out_path = os.path.join(_default_output_dir(), out_path)
        try:
            self._save_priority_sites_gpkg(rows, out_path)
        except Exception as e:
            QMessageBox.critical(
                self, tr("Priority Sites"), tr("Could not write output:\n{e}").format(e=e)
            )
            return

        # Add to project (with simple symbology + labels) if asked.
        if self._priority_autoload.isChecked():
            try:
                self._load_priority_sites_layer(out_path)
            except Exception as e:
                self._log_append(tr("Could not auto-load priority sites layer: {e}").format(e=e))

        n_geocoded = sum(1 for r in rows if r.get("display_name"))
        self._priority_status.setText(
            tr(
                "✓ {n} priority sites extracted ({m} = {t:.4f}); {g}/{n} geocoded. Output: {p}"
            ).format(n=len(rows), m=method, t=threshold, g=n_geocoded, p=out_path)
        )

    # ── Output writing ───────────────────────────────────────────────────
    def _save_priority_sites_gpkg(self, rows: list, out_path: str):
        """Write the result table to a GeoPackage point layer.

        Schema: id (int), lat, lon, suitability, quartile, country,
        province, city_county, district, display_name. The
        ``quartile`` field holds the suitability quartile id 0..3 in
        Validation mode (0 = lowest band above threshold, 3 =
        highest); Discovery mode always writes 0. The display_name
        is the full Nominatim label and acts as a self-contained
        attribution field for OSM data.
        """
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsFeature,
            QgsField,
            QgsFields,
            QgsGeometry,
            QgsPointXY,
            QgsVectorFileWriter,
            QgsWkbTypes,
        )
        from qgis.PyQt.QtCore import QVariant

        # Make sure the output directory exists. _default_output_dir()
        # creates ~/qmaxent_output, but a user can type any path into
        # the box, including a folder that hasn't been made yet.
        try:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        except Exception:
            pass

        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("lat", QVariant.Double))
        fields.append(QgsField("lon", QVariant.Double))
        fields.append(QgsField("suitability", QVariant.Double))
        # Suitability quartile id (0..3) so the layer can be styled
        # by quartile in Validation mode. Stored as Int — categorized
        # symbology renders integer fields more reliably than strings
        # across QGIS versions.
        fields.append(QgsField("quartile", QVariant.Int))
        fields.append(QgsField("country", QVariant.String))
        fields.append(QgsField("province", QVariant.String))
        fields.append(QgsField("city_county", QVariant.String))
        fields.append(QgsField("district", QVariant.String))
        fields.append(QgsField("display_name", QVariant.String))

        wgs = QgsCoordinateReferenceSystem("EPSG:4326")
        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = "GPKG"
        from qgis.core import QgsProject as _QP

        ctx = _QP.instance().transformContext()
        writer = QgsVectorFileWriter.create(out_path, fields, QgsWkbTypes.Point, wgs, ctx, opts)
        # Some QGIS builds return a writer that is None on failure
        # (locked file, read-only directory, GPKG driver missing).
        # Calling .hasError() on None would AttributeError-crash; check
        # the object itself first and surface a clear message.
        if writer is None:
            raise RuntimeError(
                f"Could not create GeoPackage writer for {out_path}. "
                f"Check that the path is writable and not locked by "
                f"another QGIS layer."
            )
        if writer.hasError() != QgsVectorFileWriter.NoError:
            err = writer.errorMessage()
            del writer
            raise RuntimeError(f"GPKG writer error: {err}")
        n_written = 0
        try:
            for i, r in enumerate(rows, start=1):
                f = QgsFeature(fields)
                f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(r["lon"]), float(r["lat"]))))
                f.setAttributes(
                    [
                        i,
                        float(r["lat"]),
                        float(r["lon"]),
                        float(r["suitability"]),
                        int(r.get("quartile", 0)),
                        r.get("country", ""),
                        r.get("province", ""),
                        r.get("city_county", ""),
                        r.get("district", ""),
                        r.get("display_name", ""),
                    ]
                )
                # addFeature returns False on failure (attribute type
                # mismatch, invalid geometry, OGR-level error). Track
                # the count and surface a warning if any drop.
                if writer.addFeature(f):
                    n_written += 1
        finally:
            # Always release the writer, even on per-feature failures,
            # otherwise the GPKG file stays locked and the next save
            # attempt fails mysteriously.
            del writer
        if n_written < len(rows):
            self._log_append(
                tr(
                    "⚠ {dropped} of {total} priority sites could not be "
                    "written to the GeoPackage and were dropped."
                ).format(dropped=len(rows) - n_written, total=len(rows))
            )

    def _load_priority_sites_layer(self, gpkg_path: str):
        """Load the priority sites layer with simple, robust styling.

        Default: single red marker (#cb181d, 3.5pt). No labels —
        administrative names are still in the attribute table
        (province, city_county, district, display_name) so users can
        switch on labelling manually from Layer Properties → Labels.

        Validation mode enhancement (v0.1.7): when the GPKG carries
        ≥2 distinct quartile values (i.e. the user ran Validation
        mode with stratify_by_quartile=True), we additionally attempt
        categorized symbology by the ``quartile`` field — green for
        the highest-suitability quartile shading to red for the
        lowest. The whole categorized step is wrapped in a defensive
        try/except because a previous QGIS 3.44 + Python 3.12 build
        produced a post-GeoPackage-write segfault when applying
        graduated-by-suitability symbology; if that recurs with
        categorized symbology too we fall back to the simple single
        marker (same as Discovery mode) without aborting.
        """
        from qgis.core import (
            QgsProject,
            QgsSymbol,
            QgsVectorLayer,
        )
        from qgis.PyQt.QtGui import QColor

        layer = QgsVectorLayer(gpkg_path, "Priority Sites for Survey", "ogr")
        if not layer.isValid():
            raise RuntimeError(f"Invalid layer: {gpkg_path}")

        # First apply the simple fallback styling. Any subsequent
        # categorized attempt either replaces the renderer (success)
        # or leaves this in place (failure).
        try:
            sym = QgsSymbol.defaultSymbol(layer.geometryType())
            if sym is not None:
                sym.setColor(QColor("#cb181d"))
                try:
                    sym.symbolLayer(0).setSize(3.5)
                except Exception:
                    pass
                layer.renderer().setSymbol(sym.clone())
        except Exception:
            pass

        # Detect Validation mode by scanning the quartile column.
        # Validation mode produces a mix of quartile ids (0..3);
        # Discovery mode writes 0 for every site.
        try:
            qvals = set()
            for f in layer.getFeatures():
                q = f["quartile"]
                if q is not None:
                    qvals.add(int(q))
            wants_categorized = len(qvals) >= 2
        except Exception:
            # If the feature scan fails (e.g. legacy GPKG written by
            # an earlier QMaxent version that has no quartile field),
            # stay with the simple single-marker styling.
            wants_categorized = False

        if wants_categorized:
            try:
                from qgis.core import (
                    QgsCategorizedSymbolRenderer,
                    QgsRendererCategory,
                )

                # Quartile palette — sequential YlOrRd (yellow→red,
                # warm sequential color scheme). Q4 (highest
                # suitability) is red — the SAME red used for
                # Discovery mode's single-marker styling — so a user
                # toggling between Discovery and Validation reads
                # "red = highest priority" consistently across both
                # modes. Q1 (lowest) drops to a light yellow to
                # signal "low-priority verification site" without
                # competing visually with the Q4 sites.
                palette = {
                    3: ("#cb181d", "Q4 (highest)"),
                    2: ("#fd8d3c", "Q3"),
                    1: ("#fdae6b", "Q2"),
                    0: ("#fdd49e", "Q1 (lowest)"),
                }
                categories = []
                for q in sorted(qvals, reverse=True):
                    color_hex, label = palette.get(q, ("#888888", f"Q{q + 1}"))
                    s = QgsSymbol.defaultSymbol(layer.geometryType())
                    if s is None:
                        continue
                    s.setColor(QColor(color_hex))
                    try:
                        s.symbolLayer(0).setSize(3.5)
                    except Exception:
                        pass
                    categories.append(QgsRendererCategory(q, s, label))
                if categories:
                    renderer = QgsCategorizedSymbolRenderer("quartile", categories)
                    layer.setRenderer(renderer)
            except Exception as e:
                # Categorized symbology failed — surface a one-line
                # log message but keep the layer on its single-marker
                # fallback. This matches the v0.1.7 design intent:
                # never let a styling glitch block the user from
                # seeing their sites on the map.
                self._log_append(
                    tr("(Categorized symbology fell back to single marker: {e})").format(e=e)
                )

        QgsProject.instance().addMapLayer(layer)

    def _on_presence_changed(self, layer):
        if layer is None:
            self._pres_info.setText(tr("Select a layer to see point count."))
            return
        try:
            n = layer.featureCount()
            self._pres_info.setText(tr("{n} presence points loaded").format(n=f"{n:,}"))
        except Exception:
            self._pres_info.setText(tr("Cannot read layer info."))

    def _add_raster_from_project(self):
        from ..bridge.raster_bridge import detect_categorical_hint

        layers = QgsProject.instance().mapLayers().values()
        rasters = [l for l in layers if isinstance(l, QgsRasterLayer)]
        if not rasters:
            self._log_append(tr("No raster layers in project."))
            return
        existing = {
            self._raster_list.item(i).data(self._ROLE_LAYER_ID)
            for i in range(self._raster_list.count())
        }
        added = 0
        for layer in rasters:
            if layer.id() in existing:
                continue
            # Default categorical = False. Only flip to True when the raster
            # file carries an EXPLICIT categorical metadata hint (LAYER_TYPE
            # == 'thematic', a colour palette, or non-empty CATEGORY_NAMES).
            # The conservative rule keeps continuous-but-integer rasters
            # (e.g. elevation in metres) safely on the continuous side.
            try:
                src_path = layer.source().split("|")[0]
                is_cat = detect_categorical_hint(src_path)
            except Exception:
                is_cat = False
            self._add_raster_item(
                layer_name=layer.name(),
                layer_id=layer.id(),
                is_categorical=is_cat,
            )
            added += 1
        if added == 0:
            self._log_append(tr("No new raster layers to add."))

    def _remove_selected_rasters(self):
        # Iterate over a snapshot of the selected items because removing
        # them mutates the live selectedItems() list. Also clean up the
        # custom row widget explicitly — takeItem() unlinks the item but
        # leaves the setItemWidget()-attached widget in Qt's deferred
        # delete queue, which is technically correct but leaves a stale
        # reference until the next event loop tick.
        for item in list(self._raster_list.selectedItems()):
            row = self._raster_list.row(item)
            row_w = self._raster_list.itemWidget(item)
            if row_w is not None:
                self._raster_list.removeItemWidget(item)
                row_w.deleteLater()
            self._raster_list.takeItem(row)

    def _move_raster(self, direction: int):
        """Move the selected raster(s) up (-1) or down (+1) in the list.

        Block-preserving multi-selection: when the user selects a
        contiguous range of rows (e.g. rows 2–6) and presses ▲ / ▼,
        the whole block shifts as one unit and the selection stays
        on those same five rasters at their new positions. Multiple
        non-adjacent blocks are each moved independently.

        Implementation: for each contiguous selected block, swap the
        adjacent non-selected row (above the block when moving up,
        below the block when moving down) past the block. This
        preserves block contents and leaves the selection on the
        same rasters.

        We can't use ``takeItem`` + ``insertItem`` here because the
        list rows carry a custom ``setItemWidget`` widget (the name
        label + categorical toggle). ``takeItem`` detaches the item
        from the widget, and reinserting the item yields an empty
        grey row. Instead we leave the items in place and swap the
        *contents* of adjacent rows.
        """
        selected_rows = sorted(
            {self._raster_list.row(it) for it in self._raster_list.selectedItems()}
        )
        if not selected_rows:
            # Fallback: no explicit selection — operate on the
            # current row (keeps single-click + ▲ / ▼ workflow
            # working as in v0.1.6).
            cur = self._raster_list.currentRow()
            if cur < 0:
                return
            selected_rows = [cur]

        # Group consecutive rows into contiguous blocks.
        # e.g. [1, 2, 3, 5, 7, 8] -> [[1,2,3], [5], [7,8]]
        blocks = []
        cur_block = [selected_rows[0]]
        for r in selected_rows[1:]:
            if r == cur_block[-1] + 1:
                cur_block.append(r)
            else:
                blocks.append(cur_block)
                cur_block = [r]
        blocks.append(cur_block)

        n = self._raster_list.count()

        # Process blocks in order that avoids them colliding with
        # each other:
        #   • up   (direction = -1): top-down — topmost block moves
        #     first, freeing the slot above for any block below.
        #   • down (direction = +1): bottom-up — bottommost block
        #     moves first, freeing the slot below for any block
        #     above.
        ordered = blocks if direction < 0 else list(reversed(blocks))
        new_selected = []

        for block in ordered:
            top, bot = block[0], block[-1]
            if direction < 0:
                # Moving up: need a non-selected row at top - 1.
                if top == 0:
                    new_selected.extend(block)
                    continue
                # Pull the row above the block down past the block
                # by walking it through one swap at a time. Visual
                # effect: the block moves up by one row, and the
                # row that was above the block now sits below it.
                for r in range(top - 1, bot):
                    self._swap_raster_rows(r, r + 1)
                new_selected.extend([r - 1 for r in block])
            else:
                # Moving down: need a non-selected row at bot + 1.
                if bot == n - 1:
                    new_selected.extend(block)
                    continue
                # Pull the row below the block up past the block,
                # walking it through one swap at a time from the
                # bottom-of-block edge upward.
                for r in range(bot + 1, top, -1):
                    self._swap_raster_rows(r, r - 1)
                new_selected.extend([r + 1 for r in block])

        # Restore selection exactly on the new positions of the
        # rasters the user had selected — same five (or whatever)
        # rasters, just at new row numbers.
        #
        # We use QItemSelectionModel.select() with ClearAndSelect|Rows
        # rather than per-item setSelected(True) because the latter
        # can be silently undone in some Qt builds when followed by
        # setCurrentRow / setCurrentItem (which behave like
        # single-selection commands on ExtendedSelection lists). The
        # selection-model approach applies the entire multi-row
        # selection atomically and survives any focus changes.
        if new_selected:
            from qgis.PyQt.QtCore import QItemSelection, QItemSelectionModel

            new_rows = sorted(set(new_selected))
            sel_model = self._raster_list.selectionModel()
            selection = QItemSelection()
            for r in new_rows:
                idx = self._raster_list.model().index(r, 0)
                selection.select(idx, idx)
            sel_model.select(
                selection,
                QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
            )
            # Anchor the current index on the topmost selected row
            # without disturbing the selection set we just applied
            # (NoUpdate keeps the multi-selection visible). The
            # current index is what ▲ / ▼ keyboard shortcuts and
            # the "outline" focus rectangle use.
            top_idx = self._raster_list.model().index(new_rows[0], 0)
            sel_model.setCurrentIndex(top_idx, QItemSelectionModel.NoUpdate)

    def _swap_raster_rows(self, row: int, new_row: int):
        """Swap the contents of two adjacent raster-list rows.

        Used by ``_move_raster`` to implement up/down arrows for
        single and multi-selection alike. Swaps the item's UserRole
        data, the visible label text, and the categorical toggle
        state — leaving the QListWidgetItem and its setItemWidget
        association in place so we never hit the takeItem blank-row
        bug.
        """
        a = self._raster_list.item(row)
        b = self._raster_list.item(new_row)
        wa = self._raster_list.itemWidget(a)
        wb = self._raster_list.itemWidget(b)
        if wa is None or wb is None:
            # Defensive: if the widgets aren't available for some
            # reason, fall back to the simple takeItem path. The
            # blank-row bug shouldn't happen on rows without a custom
            # widget anyway.
            item = self._raster_list.takeItem(row)
            self._raster_list.insertItem(new_row, item)
            return

        # 1. Swap UserRole data on the items themselves.
        for role in (self._ROLE_LAYER_ID, self._ROLE_IS_CATEG):
            ad = a.data(role)
            bd = b.data(role)
            a.setData(role, bd)
            b.setData(role, ad)

        # 2. Swap visible label text. The label is the first child
        #    QLabel inside the row widget — the QHBoxLayout in
        #    _add_raster_item adds it before the toggle.
        la = wa.findChild(QLabel)
        lb = wb.findChild(QLabel)
        if la is not None and lb is not None:
            ta, tb = la.text(), lb.text()
            la.setText(tb)
            lb.setText(ta)

        # 3. Swap toggle button state. The toggle's `toggled` signal
        #    fires `_on_categorical_toggled`, which writes the new
        #    state back into the item we just rewrote — that's
        #    consistent, so nothing extra to do.
        ba = wa.findChild(QPushButton)
        bb = wb.findChild(QPushButton)
        if ba is not None and bb is not None:
            cba, cbb = ba.isChecked(), bb.isChecked()
            # blockSignals so swap doesn't double-fire categorical
            # change handlers (we already swapped the role data).
            ba.blockSignals(True)
            bb.blockSignals(True)
            ba.setChecked(cbb)
            bb.setChecked(cba)
            ba.blockSignals(False)
            bb.blockSignals(False)
            # Manually refresh the toggle text now that signals are
            # silenced (blockSignals also suppressed the visual sync).
            self._sync_categorical_toggle(ba)
            self._sync_categorical_toggle(bb)

    # =========================================================================
    # Raster consistency: explicit Check + Harmonize workflow
    # =========================================================================
    #
    # Following biomod2's BIOMOD_FormatingData and SDMSelect's
    # Prepare_r_multi, raster harmonization is an explicit user-driven
    # step, not an opaque automatic one. The user clicks
    # "Check Raster Consistency" to inspect the current set; on
    # mismatch a "Harmonize to Folder…" button appears. The harmonize
    # action runs in a background worker, writes outputs to the chosen
    # folder, and replaces the raster list with the harmonized copies
    # so the next press of "Run Maxent" trains on aligned data.

    def _on_check_consistency(self):
        """Inspect the current rasters and update the status label.

        Shows ✓ if everything aligns, otherwise lists the dimensions
        that disagree (CRS / extent / resolution) and reveals the
        "Harmonize to Folder…" button.
        """
        from ..bridge.raster_bridge import (
            check_raster_consistency,
            layer_to_path,
        )

        layers = self._get_raster_layers()
        if not layers:
            self._consistency_lbl.setText(
                tr("Status: add at least one raster, then click Check Raster Consistency.")
            )
            self._consistency_lbl.setStyleSheet("color: gray;")
            self._harmonize_btn.setVisible(False)
            return

        paths = [layer_to_path(lyr) for lyr in layers]
        result = check_raster_consistency(paths)

        if result.get("error"):
            self._consistency_lbl.setText(
                tr("Status: ⚠ Could not read rasters ({err}).").format(err=result["error"])
            )
            self._consistency_lbl.setStyleSheet("color: #B23A2A;")
            self._harmonize_btn.setVisible(False)
            return

        if result.get("is_consistent"):
            first = result["rasters"][0]
            self._consistency_lbl.setText(
                tr("Status: ✓ All {n} rasters share grid (CRS: {crs}, resolution: {res}).").format(
                    n=len(paths),
                    crs=str(first["crs"]) if first["crs"] else "—",
                    res=f"{first['res'][0]:g} × {first['res'][1]:g}",
                )
            )
            self._consistency_lbl.setStyleSheet("color: #2A8A2A;")
            self._harmonize_btn.setVisible(False)
        else:
            mismatches = []
            if not result["crs_uniform"]:
                mismatches.append(tr("CRS"))
            if not result["extent_uniform"]:
                mismatches.append(tr("extent"))
            if not result["resolution_uniform"]:
                mismatches.append(tr("resolution"))
            self._consistency_lbl.setText(
                tr(
                    "Status: ⚠ Grid mismatch — {dims} differ across "
                    'rasters. Click "Harmonize to Folder…" to align.'
                ).format(dims=", ".join(mismatches))
            )
            self._consistency_lbl.setStyleSheet("color: #B23A2A;")
            self._harmonize_btn.setVisible(True)

    def _on_harmonize_clicked(self):
        """Prompt for an output folder and harmonize in a worker thread."""
        from ..bridge.raster_bridge import layer_to_path
        from ..workers.harmonize_worker import HarmonizeWorker

        layers = self._get_raster_layers()
        if not layers:
            return
        paths = [layer_to_path(lyr) for lyr in layers]

        # Categorical flags from the per-raster dropdowns; used so the
        # worker resamples categorical rasters with nearest-neighbor.
        cat_idx = self._get_categorical_indices()

        # Default output dir: <first raster's dir>/qmaxent_harmonized/
        # We create the folder *up front* so the file dialog can land
        # directly inside it — without that, getExistingDirectory just
        # opens the parent directory and the user has to create the
        # subfolder by hand on Windows. Pre-creating it makes the
        # default obvious: the dialog opens with the new folder
        # already selected and "OK" is one click away. The user can
        # still navigate elsewhere if they want a different location;
        # the empty qmaxent_harmonized folder will simply be left
        # behind in that case (harmless).
        default_dir = os.path.join(
            os.path.dirname(os.path.abspath(paths[0])),
            "qmaxent_harmonized",
        )
        try:
            os.makedirs(default_dir, exist_ok=True)
        except OSError:
            # If the parent directory isn't writable, fall back to the
            # parent itself — the dialog will still open at a sensible
            # location and the user can pick anywhere they have write
            # access.
            default_dir = os.path.dirname(os.path.abspath(paths[0]))
        chosen = QFileDialog.getExistingDirectory(
            self, tr("Select harmonized output folder"), default_dir
        )
        if not chosen:
            return

        try:
            os.makedirs(chosen, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(
                self,
                tr("Harmonize Rasters"),
                tr("Could not create output folder:\n{e}").format(e=e),
            )
            return

        # Modal progress dialog. We use QProgressDialog rather than
        # touching the training dock's progress bar because
        # harmonization is its own task with its own success/failure
        # path; mixing it into the training bar would hide what's
        # actually running.
        progress = QProgressDialog(
            tr("Harmonizing rasters..."),
            tr("Cancel"),
            0,
            100,
            self,
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        worker = HarmonizeWorker(
            raster_paths=paths,
            output_dir=chosen,
            categorical_indices=cat_idx,
            template_idx=0,
            parent=self,
        )
        # Hold a reference so the worker is not GC'd mid-run.
        self._harmonize_worker = worker

        def _on_progress(pct, msg):
            progress.setValue(pct)
            progress.setLabelText(msg)

        def _on_finished(ok, msg, new_paths):
            progress.setValue(100)
            progress.close()
            self._harmonize_worker = None
            if not ok:
                # User-cancel is not an error — show a non-modal status
                # message and return without the alarming red dialog.
                if msg == "Cancelled":
                    self._consistency_lbl.setText(tr("Status: harmonization cancelled by user."))
                    self._consistency_lbl.setStyleSheet("color: #B23A2A;")
                    return
                QMessageBox.critical(
                    self,
                    tr("Harmonize Rasters"),
                    tr("Harmonization failed:\n{msg}").format(msg=msg),
                )
                return

            # v0.1.7: present the user a 3-option choice instead of
            # silently swapping. The previous behavior cleared the
            # ① Data raster list but left the *unaligned originals*
            # in the QGIS project panel, leading to:
            #   - cluttered Layers panel (12 layers instead of 6)
            #   - confusion when re-using "Add from project" later
            #   - mismatched ① Data list vs QGIS project state
            # Three options let the user pick the right tradeoff for
            # their workflow:
            #   Replace — clean swap, recommended
            #   Add     — keep originals visible alongside aligned
            #   Cancel  — files on disk only, nothing added
            mbox = QMessageBox(self)
            mbox.setWindowTitle(tr("Harmonize Rasters"))
            mbox.setIcon(QMessageBox.Question)
            mbox.setText(
                tr("Harmonized {n} raster(s) to:\n{path}").format(n=len(new_paths), path=chosen)
            )
            mbox.setInformativeText(
                tr("Replace the unaligned originals in this project with the harmonized versions?")
            )
            btn_replace = mbox.addButton(tr("Replace"), QMessageBox.AcceptRole)
            mbox.addButton(
                tr("Add as additional layers"),
                QMessageBox.ActionRole,
            )
            btn_cancel = mbox.addButton(tr("Cancel"), QMessageBox.RejectRole)
            mbox.setDefaultButton(btn_replace)
            mbox.exec_()
            clicked = mbox.clickedButton()

            if clicked is btn_cancel:
                # Files on disk are kept; nothing added to project.
                self._consistency_lbl.setText(
                    tr("Status: harmonized files written to {path}; project not changed.").format(
                        path=chosen
                    )
                )
                self._consistency_lbl.setStyleSheet("color: #2E7D32;")
                return

            replace_originals = clicked is btn_replace
            self._replace_rasters_with_harmonized(new_paths, remove_originals=replace_originals)
            # Auto-refresh the consistency label
            self._on_check_consistency()

        worker.progress.connect(_on_progress)
        worker.finished.connect(_on_finished)
        # Wire the modal dialog's Cancel button to the worker's cancel
        # flag. The previous version connected to requestInterruption(),
        # which sets a Qt flag the worker never polled — clicking
        # Cancel did nothing. worker.cancel() flips a Python flag that
        # harmonize_rasters checks at the start of every raster.
        progress.canceled.connect(worker.cancel)
        worker.start()

    def _replace_rasters_with_harmonized(self, new_paths: list, remove_originals: bool = True):
        """Swap the raster list so subsequent runs use the aligned copies.

        Each harmonized output is loaded into the QGIS project as a
        new raster layer (so the user can inspect it on the map) and
        replaces the corresponding row in the raster list while
        preserving each row's continuous/categorical setting.

        When ``remove_originals`` is True (the v0.1.7 default chosen
        by the "Replace" dialog button), the original unaligned
        raster layers are also removed from the QGIS project panel
        — keeping the Layers panel uncluttered and the ① Data list
        in sync with what QGIS shows. The files themselves stay on
        disk; only the project references are removed. When False
        (the "Add as additional layers" button), originals stay
        visible in the Layers panel and the user gets two parallel
        sets of layers in the project.
        """
        # Read existing categorical settings AND layer IDs before we
        # overwrite the list. The IDs are what we need to remove the
        # originals from the QGIS project.
        existing_cats = []
        original_layer_ids = []
        for i in range(self._raster_list.count()):
            item = self._raster_list.item(i)
            existing_cats.append(bool(item.data(self._ROLE_IS_CATEG)))
            original_layer_ids.append(item.data(self._ROLE_LAYER_ID))

        self._raster_list.clear()
        for i, p in enumerate(new_paths):
            name = os.path.splitext(os.path.basename(p))[0]
            lyr = QgsRasterLayer(p, name)
            if not lyr.isValid():
                continue
            QgsProject.instance().addMapLayer(lyr)
            is_cat = existing_cats[i] if i < len(existing_cats) else False
            self._add_raster_item(
                layer_name=name,
                layer_id=lyr.id(),
                is_categorical=is_cat,
            )

        # v0.1.7: remove the unaligned originals from the QGIS
        # project AFTER adding the harmonized ones, so the Layers
        # panel reflects the new "aligned-only" state. Done in a
        # try/except per layer in case one was already removed
        # manually by the user.
        if remove_originals:
            for lid in original_layer_ids:
                if not lid:
                    continue
                try:
                    QgsProject.instance().removeMapLayer(lid)
                except Exception:
                    pass

    def _browse_save(self, le: QLineEdit, ffilter: str):
        # Open the file dialog at the most useful starting location:
        # the directory of whatever absolute path is currently in the
        # field (so successive saves stay together), or — if the field
        # is empty / relative — the unified default output folder.
        # The previous version always passed "" which dropped users
        # into whatever folder QGIS happened to remember last, often
        # somewhere unrelated to the QMaxent run.
        current = le.text().strip()
        start_dir = ""
        if current and os.path.isabs(current):
            cand = os.path.dirname(current)
            start_dir = cand if os.path.isdir(cand) else _default_output_dir()
        else:
            start_dir = _default_output_dir()
        path, _ = QFileDialog.getSaveFileName(self, tr("Select save location"), start_dir, ffilter)
        if path:
            le.setText(path)

    # =========================================================================
    # Run
    # =========================================================================

    def _run_maxent(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._run_btn.setText(tr("▶  Run Maxent"))
            self._set_status(tr("Stopping..."))
            return

        try:
            presence_layer = self._get_presence_layer()
        except ValueError as e:
            self._set_status(f"⚠ {e}")
            return

        raster_layers = self._get_raster_layers()
        if not raster_layers:
            self._set_status(f"⚠ {tr('Add environmental raster layers.')}")
            return

        try:
            from ..core.venv_manager import ensure_venv_packages_available

            ensure_venv_packages_available()
            from ..bridge.raster_bridge import layers_to_paths
            from ..bridge.vector_bridge import presence_layer_to_geodataframe

            presence_gdf = presence_layer_to_geodataframe(presence_layer)
            raster_paths = layers_to_paths(raster_layers)
            feature_names = [l.name() for l in raster_layers]
        except Exception as e:
            self._set_status(tr("Data error: {e}").format(e=e))
            return

        ft_auto = self._ft_auto_radio.isChecked()
        ft_manual = (
            [code for code, cb in self._ft_checks.items() if cb.isChecked()] if not ft_auto else []
        )

        config = {
            "presence_gdf": presence_gdf,
            "raster_paths": raster_paths,
            "feature_names": feature_names,
            "categorical_indices": self._get_categorical_indices(),
            "n_background": self._bg_spin.value(),
            "feature_types_auto": ft_auto,
            "feature_types": ft_manual,
            "beta_mult": self._beta_spin.value(),
            "transform": self._proj_transform.currentText(),
            "n_hinge": self._hinge_spin.value(),
            "n_threshold": self._thresh_spin.value(),
            "add_to_bg": self._addtobg_chk.isChecked(),
            "distance_weights": self._distweights_chk.isChecked(),
            "cv_method": self._cv_combo.currentIndex(),
            "n_folds": self._folds_spin.value(),
            "grid_size": self._grid_spin.value(),  # ⑤ was missing
            "buffer_dist": self._buffer_spin.value(),
            "random_seed": (self._seed_spin.value() if self._seed_check.isChecked() else None),
            "do_jackknife": self._jackknife_chk.isChecked(),
            "do_permutation": self._permutation_chk.isChecked(),
            "permutation_repeats": self._permutation_repeats_spin.value(),
            "output_model": self._model_path.text().strip() or None,
            "output_xlsx": self._xlsx_path.text().strip() or None,
        }

        from ..workers.maxent_worker import MaxentWorker

        self._train_log.clear()
        self._train_progress.setValue(0)
        self._run_btn.setText(tr("■  Stop"))
        self.tabs.setCurrentIndex(2)

        self._worker = MaxentWorker(config, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._log_append)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _get_presence_layer(self):
        try:
            layer = self._pres_combo.currentLayer()
        except AttributeError:
            layer = None
        if layer is None:
            raise ValueError(tr("Select a presence point layer."))
        return layer

    def _get_raster_layers(self) -> list:
        return [
            QgsProject.instance().mapLayer(self._raster_list.item(i).data(self._ROLE_LAYER_ID))
            for i in range(self._raster_list.count())
            if QgsProject.instance().mapLayer(self._raster_list.item(i).data(self._ROLE_LAYER_ID))
        ]

    def _get_categorical_indices(self) -> list:
        """Return 0-based indices of rasters flagged as categorical.

        The order matches _get_raster_layers — i.e. it tracks the index
        of each raster in the user's training-time covariate stack, the
        format elapid expects for ``MaxentModel.fit(categorical=...)``.
        """
        indices = []
        out_idx = 0
        for i in range(self._raster_list.count()):
            item = self._raster_list.item(i)
            lyr_id = item.data(self._ROLE_LAYER_ID)
            # Only count rows whose layer is still in the project, to keep
            # the indexing consistent with _get_raster_layers.
            if QgsProject.instance().mapLayer(lyr_id) is None:
                continue
            if bool(item.data(self._ROLE_IS_CATEG)):
                indices.append(out_idx)
            out_idx += 1
        return indices

    # =========================================================================
    # Worker callbacks
    # =========================================================================

    def _on_progress(self, pct: int, msg: str):
        self._train_progress.setValue(pct)
        self._train_status.setText(msg)
        self._set_status(msg)

    def _log_append(self, text: str):
        self._train_log.appendPlainText(text)

    # ── Training log — manual export helpers ───────────────────────
    # The auto-save next to prediction.tif covers the common case, but
    # these two buttons give the user direct control: copy to clipboard
    # for paste-into-Slack-or-email, or "Save log as…" for an
    # arbitrary file location (e.g. a Zenodo deposit folder, an
    # iCloud / Dropbox sync path, or a co-author's review folder).
    # Both operate on the QPlainTextEdit's current content, so they
    # always reflect exactly what the user sees in the Training tab.

    def _copy_train_log_to_clipboard(self):
        from qgis.PyQt.QtWidgets import QApplication

        text = self._train_log.toPlainText()
        if not text:
            self._set_status(tr("Log is empty — nothing to copy."))
            return
        QApplication.clipboard().setText(text)
        self._set_status(
            tr("Training log copied to clipboard ({n} characters).").format(n=len(text))
        )

    def _save_train_log_as(self):
        from qgis.PyQt.QtWidgets import QFileDialog

        text = self._train_log.toPlainText()
        if not text:
            self._set_status(tr("Log is empty — nothing to save."))
            return
        # Default suggestion: same directory as the prediction raster
        # (or, failing that, the default output directory) with a
        # timestamped filename so successive saves don't overwrite.
        default_dir = _default_output_dir()
        try:
            proj_path = (self._proj_path.text() or "").strip()
            if proj_path:
                cand = os.path.dirname(proj_path) or default_dir
                if os.path.isdir(cand):
                    default_dir = cand
        except Exception:
            pass
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = os.path.join(default_dir, f"training_log_{stamp}.txt")
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            tr("Save training log"),
            default_name,
            tr("Text files (*.txt);;All files (*)"),
        )
        if not out_path:
            return
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            self._set_status(tr("Training log saved: {path}").format(path=out_path))
        except OSError as e:
            self._set_status(tr("Save failed: {e}").format(e=e))

    def _autosave_training_log(self, target_path: str) -> None:
        """Auto-save training_log.txt next to model.pkl.

        Triggered from ``_on_finished`` at the end of ▶ Run Maxent
        (v0.1.7) — the natural moment a researcher would want the
        log. Previously gated on a Parameters-tab checkbox and only
        written next to the prediction raster; that checkbox is
        removed in v0.1.7 since the cost is negligible (a few KB) and
        the value to reviewers / collaborators retracing a run is
        substantial. The log content is whatever is currently in the
        ③ Training tab — same source as the manual Copy / Save-as
        buttons, ensuring consistency. Silent on success; errors are
        logged to the training tab so the user can see them but the
        Run Maxent still succeeds.
        """
        try:
            text = self._train_log.toPlainText()
            if not text:
                return
            out_dir = os.path.dirname(target_path) or _default_output_dir()
            # Auto-create the directory so a user-typed fresh path
            # works on the first run (matches the same guarantee made
            # by the worker for model.pkl / results.xlsx).
            os.makedirs(out_dir, exist_ok=True)
            log_path = os.path.join(out_dir, "training_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(text)
            self._log_append(tr("Training log saved: {path}").format(path=log_path))
        except Exception as e:
            # Never let a log-save failure break the run — the model
            # is the primary deliverable.
            self._log_append(tr("(Could not save training log: {e})").format(e=e))

    def _on_finished(self, success: bool, msg: str, results: dict):
        self._run_btn.setText(tr("▶  Run Maxent"))
        self._worker = None

        if not success:
            self._set_status(f"✗ {msg[:120]}")
            self._log_append(f"\n[Error]\n{msg}")
            return

        self._results = results
        self._model = results.get("model")
        self._meta = results.get("meta", {})

        self._populate_response_combo()
        self._plot_importance()
        self._plot_permutation()
        self._proj_btn.setEnabled(self._model is not None)
        # Priority Sites for Survey is available once a model exists —
        # the actual run-time check (prediction raster present) happens
        # in _run_priority_extraction.
        self._priority_btn.setEnabled(self._model is not None)
        # Refresh the threshold display from the new model's meta so
        # the user sees the T10 / MTP / MaxSSS values without having
        # to re-pick the method. Also recompute the Discovery
        # minimum-suitability floor from the new prediction raster
        # (raster_max × 0.9), if any has been generated.
        try:
            self._on_priority_threshold_method_changed(self._priority_thr_method.currentIndex())
            self._refresh_discovery_floor()
        except Exception:
            pass

        n_p = self._meta.get("n_presence", "?")
        n_b = self._meta.get("n_background", "?")
        fauc = results.get("full_auc")
        cv_aucs = results.get("cv_aucs", [])
        if isinstance(n_b, int):
            parts = [
                tr("presence={n}").format(n=n_p) + "  " + tr("background={n}").format(n=f"{n_b:,}")
            ]
        else:
            parts = [tr("presence={n}").format(n=n_p)]
        if fauc is not None:
            parts.append(tr("train AUC={v:.4f}").format(v=fauc))
        if cv_aucs:
            parts.append(tr("CV AUC={v:.4f}").format(v=float(np.mean(cv_aucs))))
        self._set_status("✓ " + "  |  ".join(parts))

        self.tabs.setCurrentIndex(3)
        self._log_append(tr("✓ All analysis complete."))

        # v0.1.7: write training_log.txt next to model.pkl so the log
        # follows the artifact a researcher carries between sessions.
        # Done after _log_append so the "All analysis complete." line
        # is included in the saved file. The model path comes from
        # the LineEdit on ② Parameters → Output Files.
        try:
            model_path = self._model_path.text().strip()
            if model_path:
                self._autosave_training_log(model_path)
        except Exception:
            # The log itself is best-effort — never let it break the
            # successful Run Maxent flow.
            pass

    # =========================================================================
    # Response curves — English matplotlib labels (② fix)
    # =========================================================================

    def _populate_response_combo(self):
        names = self._meta.get("feature_names", [])
        self._response_var_combo.blockSignals(True)
        self._response_var_combo.clear()
        self._response_var_combo.addItems(names)
        self._response_var_combo.blockSignals(False)
        if names:
            self._plot_response(0)

    def _on_transform_changed(self, _idx: int):
        """Redraw the active response curve when output transform changes."""
        if self._model is None or not self._meta:
            return
        cur = self._response_var_combo.currentIndex()
        if cur >= 0:
            self._plot_response(cur)

    def _plot_response(self, idx: int):
        if self._model is None or not self._meta:
            return
        names = self._meta.get("feature_names", [])
        if not names or idx < 0 or idx >= len(names):
            return

        self._clear_widget(self._response_canvas_widget)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

            var = names[idx]
            vmin = self._meta["varmin"].get(var, 0.0)
            vmax = self._meta["varmax"].get(var, 1.0)
            sm = self._meta.get("samplemeans", {})
            # Single source of truth: the Results-tab Output transform
            # combo. When it differs from the value baked into the model
            # at training time, update the model so predict() returns the
            # requested transform's outputs.
            trans = self._proj_transform.currentText()
            if getattr(self._model, "transform", None) != trans:
                self._model.transform = trans

            margin = 0.1 * (vmax - vmin) if vmax > vmin else 0.1

            # Categorical vs continuous response curve.
            #
            # For continuous variables we sweep a smooth 100-point grid
            # along the variable's training range and predict at each
            # point — this is the standard SDM response curve.
            #
            # For categorical variables a smooth sweep is meaningless
            # because the codes are not ordered (e.g. land cover code 7
            # is not "between" 6 and 8 in any ecological sense). We
            # instead show a bar chart with one bar per observed category
            # value, matching the convention used by dismo, SDMtune and
            # maxent.jar (Phillips et al. 2017).
            cat_indices = self._meta.get("categorical_indices", []) or []
            is_categorical = idx in set(cat_indices)

            fig, ax = plt.subplots(figsize=(4.5, 3))
            if is_categorical:
                # Use the actual category codes seen during training,
                # not a contiguous integer range. Categorical rasters
                # (biome, land cover, soil type) typically have sparse
                # codes (e.g. {1, 2, 4, 7, 8, 12, 13} — 3, 5, 6, etc.
                # are absent). Sweeping a contiguous range hands
                # sklearn's OneHotEncoder unknown codes during
                # predict(), which it refuses with "Found unknown
                # categories ... during transform".
                levels_map = self._meta.get("categorical_levels", {}) or {}
                cats = levels_map.get(var)
                if not cats:
                    # Fallback for older .pkl files that don't carry
                    # the levels map. Newer models always populate
                    # this in MaxentWorker._execute, so this path is
                    # only hit when loading legacy saved models.
                    lo, hi = int(round(vmin)), int(round(vmax))
                    cats = list(range(lo, hi + 1))
                X = np.tile([float(sm.get(n, 0.0)) for n in names], (len(cats), 1)).astype(
                    np.float32
                )
                for k, c in enumerate(cats):
                    X[k, idx] = float(c)
                preds = self._model.predict(X)
                # Use string-labelled ticks so the codes display as
                # discrete classes; a numeric x-axis would visually
                # imply that code 13 is "6× larger than" code 2,
                # which is meaningless for nominal categories.
                positions = list(range(len(cats)))
                # Match the colour of the continuous response line
                # exactly: solid #1D9E75 fill with no outline. Same
                # hue as the line plot so a user toggling between a
                # continuous and a categorical variable sees the
                # same visual identity for "this is a response".
                ax.bar(positions, preds, color="#1D9E75", linewidth=0)
                ax.set_xticks(positions)
                ax.set_xticklabels([str(c) for c in cats])
                ax.set_xlabel(f"{var} (category code)", fontsize=9)
            else:
                # Continuous: smooth sweep across training range.
                xs = np.linspace(vmin - margin, vmax + margin, 100)
                X = np.tile([float(sm.get(n, 0.0)) for n in names], (100, 1)).astype(np.float32)
                X[:, idx] = xs.astype(np.float32)
                preds = self._model.predict(X)
                ax.plot(xs, preds, color="#1D9E75", linewidth=2)
                ax.axvspan(vmin, vmax, alpha=0.08, color="#1D9E75", label="Training range")
                ax.axvline(
                    float(sm.get(var, 0)), color="#666", linewidth=1, linestyle=":", label="Mean"
                )
                ax.set_xlabel(var, fontsize=9)
                ax.legend(fontsize=7)

            # Y-axis label — academic SDM convention is to spell out
            # what the axis represents (predicted habitat suitability)
            # alongside the transform that produced the values. For raw
            # output we drop the "suitability" framing because raw is
            # an unbounded relative score, not a probability-like
            # quantity (Phillips et al. 2017).
            if trans == "cloglog":
                y_label = "Predicted suitability (cloglog)"
            elif trans == "logistic":
                y_label = "Predicted suitability (logistic)"
            else:
                y_label = "Raw output"
            ax.set_ylabel(y_label, fontsize=9)
            ax.set_title(f"Response curve: {var}", fontsize=10)
            if trans in ("cloglog", "logistic"):
                ax.set_ylim(0, 1)
            ax.tick_params(labelsize=8)
            fig.tight_layout()

            canvas = FigureCanvasQTAgg(fig)
            self._response_canvas_widget.layout().addWidget(canvas)
            plt.close(fig)

        except Exception as e:
            lbl = QLabel(tr("Response curve error: {e}").format(e=e))
            lbl.setWordWrap(True)
            self._response_canvas_widget.layout().addWidget(lbl)

    def _make_response_figure(self):
        """Create a grid figure of all response curves (for PNG export)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = self._meta.get("feature_names", [])
        if not names or self._model is None:
            return None

        sm = self._meta.get("samplemeans", {})
        # Single source of truth: Results-tab Output transform combo.
        trans = self._proj_transform.currentText()
        if getattr(self._model, "transform", None) != trans:
            self._model.transform = trans
        n = len(names)
        ncols = min(3, n)
        nrows = -(-n // ncols)  # ceiling division

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes = np.array(axes).flatten()

        cat_indices = set(self._meta.get("categorical_indices", []) or [])

        for i, var in enumerate(names):
            vmin = self._meta["varmin"].get(var, 0.0)
            vmax = self._meta["varmax"].get(var, 1.0)
            margin = 0.1 * (vmax - vmin) if vmax > vmin else 0.1
            try:
                ax = axes[i]
                if i in cat_indices:
                    # Categorical: bar chart with only the codes seen
                    # at training time (see _plot_response for the
                    # detailed reason — sklearn rejects unknown
                    # codes during predict()).
                    levels_map = self._meta.get("categorical_levels", {}) or {}
                    cats = levels_map.get(var)
                    if not cats:
                        lo, hi = int(round(vmin)), int(round(vmax))
                        cats = list(range(lo, hi + 1))
                    X = np.tile([float(sm.get(n2, 0.0)) for n2 in names], (len(cats), 1)).astype(
                        np.float32
                    )
                    for k, c in enumerate(cats):
                        X[k, i] = float(c)
                    preds = self._model.predict(X)
                    positions = list(range(len(cats)))
                    # Same as _plot_response — solid #1D9E75 fill,
                    # no outline, identical to the continuous line.
                    ax.bar(positions, preds, color="#1D9E75", linewidth=0)
                    ax.set_xticks(positions)
                    ax.set_xticklabels([str(c) for c in cats])
                    ax.set_xlabel(f"{var} (category code)", fontsize=9)
                else:
                    xs = np.linspace(vmin - margin, vmax + margin, 100)
                    X = np.tile([float(sm.get(n2, 0.0)) for n2 in names], (100, 1)).astype(
                        np.float32
                    )
                    X[:, i] = xs.astype(np.float32)
                    preds = self._model.predict(X)
                    ax.plot(xs, preds, color="#1D9E75", linewidth=2)
                    ax.axvspan(vmin, vmax, alpha=0.08, color="#1D9E75")
                    ax.axvline(float(sm.get(var, 0)), color="#666", linewidth=1, linestyle=":")
                    ax.set_xlabel(var, fontsize=9)
                # Same Y-axis label convention as the on-screen single
                # plot above — see comment there.
                if trans == "cloglog":
                    y_label = "Predicted suitability (cloglog)"
                elif trans == "logistic":
                    y_label = "Predicted suitability (logistic)"
                else:
                    y_label = "Raw output"
                ax.set_ylabel(y_label, fontsize=9)
                ax.set_title(f"Response: {var}", fontsize=10)
                if trans in ("cloglog", "logistic"):
                    ax.set_ylim(0, 1)
                ax.tick_params(labelsize=8)
            except Exception:
                pass

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        return fig

    # =========================================================================
    # Variable importance — ROC curve + Jackknife (③ fix)
    # =========================================================================

    def _plot_importance(self):
        self._clear_widget(self._importance_canvas_widget)
        if self._results is None:
            return

        jk = self._results.get("jackknife_results", [])
        cv_aucs = self._results.get("cv_aucs", [])
        roc_fpr = self._results.get("roc_fpr", [])
        roc_tpr = self._results.get("roc_tpr", [])
        full_auc = self._results.get("full_auc")

        if not jk and not cv_aucs and not roc_fpr:
            lbl = QLabel(tr("No jackknife or CV results.\nEnable them in Parameters and re-run."))
            lbl.setWordWrap(True)
            self._importance_canvas_widget.layout().addWidget(lbl)
            return

        try:
            fig = self._make_importance_figure(
                jk,
                cv_aucs,
                roc_fpr,
                roc_tpr,
                full_auc,
                full_test_auc=self._results.get("jk_full_test_auc"),
                cv_roc_fpr_list=self._results.get("cv_roc_fpr_list", []),
                cv_roc_tpr_list=self._results.get("cv_roc_tpr_list", []),
            )
            import matplotlib.backends.backend_qt5agg as _mplqt

            canvas = _mplqt.FigureCanvasQTAgg(fig)
            self._importance_canvas_widget.layout().addWidget(canvas)
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception as e:
            lbl = QLabel(tr("Chart error: {e}").format(e=e))
            lbl.setWordWrap(True)
            self._importance_canvas_widget.layout().addWidget(lbl)

    def _plot_permutation(self):
        """Render the permutation importance bar chart (added v0.1.3).

        Mirrors the pattern of _plot_importance: clears the existing
        canvas, builds a fresh matplotlib figure, embeds it in the
        Qt widget, and enables the Save PNG / Save CSV buttons.
        """
        self._clear_widget(self._permutation_canvas_widget)
        if self._results is None:
            return

        pi_rows = self._results.get("permutation_results", []) or []
        if not pi_rows:
            lbl = QLabel(
                tr("No permutation importance results.\nEnable it in Parameters and re-run.")
            )
            lbl.setWordWrap(True)
            self._permutation_canvas_widget.layout().addWidget(lbl)
            self._permutation_save_png_btn.setEnabled(False)
            self._permutation_save_csv_btn.setEnabled(False)
            return

        try:
            fig = self._make_permutation_figure(
                pi_rows,
                source=self._results.get("permutation_source", "training set"),
                n_repeats=self._results.get("permutation_n_repeats", 10),
            )
            import matplotlib.backends.backend_qt5agg as _mplqt

            canvas = _mplqt.FigureCanvasQTAgg(fig)
            self._permutation_canvas_widget.layout().addWidget(canvas)
            self._permutation_fig_cached = fig
            self._permutation_save_png_btn.setEnabled(True)
            self._permutation_save_csv_btn.setEnabled(True)
            # Note: we do NOT close the figure here because the canvas
            # holds a reference to it for redrawing. The same pattern
            # applies in _plot_importance.
        except Exception as e:
            lbl = QLabel(tr("Chart error: {e}").format(e=e))
            lbl.setWordWrap(True)
            self._permutation_canvas_widget.layout().addWidget(lbl)
            self._permutation_save_png_btn.setEnabled(False)
            self._permutation_save_csv_btn.setEnabled(False)

    def _make_permutation_figure(self, pi_rows, source: str, n_repeats):
        """Build the matplotlib Figure for the permutation tab.

        Horizontal bar chart with ±std error bars; variables sorted
        by mean importance descending (most important on top). The
        x-axis is the normalized percentage of total (matches
        maxent.jar's "permutation importance" column convention).
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # pi_rows is already sorted by descending importance_mean
        # (see maxent_worker._execute step 7). We reverse for the
        # bar chart so the top variable appears at the top, matching
        # the academic-paper convention.
        rows = list(reversed(pi_rows))
        names = [r["variable"] for r in rows]
        pcts = [r.get("importance_pct", 0.0) for r in rows]

        # Convert std (which is in raw AUC-drop units) to the same
        # percentage scale as importance_pct so the error bars are
        # interpretable on the chart's x-axis. Total stays the same
        # across rows so we can factor it out.
        total_raw = sum(r.get("importance_mean", 0.0) for r in rows)
        if total_raw > 0:
            stds_pct = [100.0 * r.get("importance_std", 0.0) / total_raw for r in rows]
        else:
            stds_pct = [0.0] * len(rows)

        n = len(rows)
        fig_h = max(3.0, 0.45 * n + 1.5)
        fig, ax = plt.subplots(figsize=(8, fig_h), tight_layout=True)

        y_pos = list(range(n))
        # Plugin palette: primary green fill + darker green outline,
        # matching the Jackknife "with only variable" bars in
        # _make_jackknife_only_figure for visual consistency across
        # the Results sub-tabs.
        ax.barh(
            y_pos,
            pcts,
            xerr=stds_pct,
            color="#1D9E75",
            edgecolor="#0F4D3A",
            linewidth=0.5,
            error_kw={"capsize": 3, "alpha": 0.5, "ecolor": "#666"},
        )

        # Per-bar value labels — same convention as the Jackknife
        # importance bars: white text inside a long bar, dark text
        # just past the end of a short bar. The cut-off uses 10% of
        # the longest bar so very short bars (like bio6 at ~4%)
        # render readable text outside the bar instead of overflowing
        # past the bar edge.
        max_pct = max(pcts) if pcts else 1.0
        for i, p in enumerate(pcts):
            if p > max_pct * 0.10:
                # Inside the bar, white text near the left edge so
                # the error bar's right whisker doesn't visually
                # collide with the label.
                ax.text(p * 0.02, i, f"{p:.1f}", va="center", ha="left", fontsize=9, color="white")
            else:
                # Outside the bar, dark text just past the error
                # bar's right whisker.
                err = stds_pct[i] if i < len(stds_pct) else 0.0
                ax.text(
                    p + err + max_pct * 0.015,
                    i,
                    f"{p:.1f}",
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="#222",
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("Permutation importance (% of total)")
        ax.set_title(f"Permutation importance — {source}, n_repeats={n_repeats}")
        ax.grid(axis="x", alpha=0.3)
        # A subtle note for users comparing against maxent.jar's
        # exact percentage values: the absolute % depends on the
        # AUC drop sum, which differs slightly between QMaxent and
        # maxent.jar (different solver). The ranking is what matters
        # for cross-implementation comparison.
        return fig

    def _on_save_permutation_png(self):
        """Save the permutation importance chart as PNG."""
        if not getattr(self, "_permutation_fig_cached", None):
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("Save permutation importance chart"),
            os.path.join(_default_output_dir(), "permutation_importance.png"),
            "PNG (*.png)",
        )
        if path:
            try:
                self._permutation_fig_cached.savefig(path, dpi=150)
                self._set_status(tr("Saved: {p}").format(p=path))
            except Exception as e:
                QMessageBox.critical(self, "QMaxent", tr("Failed to save PNG: {e}").format(e=e))

    def _on_save_permutation_csv(self):
        """Save the permutation importance data as CSV."""
        if not self._results:
            return
        pi_rows = self._results.get("permutation_results", []) or []
        if not pi_rows:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("Save permutation importance data"),
            os.path.join(_default_output_dir(), "permutation_importance.csv"),
            "CSV (*.csv)",
        )
        if not path:
            return
        try:
            import csv

            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Variable", "Mean importance", "Std", "Normalized (%)"])
                for r in pi_rows:
                    w.writerow(
                        [
                            r.get("variable", ""),
                            f"{r.get('importance_mean', 0.0):.6f}",
                            f"{r.get('importance_std', 0.0):.6f}",
                            f"{r.get('importance_pct', 0.0):.2f}",
                        ]
                    )
            self._set_status(tr("Saved: {p}").format(p=path))
        except Exception as e:
            QMessageBox.critical(self, "QMaxent", tr("Failed to save CSV: {e}").format(e=e))

    # ── SWD export handlers (added v0.1.3) ──────────────────────────────

    def _browse_swd_dir(self):
        """Pick the output folder for SWD export."""
        d = QFileDialog.getExistingDirectory(
            self,
            tr("Select output folder for SWD export"),
            self._swd_dir_edit.text() or _default_output_dir(),
        )
        if d:
            self._swd_dir_edit.setText(d)

    def _on_swd_export_clicked(self):
        """Validate inputs and launch the SWD/raster-samples export worker.

        Reuses the same presence-layer + raster-list + categorical-flag
        machinery as the Run Maxent button so the export reflects
        exactly what would go into a training run. This way the
        downstream maxent.jar invocation operates on the same data
        QMaxent would have used internally.

        In v0.1.4 the format is chosen via two radio buttons. The
        worker dispatches on cfg["format"] — "swd" or "raster_samples".
        """
        # Validate presence layer.
        try:
            presence_layer = self._get_presence_layer()
        except ValueError as e:
            QMessageBox.warning(self, "QMaxent", str(e))
            return

        # Validate raster list — at least one continuous variable.
        raster_layers = self._get_raster_layers()
        if not raster_layers:
            QMessageBox.warning(self, "QMaxent", tr("Add at least one environmental raster."))
            return

        # Validate output folder.
        output_dir = self._swd_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "QMaxent", tr("Specify an output folder."))
            return

        # Build worker config — same data dependencies as a real run.
        try:
            from ..core.venv_manager import ensure_venv_packages_available

            ensure_venv_packages_available()
            from ..bridge.raster_bridge import layers_to_paths
            from ..bridge.vector_bridge import presence_layer_to_geodataframe

            presence_gdf = presence_layer_to_geodataframe(presence_layer)
            raster_paths = layers_to_paths(raster_layers)
            feature_names = [l.name() for l in raster_layers]
        except Exception as e:
            QMessageBox.critical(self, "QMaxent", tr("Data error: {e}").format(e=e))
            return

        # Resolve format choice from the radio buttons. The "raster +
        # samples" path needs the .asc converter, which runs once per
        # raster, so it takes proportionally longer than SWD on big
        # stacks; the progress label communicates this.
        if self._swd_fmt_swd.isChecked():
            fmt = "swd"
            progress_label = tr("Exporting SWD files...")
        else:
            fmt = "raster_samples"
            progress_label = tr("Exporting samples + raster...")

        cfg = {
            "format": fmt,
            "presence_gdf": presence_gdf,
            "raster_paths": raster_paths,
            "feature_names": feature_names,
            "categorical_indices": self._get_categorical_indices(),
            "n_background": self._bg_spin.value(),
            "output_dir": output_dir,
            "species_name": presence_layer.name(),
            "bias_path": None,
        }

        # Launch worker with a modal progress dialog. The QProgressDialog
        # gives the user a Cancel button and prevents accidental
        # interaction with the main dock while data extraction runs.
        from ..workers.swd_export_worker import SWDExportWorker

        progress = QProgressDialog(progress_label, tr("Cancel"), 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        worker = SWDExportWorker(cfg, self)
        # Keep references so they survive past the local scope of this
        # method — Qt requires both the worker and the dialog to stay
        # alive until finished() fires.
        self._swd_worker = worker
        self._swd_progress = progress

        def _on_progress(pct, msg):
            progress.setValue(int(pct))
            progress.setLabelText(msg)

        def _on_finished(ok, msg, result):
            progress.close()
            self._swd_worker = None
            self._swd_progress = None
            if ok:
                # Format-specific completion message — both modes have
                # different output structure, so we tell the user
                # exactly what was produced.
                rfmt = result.get("format", "swd")
                if rfmt == "swd":
                    body = tr(
                        "SWD export complete.\n\n"
                        "Presence:    {p} ({np} rows)\n"
                        "Background:  {b} ({nb} rows)\n"
                        "README:      {r}\n\n"
                        "The README contains the maxent.jar command "
                        "line to run on this dataset."
                    ).format(
                        p=result.get("presence_path", ""),
                        np=result.get("n_presence", 0),
                        b=result.get("background_path", ""),
                        nb=result.get("n_background", 0),
                        r=result.get("readme_path", ""),
                    )
                else:
                    body = tr(
                        "Samples + Raster export complete.\n\n"
                        "Samples:   {s} ({np} rows)\n"
                        "Layers:    {l} ({nr} ASCII grids)\n"
                        "README:    {r}\n\n"
                        "The README contains the maxent.jar command "
                        "line to run on this dataset."
                    ).format(
                        s=result.get("samples_path", ""),
                        np=result.get("n_presence", 0),
                        l=result.get("layers_dir", ""),
                        nr=result.get("n_rasters", 0),
                        r=result.get("readme_path", ""),
                    )
                QMessageBox.information(self, "QMaxent", body)
            else:
                QMessageBox.critical(self, "QMaxent", tr("Export failed:\n{msg}").format(msg=msg))

        worker.progress.connect(_on_progress)
        worker.finished.connect(_on_finished)
        progress.canceled.connect(worker.cancel)
        worker.start()

    def _make_importance_figure(
        self,
        jk,
        cv_aucs,
        roc_fpr,
        roc_tpr,
        full_auc,
        full_test_auc=None,
        cv_roc_fpr_list=None,
        cv_roc_tpr_list=None,
    ):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_vars = len(jk)
        # Combined figure size: keep the dock readable regardless of how
        # many variables are present.
        #   • Width 11 with two columns gives each panel ~5.5 inches
        #     wide — enough for the ROC panel to land near a 1:1 aspect
        #     ratio (which is the academic norm) and enough for the
        #     Jackknife labels to fit even when variable names are
        #     long.
        #   • Height grows with n_vars so each Jackknife row stays
        #     comfortably tall, but with a higher floor (4.5) than
        #     before so the ROC panel doesn't get squeezed when
        #     n_vars is small.
        fig_h = max(4.5, 1.6 + n_vars * 0.35)
        has_roc = len(roc_fpr) > 0
        n_cols = 2 if (jk or has_roc) else 1
        fig_w = 11 if n_cols == 2 else 5.5
        fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
        if n_cols == 1:
            axes = [axes]

        # ── Left: ROC curve ──────────────────────────────────────────────
        # Academic-standard ROC panel:
        #   • Each CV fold's ROC drawn faintly in the background, so
        #     the spread of the K curves is immediately visible — a
        #     proxy for model stability that a single mean number
        #     can't convey (Sofaer et al. 2019, Methods Ecol. Evol.).
        #   • Mean CV ROC drawn on top in a darker shade with a ±1 SD
        #     band, summarising the K folds.
        #   • Training ROC drawn in the darkest shade, since it's the
        #     headline "in-sample fit" reference.
        #   • Random-classifier diagonal (dashed) for orientation.
        ax0 = axes[0]
        cv_fpr_list = list(cv_roc_fpr_list or [])
        cv_tpr_list = list(cv_roc_tpr_list or [])
        if has_roc:
            if cv_fpr_list and cv_tpr_list:
                fold_label_used = False
                for f_fpr, f_tpr in zip(cv_fpr_list, cv_tpr_list):
                    if not f_fpr or not f_tpr:
                        continue
                    ax0.plot(
                        f_fpr,
                        f_tpr,
                        color="#9FE1CB",
                        linewidth=0.8,
                        alpha=0.55,
                        label=(
                            "CV folds (n=" + str(len(cv_fpr_list)) + ")"
                            if not fold_label_used
                            else None
                        ),
                    )
                    fold_label_used = True
                try:
                    mean_fpr = np.linspace(0, 1, 101)
                    tprs = []
                    for f_fpr, f_tpr in zip(cv_fpr_list, cv_tpr_list):
                        if not f_fpr or not f_tpr:
                            continue
                        interp_tpr = np.interp(mean_fpr, f_fpr, f_tpr)
                        interp_tpr[0] = 0.0
                        tprs.append(interp_tpr)
                    if tprs:
                        tprs_arr = np.vstack(tprs)
                        mean_tpr = tprs_arr.mean(axis=0)
                        std_tpr = tprs_arr.std(axis=0)
                        upper = np.minimum(mean_tpr + std_tpr, 1.0)
                        lower = np.maximum(mean_tpr - std_tpr, 0.0)

                        cv_label = "Mean CV ROC"
                        if cv_aucs:
                            valid = [a for a in cv_aucs if not np.isnan(a)]
                            if valid:
                                cv_label += f" (mean AUC={np.mean(valid):.3f})"
                        ax0.fill_between(mean_fpr, lower, upper, color="#1D9E75", alpha=0.18)
                        ax0.plot(
                            mean_fpr,
                            mean_tpr,
                            color="#1D9E75",
                            linewidth=1.6,
                            linestyle="--",
                            label=cv_label,
                        )
                except Exception:
                    pass

            train_label = f"Training ROC (AUC={full_auc:.3f})"
            ax0.plot(roc_fpr, roc_tpr, color="#0F4D3A", linewidth=2, label=train_label)

            ax0.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
            ax0.set_xlabel("False Positive Rate", fontsize=9)
            ax0.set_ylabel("True Positive Rate", fontsize=9)
            ax0.set_title("ROC Curve", fontsize=10)
            # Reorder legend by performance descending (Training top,
            # mean CV next, CV folds, random classifier at the bottom)
            # — standard academic ROC plot convention. matplotlib's
            # default legend ordering is plot-order, which would put
            # the high-importance Training curve at index 3.
            handles, labels = ax0.get_legend_handles_labels()
            preferred_order = [
                "Training ROC",
                "Mean CV ROC",
                "CV folds",
                "Random classifier",
            ]

            def _rank(lbl):
                for i, key in enumerate(preferred_order):
                    if lbl.startswith(key):
                        return i
                return len(preferred_order)

            paired = sorted(zip(handles, labels), key=lambda p: _rank(p[1]))
            handles, labels = zip(*paired) if paired else ([], [])
            ax0.legend(handles, labels, fontsize=7, loc="lower right")
            ax0.set_xlim(0, 1)
            ax0.set_ylim(0, 1)
        elif cv_aucs:
            valid = [a for a in cv_aucs if not np.isnan(a)]
            ax0.boxplot(valid, vert=True, patch_artist=True, boxprops=dict(facecolor="#9FE1CB"))
            mean_v = float(np.mean(valid))
            ax0.axhline(
                mean_v, color="#0F6E56", linewidth=1.2, linestyle="--", label=f"Mean={mean_v:.3f}"
            )
            ax0.legend(fontsize=8)
            ax0.set_title(f"CV AUC  (n={len(valid)} folds)", fontsize=10)
            ax0.set_ylim(0, 1)
            ax0.set_ylabel("AUC", fontsize=9)
            ax0.set_xticks([1])
            ax0.set_xticklabels(["CV folds"], fontsize=8)

        # ── Right: Jackknife (maxent.jar single-row overlay layout) ──────
        if jk and n_cols == 2:
            ax1 = axes[1]
            vars_ = [r["variable"] for r in jk]
            only_tr = [r["only_train_auc"] for r in jk]
            only_te = [r["only_test_auc"] for r in jk]
            wo_tr = [r["without_train_auc"] for r in jk]
            wo_te = [r["without_test_auc"] for r in jk]
            drop_tr = [r.get("train_drop_without", float("nan")) for r in jk]
            drop_te = [r.get("test_drop_without", float("nan")) for r in jk]
            has_test = any(not np.isnan(v) for v in only_te + wo_te)

            # Sort variables by importance: largest test-AUC drop on
            # top. matplotlib's barh puts y=0 at the bottom, so sort
            # least-important first (lowest y index).
            def _drop(i):
                d = drop_te[i] if has_test else drop_tr[i]
                return float("-inf") if np.isnan(d) else d

            order = sorted(range(len(vars_)), key=_drop)
            vars_ = [vars_[i] for i in order]
            only_tr = [only_tr[i] for i in order]
            only_te = [only_te[i] for i in order]
            wo_tr = [wo_tr[i] for i in order]
            wo_te = [wo_te[i] for i in order]
            drop_tr = [drop_tr[i] for i in order]
            drop_te = [drop_te[i] for i in order]

            n_vars = len(vars_)
            yp = list(range(n_vars))
            h = 0.35  # two-bar layout per variable

            if has_test:
                only_vals = only_te
                wo_vals = wo_te
                only_lbl = "With only variable"
                wo_lbl = "Without variable"
            else:
                only_vals = only_tr
                wo_vals = wo_tr
                only_lbl = "With only variable (train)"
                wo_lbl = "Without variable (train)"

            # Two-row layout per variable, top-to-bottom:
            #   With only variable  (dark green, same hue as response line)
            #   Without variable    (light green, same hue as response shading)
            ax1.barh(
                [p + 0.5 * h for p in yp],
                only_vals,
                height=h,
                color="#1D9E75",
                linewidth=0,
                label=only_lbl,
            )
            ax1.barh(
                [p - 0.5 * h for p in yp],
                wo_vals,
                height=h,
                color="#C0E6D6",
                linewidth=0,
                label=wo_lbl,
            )

            # Numeric labels on each bar — Reuters-style. Long enough
            # bars get the value inside the bar in white (high
            # contrast); short bars get it just to the right in dark
            # text so the number never collides with its own bar.
            def _label_bar(value, y_pos, dark_bg):
                if np.isnan(value):
                    return
                if value >= 0.08:
                    ax1.text(
                        0.012,
                        y_pos,
                        f"{value:.3f}",
                        va="center",
                        ha="left",
                        fontsize=7,
                        color="white" if dark_bg else "#222222",
                    )
                else:
                    ax1.text(
                        value + 0.008,
                        y_pos,
                        f"{value:.3f}",
                        va="center",
                        ha="left",
                        fontsize=7,
                        color="#222222",
                    )

            for k in range(n_vars):
                _label_bar(only_vals[k], k + 0.5 * h, dark_bg=True)
                _label_bar(wo_vals[k], k - 0.5 * h, dark_bg=False)

            for k in range(n_vars):
                if np.isnan(only_vals[k]):
                    ax1.text(
                        0.02,
                        k + 0.5 * h,
                        "only-* skipped (categorical)",
                        va="center",
                        ha="left",
                        fontsize=7,
                        style="italic",
                        color="#444444",
                    )

            ax1.set_yticks(yp)
            ax1.set_yticklabels(vars_, fontsize=8)
            ax1.set_xlim(0, 1)
            ax1.set_xlabel("AUC", fontsize=9)
            ax1.set_title("Jackknife Variable Importance", fontsize=10)
            ax1.legend(fontsize=7, loc="lower right")

        fig.tight_layout()
        return fig

    def _make_roc_only_figure(
        self, roc_fpr, roc_tpr, cv_aucs, full_auc, cv_roc_fpr_list=None, cv_roc_tpr_list=None
    ):
        """Build a standalone ROC figure for PNG export.

        Same artistic content as the left panel of the dock chart —
        per-fold curves, mean ± SD band, training ROC, random
        diagonal — but rendered at a 1:1 aspect ratio (5×5 inches).
        That ratio is the academic norm and stops the curve from
        looking compressed when it's read on its own.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not (len(roc_fpr) > 0 or cv_aucs):
            return None

        fig, ax0 = plt.subplots(figsize=(5, 5))
        cv_fpr_list = list(cv_roc_fpr_list or [])
        cv_tpr_list = list(cv_roc_tpr_list or [])

        if cv_fpr_list and cv_tpr_list:
            fold_label_used = False
            for f_fpr, f_tpr in zip(cv_fpr_list, cv_tpr_list):
                if not f_fpr or not f_tpr:
                    continue
                kw = {"color": "#9FE1CB", "linewidth": 0.8, "alpha": 0.7}
                if not fold_label_used:
                    kw["label"] = f"CV folds (n={len(cv_fpr_list)})"
                    fold_label_used = True
                ax0.plot(f_fpr, f_tpr, **kw)

            try:
                mean_fpr = np.linspace(0, 1, 100)
                tprs = []
                for f_fpr, f_tpr in zip(cv_fpr_list, cv_tpr_list):
                    if not f_fpr or not f_tpr:
                        continue
                    interp_tpr = np.interp(mean_fpr, f_fpr, f_tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                if tprs:
                    tprs_arr = np.vstack(tprs)
                    mean_tpr = tprs_arr.mean(axis=0)
                    std_tpr = tprs_arr.std(axis=0)
                    upper = np.minimum(mean_tpr + std_tpr, 1.0)
                    lower = np.maximum(mean_tpr - std_tpr, 0.0)

                    cv_label = "Mean CV ROC"
                    if cv_aucs:
                        valid = [a for a in cv_aucs if not np.isnan(a)]
                        if valid:
                            cv_label += f" (mean AUC={np.mean(valid):.3f})"
                    ax0.fill_between(mean_fpr, lower, upper, color="#1D9E75", alpha=0.18)
                    ax0.plot(
                        mean_fpr,
                        mean_tpr,
                        color="#1D9E75",
                        linewidth=1.6,
                        linestyle="--",
                        label=cv_label,
                    )
            except Exception:
                pass

        if len(roc_fpr) > 0:
            ax0.plot(
                roc_fpr,
                roc_tpr,
                color="#0F4D3A",
                linewidth=2,
                label=f"Training ROC (AUC={full_auc:.3f})",
            )
        ax0.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
        ax0.set_xlabel("False Positive Rate", fontsize=10)
        ax0.set_ylabel("True Positive Rate", fontsize=10)
        ax0.set_title("ROC Curve", fontsize=11)
        # Reorder legend — same convention as the on-screen ROC panel.
        handles, labels = ax0.get_legend_handles_labels()
        preferred_order = [
            "Training ROC",
            "Mean CV ROC",
            "CV folds",
            "Random classifier",
        ]

        def _rank(lbl):
            for i, key in enumerate(preferred_order):
                if lbl.startswith(key):
                    return i
            return len(preferred_order)

        paired = sorted(zip(handles, labels), key=lambda p: _rank(p[1]))
        handles, labels = zip(*paired) if paired else ([], [])
        ax0.legend(handles, labels, fontsize=8, loc="lower right")
        ax0.set_xlim(0, 1)
        ax0.set_ylim(0, 1)
        # Force 1:1 aspect — the academic standard for ROC plots; the
        # diagonal random reference should look like a 45° line.
        ax0.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        return fig

    def _make_jackknife_only_figure(self, jk):
        """Build a standalone Jackknife figure for PNG export.

        Width is fixed at 6 inches; height grows with n_vars so each
        row stays comfortably tall in the rendered file. The drawing
        logic is the same two-row layout as the right panel of the
        dock chart (Only on top, Without below, no Full bar).
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not jk:
            return None

        vars_ = [r["variable"] for r in jk]
        only_tr = [r["only_train_auc"] for r in jk]
        only_te = [r["only_test_auc"] for r in jk]
        wo_tr = [r["without_train_auc"] for r in jk]
        wo_te = [r["without_test_auc"] for r in jk]
        drop_tr = [r.get("train_drop_without", float("nan")) for r in jk]
        drop_te = [r.get("test_drop_without", float("nan")) for r in jk]
        has_test = any(not np.isnan(v) for v in only_te + wo_te)

        def _drop(i):
            d = drop_te[i] if has_test else drop_tr[i]
            return float("-inf") if np.isnan(d) else d

        order = sorted(range(len(vars_)), key=_drop)
        vars_ = [vars_[i] for i in order]
        only_tr = [only_tr[i] for i in order]
        only_te = [only_te[i] for i in order]
        wo_tr = [wo_tr[i] for i in order]
        wo_te = [wo_te[i] for i in order]

        n_vars = len(vars_)
        if has_test:
            only_vals, wo_vals = only_te, wo_te
            only_lbl = "With only variable"
            wo_lbl = "Without variable"
        else:
            only_vals, wo_vals = only_tr, wo_tr
            only_lbl = "With only variable (train)"
            wo_lbl = "Without variable (train)"

        fig_h = max(3.5, 1.4 + n_vars * 0.4)
        fig, ax1 = plt.subplots(figsize=(6, fig_h))

        yp = list(range(n_vars))
        h = 0.35
        ax1.barh(
            [p + 0.5 * h for p in yp],
            only_vals,
            height=h,
            color="#1D9E75",
            linewidth=0,
            label=only_lbl,
        )
        ax1.barh(
            [p - 0.5 * h for p in yp], wo_vals, height=h, color="#C0E6D6", linewidth=0, label=wo_lbl
        )

        # Numeric labels — same approach as the dock figure: long bars
        # get a white in-bar label, short bars get a dark out-of-bar
        # label so the number never collides with its own bar.
        def _label_bar(value, y_pos, dark_bg):
            if np.isnan(value):
                return
            if value >= 0.08:
                ax1.text(
                    0.012,
                    y_pos,
                    f"{value:.3f}",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="white" if dark_bg else "#222222",
                )
            else:
                ax1.text(
                    value + 0.008,
                    y_pos,
                    f"{value:.3f}",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="#222222",
                )

        for k in range(n_vars):
            _label_bar(only_vals[k], k + 0.5 * h, dark_bg=True)
            _label_bar(wo_vals[k], k - 0.5 * h, dark_bg=False)

        for k in range(n_vars):
            if np.isnan(only_vals[k]):
                ax1.text(
                    0.02,
                    k + 0.5 * h,
                    "only-* skipped (categorical)",
                    va="center",
                    ha="left",
                    fontsize=7,
                    style="italic",
                    color="#444444",
                )

        ax1.set_yticks(yp)
        ax1.set_yticklabels(vars_, fontsize=9)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel("AUC", fontsize=10)
        ax1.set_title("Jackknife Variable Importance", fontsize=11)
        ax1.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        return fig

    def _preflight_projection(self, raster_paths, feature_names, meta):
        """Sample the projection rasters and compare them to the model's
        training-time variable ranges.

        Returns:
            (extrap_pct, unknown_cats) where
              extrap_pct: dict[var_name, pct of pixels outside [varmin, varmax]]
              unknown_cats: dict[var_name, list of category codes not seen
                            during training] — only populated for
                            categorical variables.

        The check matters for the cross-region transfer use case
        (e.g. SDM trained in SE Asia, projected to the Caribbean):
          • Continuous extrapolation produces unreliable predictions
            outside the training envelope (Elith et al. 2010,
            *Methods in Ecology and Evolution*). We compute the
            fraction of new-region pixels falling outside the
            training range so the user can decide whether to
            proceed.
          • Categorical "unknown codes" cause elapid's OneHotEncoder
            to raise mid-projection — better to detect this up
            front and refuse with a clear message than to fail
            after 30 seconds of raster I/O.

        Sampling strategy: read each raster at a downsampled
        resolution targeting ~40k pixels. This keeps preflight under
        a second even for continent-scale rasters.
        """
        import numpy as np
        import rasterio

        varmin = meta.get("varmin", {}) or {}
        varmax = meta.get("varmax", {}) or {}
        cat_set = set(meta.get("categorical_indices", []) or [])
        cat_levels = meta.get("categorical_levels", {}) or {}

        TARGET_PIXELS = 40_000
        extrap_pct = {}
        unknown_cats = {}

        for i, (path, name) in enumerate(zip(raster_paths, feature_names)):
            try:
                is_categorical = i in cat_set
                with rasterio.open(path) as src:
                    h, w = src.height, src.width
                    if not is_categorical and h * w > TARGET_PIXELS:
                        # Downsample for the extrapolation check on
                        # continuous variables — extrapolation is a
                        # fraction-of-pixels measure so a stratified
                        # subsample is fine. Categorical rasters MUST
                        # be read at full resolution: downsampling can
                        # silently drop rare codes that nonetheless
                        # appear at full resolution and would crash the
                        # projection mid-run.
                        factor = max(1, int(np.sqrt(h * w / TARGET_PIXELS)))
                        out_h, out_w = max(1, h // factor), max(1, w // factor)
                        arr = src.read(1, out_shape=(out_h, out_w))
                    else:
                        arr = src.read(1)
                    nodata = src.nodata
                vals = arr.flatten().astype(np.float64)
                # Drop NoData. Some drivers store NoData as NaN even
                # when the nodata tag is None, so we filter both.
                if nodata is not None:
                    vals = vals[vals != nodata]
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue

                if i in cat_set:
                    trained = set(cat_levels.get(name, []))
                    if trained:
                        new_codes = set(int(round(float(v))) for v in vals)
                        unseen = new_codes - trained
                        if unseen:
                            unknown_cats[name] = sorted(unseen)
                else:
                    vmin = varmin.get(name)
                    vmax = varmax.get(name)
                    if vmin is not None and vmax is not None and len(vals):
                        out = np.sum((vals < vmin) | (vals > vmax))
                        extrap_pct[name] = 100.0 * float(out) / len(vals)
            except Exception:
                # Per-variable failure must not break the whole
                # preflight. The user can still proceed; the
                # actual projection will surface any real issue.
                pass

        return extrap_pct, unknown_cats

    def _show_preflight_dialog(self, extrap_pct, unknown_cats):
        """Present preflight findings in a single unified dialog.

        Two outcomes can be reported, individually or together:
          • Categorical unknown codes → masked to NoData by
            ``_mask_unknown_categorical_codes``. Informational only.
          • Continuous extrapolation > 10% on any variable → asks
            for explicit confirmation, since extrapolated predictions
            can be unreliable (Elith et al. 2010).

        Returns True unless the user cancels.
        """
        from qgis.PyQt.QtWidgets import QMessageBox

        EXTRAP_THRESHOLD = 10.0
        over = {n: p for n, p in extrap_pct.items() if p >= EXTRAP_THRESHOLD}
        if not unknown_cats and not over:
            return True

        sections = []
        if unknown_cats:
            details = "\n".join(
                f"  • {name}: codes {codes} not present in training data"
                for name, codes in sorted(unknown_cats.items())
            )
            sections.append(
                tr(
                    "Categorical codes unseen in training will be "
                    "masked to NoData in the output (transparent "
                    "pixels):\n\n{details}"
                ).format(details=details)
            )
        if over:
            details = "\n".join(
                f"  • {name}: {pct:.1f}% of pixels outside training range"
                for name, pct in sorted(over.items(), key=lambda kv: -kv[1])
            )
            sections.append(
                tr(
                    "Continuous variables extending beyond the model's "
                    "training range — predictions there may be unreliable "
                    "(Elith et al. 2010):\n\n{details}"
                ).format(details=details)
            )

        body = "\n\n———\n\n".join(sections)

        if over:
            reply = QMessageBox.warning(
                self,
                tr("Projection preflight"),
                body + "\n\n" + tr("Proceed with the projection anyway?"),
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            return reply == QMessageBox.Yes

        QMessageBox.information(
            self,
            tr("Projection preflight"),
            body
            + "\n\n"
            + tr(
                "The projection will continue. Masked pixels appear "
                "as transparent / no-data in the output map."
            ),
        )
        return True

    def _mask_unknown_categorical_codes(self, raster_paths, unknown_cats, feature_names):
        """Write temp copies of categorical rasters where codes the
        model never saw during training are replaced with NoData.

        Returns a new ``raster_paths`` list. Rasters with no unseen
        codes pass through unchanged.
        """
        import os
        import tempfile

        import numpy as np
        import rasterio

        if not unknown_cats:
            return raster_paths

        new_paths = list(raster_paths)
        tmpdir = tempfile.gettempdir()

        for i, (path, name) in enumerate(zip(raster_paths, feature_names)):
            unseen = unknown_cats.get(name)
            if not unseen:
                continue
            try:
                with rasterio.open(path) as src:
                    profile = src.profile.copy()
                    arr = src.read(1)
                    nodata = src.nodata
                if nodata is None:
                    if np.issubdtype(arr.dtype, np.integer):
                        info = np.iinfo(arr.dtype)
                        nodata = -9999 if info.min <= -9999 <= info.max else int(info.min)
                    else:
                        nodata = float("nan")
                mask = np.isin(arr, list(unseen))
                if not mask.any():
                    continue
                arr_out = arr.copy()
                arr_out[mask] = nodata
                profile.update(nodata=nodata)
                safe = "".join(c if c.isalnum() else "_" for c in str(name))
                out_path = os.path.join(tmpdir, f"qmaxent_masked_{safe}.tif")
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(arr_out, 1)
                new_paths[i] = out_path
                QgsMessageLog.logMessage(
                    f"QMaxent: masked {int(mask.sum())} pixels in "
                    f"{name} (unseen codes {sorted(unseen)}) "
                    f"→ {out_path}",
                    "QMaxent",
                    Qgis.Info,
                )
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"QMaxent: failed to mask unknown codes in {name}: {e}",
                    "QMaxent",
                    Qgis.Warning,
                )

        return new_paths

    def _project_model(self):
        if self._model is None:
            self._proj_status.setText(tr("No model. Run Maxent first."))
            return
        output_path = self._proj_path.text().strip()
        if not output_path:
            self._proj_status.setText(tr("Enter output raster path."))
            return

        raster_layers = self._get_raster_layers()
        if raster_layers:
            from ..bridge.raster_bridge import layers_to_paths

            raster_paths = layers_to_paths(raster_layers)
        else:
            raster_paths = (self._meta or {}).get("raster_paths", [])
        if not raster_paths:
            self._proj_status.setText(tr("Set raster layers in ① Data tab."))
            return

        # ── Preflight: training-range comparison ───────────────────────
        # Sample the projection rasters and compare them to the model's
        # training-time variable ranges. Refuses up front if any
        # categorical raster contains codes the model has never seen
        # (projection would otherwise crash mid-run); warns if any
        # continuous variable's projection extent extrapolates >=10%
        # beyond the training envelope. The percentages are also
        # written to the QGIS message log so they survive the dialog
        # and can be cited in academic write-ups.
        feature_names = (self._meta or {}).get("feature_names", [])
        if self._meta and feature_names and len(feature_names) == len(raster_paths):
            self._proj_status.setText(tr("Preflight: sampling rasters..."))
            try:
                extrap_pct, unknown_cats = self._preflight_projection(
                    raster_paths, feature_names, self._meta
                )
            except Exception as e:
                # Preflight is advisory — never block the actual run
                # because of a sampling glitch. Log the failure and
                # let the projection proceed; if there's a real
                # problem the projection itself will surface it.
                QgsMessageLog.logMessage(
                    f"QMaxent preflight failed (continuing anyway): {e}",
                    "QMaxent",
                    Qgis.Warning,
                )
                extrap_pct, unknown_cats = {}, {}
            if extrap_pct:
                tbl = ", ".join(
                    f"{n}={p:.1f}%" for n, p in sorted(extrap_pct.items(), key=lambda kv: -kv[1])
                )
                QgsMessageLog.logMessage(
                    f"QMaxent preflight — extrapolation by variable: {tbl}",
                    "QMaxent",
                    Qgis.Info,
                )
            if not self._show_preflight_dialog(extrap_pct, unknown_cats):
                self._proj_status.setText(tr("Projection cancelled."))
                return

            # v0.1.2 — auto-mask training-unseen categorical codes to
            # NoData so the projection runs across the full extent
            # rather than refusing outright.
            if unknown_cats:
                raster_paths = self._mask_unknown_categorical_codes(
                    raster_paths, unknown_cats, feature_names
                )

        self._proj_btn.setEnabled(False)
        self._proj_progress.setValue(0)
        self._proj_progress.show()
        self._proj_status.setText(tr("Running projection..."))

        transform = self._proj_transform.currentText()
        self._model.transform = transform

        from ..workers.projection_worker import ProjectionWorker

        self._proj_worker = ProjectionWorker(
            model=self._model,
            raster_paths=raster_paths,
            output_path=output_path,
            parent=self,
        )
        # ① connect determinate progress
        self._proj_worker.progress.connect(self._on_proj_progress)
        self._proj_worker.finished.connect(self._on_projection_finished)
        self._proj_worker.start()

    def _on_proj_progress(self, pct: int, msg: str):
        # The progress bar already renders the percentage in its own
        # text, so the status label only needs a brief activity note.
        # We deliberately ignore the worker's "Window N/M" message
        # because it surfaces a rasterio implementation detail (block
        # iteration over the output GeoTIFF) that's meaningless to
        # users — they care about completion, not block indices.
        self._proj_progress.setValue(pct)
        if pct >= 100:
            self._proj_status.setText(tr("Projection complete."))
        else:
            self._proj_status.setText(tr("Projecting..."))

    def _on_projection_finished(self, success: bool, msg: str):
        self._proj_btn.setEnabled(True)
        self._proj_progress.hide()
        self._proj_worker = None

        output_path = self._proj_path.text().strip()
        transform = self._proj_transform.currentText()

        if not success:
            self._proj_status.setText(f"✗ {msg[:200]}")
            return

        self._proj_status.setText(tr("Done: {path}").format(path=output_path))
        self._log_append(tr("Done: {path}").format(path=output_path))

        # ④ Save PNG charts alongside the raster
        self._export_charts_as_png(output_path)

        # v0.1.7: training_log.txt auto-save was moved to _on_finished
        # (▶ Run Maxent completion) so the log accompanies model.pkl,
        # not the prediction raster — model.pkl is the artifact a
        # researcher carries between sessions, while prediction.tif is
        # a derived re-runnable output.

        # New prediction raster is on disk → recompute the Discovery
        # minimum-suitability floor (raster_max × 0.9) so the
        # ⑤ Priority Sites tab opens with a value adapted to this
        # specific dataset rather than the static 0.9 placeholder.
        try:
            self._refresh_discovery_floor()
        except Exception:
            pass

        if self._proj_load_chk.isChecked():
            try:
                from ..bridge.raster_bridge import load_raster_to_qgis

                name = os.path.splitext(os.path.basename(output_path))[0]
                load_raster_to_qgis(output_path, name, transform=transform)
                self._log_append(tr("Layer loaded: {name}").format(name=name))
            except Exception as e:
                self._proj_status.setText(
                    self._proj_status.text() + f"\n{tr('Layer load error: {e}').format(e=e)}"
                )

    def _export_charts_as_png(self, raster_path: str):
        """Save Response Curves, ROC, Jackknife, and Permutation charts as
        separate PNG files alongside the prediction GeoTIFF.

        Four files are written when the analyses are available:
          • <name>_response_curves.png   — one image, all variables
          • <name>_roc.png               — ROC panel (training + CV folds + mean)
          • <name>_jackknife.png         — variable importance bars (jackknife)
          • <name>_permutation.png       — variable importance bars (permutation)

        ROC and the two importance plots are split into separate files
        because they get cited independently in academic figures —
        Jackknife usually in the main text, ROC in supplementary or
        vice versa — and each panel deserves its own aspect ratio:
        ROC ≈ square, the bar charts wide-and-tall-by-n_vars.

        The whole step is skipped when the user has unticked the
        "Save analysis charts as PNG" checkbox.
        """
        if (
            not getattr(self, "_proj_save_charts_chk", None)
            or not self._proj_save_charts_chk.isChecked()
        ):
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = os.path.dirname(raster_path)
        base = os.path.splitext(os.path.basename(raster_path))[0]
        saved = []

        # ── Response curves grid ──────────────────────────────────
        try:
            fig = self._make_response_figure()
            if fig is not None:
                p = os.path.join(out_dir, f"{base}_response_curves.png")
                fig.savefig(p, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved.append(os.path.basename(p))
        except Exception as e:
            self._log_append(f"PNG export (response curves) failed: {e}")

        # ── ROC panel (standalone) ────────────────────────────────
        try:
            roc_fpr = (self._results or {}).get("roc_fpr", [])
            roc_tpr = (self._results or {}).get("roc_tpr", [])
            cv_aucs = (self._results or {}).get("cv_aucs", [])
            fauc = (self._results or {}).get("full_auc")
            if roc_fpr or cv_aucs:
                fig = self._make_roc_only_figure(
                    roc_fpr,
                    roc_tpr,
                    cv_aucs,
                    fauc,
                    cv_roc_fpr_list=(self._results or {}).get("cv_roc_fpr_list", []),
                    cv_roc_tpr_list=(self._results or {}).get("cv_roc_tpr_list", []),
                )
                if fig is not None:
                    p = os.path.join(out_dir, f"{base}_roc.png")
                    fig.savefig(p, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    saved.append(os.path.basename(p))
        except Exception as e:
            self._log_append(f"PNG export (ROC) failed: {e}")

        # ── Jackknife panel (standalone) ──────────────────────────
        try:
            jk = (self._results or {}).get("jackknife_results", [])
            if jk:
                fig = self._make_jackknife_only_figure(jk)
                if fig is not None:
                    p = os.path.join(out_dir, f"{base}_jackknife.png")
                    fig.savefig(p, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    saved.append(os.path.basename(p))
        except Exception as e:
            self._log_append(f"PNG export (jackknife) failed: {e}")

        # ── Permutation importance panel (standalone) ─────────────
        # Added v0.1.7: previously only the manual "Save PNG..."
        # button on the Permutation Importance sub-tab could save
        # this figure. With the "Save analysis charts as PNG"
        # checkbox now covering it too, the four variable-analysis
        # outputs all land together when the user opts in.
        try:
            pi_rows = (self._results or {}).get("permutation_results", []) or []
            if pi_rows:
                fig = self._make_permutation_figure(
                    pi_rows,
                    source=(self._results or {}).get("permutation_source", "training set"),
                    n_repeats=(self._results or {}).get("permutation_n_repeats", 10),
                )
                if fig is not None:
                    p = os.path.join(out_dir, f"{base}_permutation.png")
                    fig.savefig(p, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    saved.append(os.path.basename(p))
        except Exception as e:
            self._log_append(f"PNG export (permutation) failed: {e}")

        if saved:
            self._log_append(f"PNG charts saved: {', '.join(saved)}")

    # =========================================================================
    # Utilities
    # =========================================================================

    def _set_status(self, text: str):
        self._status_lbl.setText(text)

    def _clear_widget(self, widget: QWidget):
        """Ensure exactly one QVBoxLayout; clear children without recreating."""
        layout = widget.layout()
        if layout is None:
            new = QVBoxLayout(widget)
            new.setContentsMargins(0, 0, 0, 0)
        else:
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w:
                    w.hide()
                    w.deleteLater()

    def _on_load_model_clicked(self):
        """Entry point for the 'Load existing model (.pkl)...' button."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Load QMaxent model"),
            "",
            tr("Pickle files (*.pkl)"),
        )
        if not path:
            return
        self.load_model_from_file(path)

    def load_model_from_file(self, path: str):
        """Load a .pkl, then ask the user to map model variables to rasters.

        The matching step is mandatory because elapid's
        apply_model_to_rasters relies on raster *position*, not name.
        Allowing the user to skip mapping would silently produce wrong
        predictions when raster ordering differs from training time.
        """
        try:
            from ..core.venv_manager import ensure_venv_packages_available

            ensure_venv_packages_available()
            from ..bridge import elapid_bridge as eb

            model = eb.load_object(path)
        except Exception as e:
            self._set_status(f"✗ {e}")
            return

        meta = getattr(model, "_qmaxent_meta", {}) or {}
        feature_names = meta.get("feature_names", [])
        if not feature_names:
            self._set_status(
                tr("Loaded file has no QMaxent metadata (was it saved by this plugin?).")
            )
            return

        # Variable-mapping dialog (mandatory)
        from .model_load_dialog import ModelLoadDialog

        dlg = ModelLoadDialog(
            self,
            feature_names=feature_names,
            categorical_indices=meta.get("categorical_indices") or [],
        )
        from qgis.PyQt.QtWidgets import QDialog

        if dlg.exec_() != QDialog.Accepted:
            self._set_status(tr("Model load cancelled."))
            return

        layers = dlg.selected_layers()
        if any(l is None for l in layers) or len(layers) != len(feature_names):
            self._set_status(tr("Model load cancelled."))
            return

        # Apply the user's matching: rebuild the Data-tab raster list in
        # the model's training order, and adopt the model. Categorical
        # flags are restored from the saved metadata so the loaded model
        # behaves identically to the moment it was trained.
        cat_set = set(meta.get("categorical_indices", []) or [])
        self._raster_list.clear()
        for i, (lyr, name) in enumerate(zip(layers, feature_names)):
            self._add_raster_item(
                layer_name=lyr.name(),
                layer_id=lyr.id(),
                is_categorical=(i in cat_set),
                display_text=f"{name}  ←  {lyr.name()}",
            )

        self._model = model
        self._meta = meta
        # Restore academic results (ROC, AUC, CV AUCs, jackknife) from
        # the saved meta so the Results tab populates exactly as it
        # would after fresh training. Models trained before this
        # feature shipped have no ``academic_results`` key; the .get
        # default keeps the load path working with empty charts, which
        # is the same behaviour the user already saw on v0.1.0 .pkl
        # files.
        academic = meta.get("academic_results", {}) or {}
        self._results = {
            "model": model,
            "meta": meta,
            "feature_names": meta.get("feature_names", []),
            "roc_fpr": list(academic.get("roc_fpr", [])),
            "roc_tpr": list(academic.get("roc_tpr", [])),
            "full_auc": academic.get("full_auc"),
            "jk_full_train_auc": academic.get("jk_full_train_auc"),
            "jk_full_test_auc": academic.get("jk_full_test_auc"),
            "cv_aucs": list(academic.get("cv_aucs", [])),
            "cv_roc_fpr_list": list(academic.get("cv_roc_fpr_list", [])),
            "cv_roc_tpr_list": list(academic.get("cv_roc_tpr_list", [])),
            "jackknife_results": list(academic.get("jackknife_results", [])),
            # Permutation importance restored from .pkl (added v0.1.3).
            # Same defensive list-copy pattern as jackknife to keep the
            # restored result dict self-contained.
            "permutation_results": list(academic.get("permutation_results", [])),
            "permutation_source": academic.get("permutation_source"),
            "permutation_n_repeats": academic.get("permutation_n_repeats"),
        }
        self._populate_response_combo()
        # Replay the importance plot so the Jackknife tab shows the
        # restored data instead of staying blank from a previous run.
        try:
            self._plot_importance()
            self._plot_permutation()
        except Exception:
            # Plot routine can fail when matplotlib isn't installed
            # yet (e.g., user loads model before installing deps).
            # The chart will simply stay blank — non-fatal.
            pass
        # Sync the Output transform combo with the model's stored value
        saved_trans = meta.get("transform", "cloglog")
        idx = self._proj_transform.findText(saved_trans)
        if idx >= 0:
            self._proj_transform.setCurrentIndex(idx)
        self._proj_btn.setEnabled(True)
        # Same activation as fresh training — see _on_run_finished.
        self._priority_btn.setEnabled(True)
        try:
            self._on_priority_threshold_method_changed(self._priority_thr_method.currentIndex())
        except Exception:
            pass
        # Status bar: show the restored academic metrics alongside the
        # filename so the user immediately sees the same train/CV AUC
        # numbers they had at training time. Falls back to the
        # plain "Loaded: ..." message for old .pkl files that don't
        # carry academic_results.
        n_p = meta.get("n_presence")
        n_b = meta.get("n_background")
        fauc = self._results.get("full_auc")
        cv_aucs = self._results.get("cv_aucs", [])
        parts = [
            tr("✓ Model loaded: {name}  ({n} variables)").format(
                name=os.path.basename(path),
                n=len(feature_names),
            )
        ]
        if isinstance(n_p, int) and isinstance(n_b, int):
            parts.append(
                tr("presence={n}").format(n=n_p) + "  " + tr("background={n}").format(n=f"{n_b:,}")
            )
        if fauc is not None:
            parts.append(tr("train AUC={v:.4f}").format(v=fauc))
        if cv_aucs:
            valid = [
                a for a in cv_aucs if a is not None and not (isinstance(a, float) and a != a)
            ]  # NaN filter
            if valid:
                parts.append(tr("CV AUC={v:.4f}").format(v=float(np.mean(valid))))
        self._set_status("  |  ".join(parts))
        self.tabs.setCurrentIndex(3)
