"""Dialog: load presence records from a CSV / TXT table (v0.2.0).

Reviewer comment R1 asked for a wider range of input formats. QMaxent
already accepts any point layer QGIS can open; this dialog removes the
extra step for the most common case, a table with longitude / latitude
(or X / Y) columns such as a GBIF export or the dismo Bradypus file.
"""

import csv
import os
import urllib.parse

from qgis.core import QgsCoordinateReferenceSystem, QgsProject, QgsVectorLayer
from qgis.gui import QgsProjectionSelectionWidget
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..i18n import tr

_X_NAMES = ("decimallongitude", "longitude", "long", "lon", "lng", "x", "easting")
_Y_NAMES = ("decimallatitude", "latitude", "lat", "y", "northing")
_DELIMS = {",": "Comma (,)", ";": "Semicolon (;)", "\t": "Tab", "|": "Pipe (|)"}


def sniff_table(path: str):
    """Return (delimiter, header list) for a delimited text file."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        sample = fh.read(65536)
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delim = ","
    first = sample.splitlines()[0] if sample else ""
    header = next(csv.reader([first], delimiter=delim)) if first else []
    return delim, [h.strip() for h in header]


def guess_xy(header):
    """Pick the most likely X and Y columns (case-insensitive)."""
    low = [h.lower() for h in header]

    def pick(names):
        for n in names:
            if n in low:
                return header[low.index(n)]
        return None

    return pick(_X_NAMES), pick(_Y_NAMES)


def make_layer(path, delim, xf, yf, crs_authid, name=None):
    """Create a delimited-text point layer (not yet added to the project)."""
    q = {
        "type": "csv",
        "delimiter": delim,
        "xField": xf,
        "yField": yf,
        "crs": crs_authid,
        "detectTypes": "yes",
        "geomType": "point",
        "spatialIndex": "no",
        "subsetIndex": "no",
        "watchFile": "no",
    }

    # Same form QGIS's own Data Source Manager writes: plain values,
    # only characters that would break the query are escaped. Fully
    # percent-encoded values (e.g. delimiter=%3B) are not decoded by the
    # delimited-text provider.
    def _v(val):
        if val == "\t":
            return "%5Ct"
        return urllib.parse.quote(str(val), safe=":;,|_.-()")

    uri = QUrl.fromLocalFile(path).toString() + "?" + "&".join(f"{k}={_v(v)}" for k, v in q.items())
    name = name or os.path.splitext(os.path.basename(path))[0]
    return QgsVectorLayer(uri, name, "delimitedtext")


class CsvPresenceDialog(QDialog):
    """Pick a CSV, the X/Y columns and the CRS; returns a point layer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Load presence points from CSV"))
        self.setMinimumWidth(460)
        self.layer = None

        v = QVBoxLayout(self)
        form = QFormLayout()
        file_row = QHBoxLayout()
        self._path = QLineEdit()
        browse = QPushButton(tr("Browse..."))
        browse.clicked.connect(self._browse)
        file_row.addWidget(self._path)
        file_row.addWidget(browse)
        form.addRow(tr("File:"), file_row)

        self._delim = QComboBox()
        for d, label in _DELIMS.items():
            self._delim.addItem(label, d)
        self._delim.currentIndexChanged.connect(self._reload_header)
        form.addRow(tr("Delimiter:"), self._delim)

        self._x = QComboBox()
        self._y = QComboBox()
        form.addRow(tr("X / longitude column:"), self._x)
        form.addRow(tr("Y / latitude column:"), self._y)

        self._crs = QgsProjectionSelectionWidget()
        self._crs.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
        form.addRow(tr("Coordinate reference system:"), self._crs)
        v.addLayout(form)

        hint = QLabel(
            tr(
                "Decimal-degree tables (e.g. GBIF decimalLongitude / "
                "decimalLatitude) use EPSG:4326. The layer is added to the "
                "project and selected as the presence layer."
            )
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(hint)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Select presence table"),
            "",
            tr("Delimited text (*.csv *.txt *.tsv);;All files (*)"),
        )
        if not path:
            return
        self._path.setText(path)
        delim, _ = sniff_table(path)
        idx = self._delim.findData(delim)
        if idx >= 0:
            self._delim.blockSignals(True)
            self._delim.setCurrentIndex(idx)
            self._delim.blockSignals(False)
        self._reload_header()

    def _reload_header(self):
        path = self._path.text().strip()
        if not os.path.isfile(path):
            return
        delim = self._delim.currentData()
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            first = fh.readline()
        header = [h.strip() for h in next(csv.reader([first], delimiter=delim))]
        self._x.clear()
        self._y.clear()
        self._x.addItems(header)
        self._y.addItems(header)
        gx, gy = guess_xy(header)
        if gx:
            self._x.setCurrentText(gx)
        if gy:
            self._y.setCurrentText(gy)

    def _accept(self):
        path = self._path.text().strip()
        if not os.path.isfile(path):
            QMessageBox.warning(self, self.windowTitle(), tr("Choose an existing CSV / TXT file."))
            return
        xf, yf = self._x.currentText(), self._y.currentText()
        if not xf or not yf or xf == yf:
            QMessageBox.warning(
                self, self.windowTitle(), tr("Choose two different columns for X and Y.")
            )
            return
        crs = self._crs.crs()
        if not crs.isValid():
            QMessageBox.warning(
                self, self.windowTitle(), tr("Choose a coordinate reference system.")
            )
            return
        lyr = make_layer(path, self._delim.currentData(), xf, yf, crs.authid() or crs.toWkt())
        if not lyr.isValid() or lyr.featureCount() == 0:
            QMessageBox.warning(
                self,
                self.windowTitle(),
                tr(
                    "No points could be read. Check the delimiter and "
                    "that the X / Y columns contain numbers."
                ),
            )
            return
        QgsProject.instance().addMapLayer(lyr)
        self.layer = lyr
        self.accept()
