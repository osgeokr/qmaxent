"""QMaxent Setup Dock — dependency installation panel."""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QPushButton, QGroupBox,
)
from qgis.PyQt.QtGui import QFont

from ..i18n import tr


class SetupDockWidget(QDockWidget):
    """Guides the user through first-time dependency installation."""

    def __init__(self, parent=None):
        super().__init__(tr("QMaxent — Dependencies"), parent)
        self.setObjectName("QMaxentSetupDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._worker = None
        self._build_ui()
        self._refresh_status()

    def _build_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Status
        status_group = QGroupBox(tr("Environment Status"))
        sg = QVBoxLayout(status_group)
        self.status_label = QLabel(tr("Checking..."))
        self.status_label.setWordWrap(True)
        sg.addWidget(self.status_label)
        layout.addWidget(status_group)

        # About — three sections:
        #   (1) Plugin info (author, version, license, repository)
        #   (2) Citation (how to cite this plugin in publications)
        #   (3) Dependencies (what gets installed and where)
        about_group = QGroupBox(tr("About"))
        ag = QVBoxLayout(about_group)

        # (1) Plugin info — compact rich-text block
        plugin_info = QLabel(tr(
            "<b>QMaxent</b> 0.1.0<br>"
            "Author: Byeong-Hyeok Yu &lt;bhyu@knps.or.kr&gt;<br>"
            "License: MIT — Copyright © 2026 Byeong-Hyeok Yu<br>"
            "Repository: "
            "<a href=\"https://github.com/osgeokr/qmaxent\">"
            "github.com/osgeokr/qmaxent</a>"
        ))
        plugin_info.setWordWrap(True)
        plugin_info.setOpenExternalLinks(True)
        plugin_info.setTextInteractionFlags(Qt.TextBrowserInteraction)
        ag.addWidget(plugin_info)

        # Note: Citation block is intentionally omitted until the SoftwareX
        # paper is accepted and assigned a DOI. The CITATION.cff file in
        # the repository root provides programmatic citation metadata in
        # the meantime (GitHub's "Cite this repository" button reads it).

        # (2) Dependencies
        info_text = QLabel(tr(
            "<b>Dependencies:</b><br>"
            "QMaxent installs its dependencies into an isolated virtual "
            "environment so they do not affect QGIS.<br><br>"
            "One-time setup installs:<br>"
            "&nbsp;&nbsp;• elapid (Maxent engine)<br>"
            "&nbsp;&nbsp;• rasterio, geopandas (spatial I/O)<br>"
            "&nbsp;&nbsp;• scikit-learn, scipy, numpy<br>"
            "&nbsp;&nbsp;• matplotlib (result plots)<br><br>"
            "Approximate size: 300–500 MB"
        ))
        info_text.setWordWrap(True)
        info_text.setStyleSheet("padding-top: 6px;")
        ag.addWidget(info_text)

        from ..core.venv_manager import CACHE_DIR
        import os
        home = os.path.expanduser("~")
        display = ("~" + CACHE_DIR[len(home):]) if CACHE_DIR.startswith(home) else CACHE_DIR
        loc_label = QLabel(
            f"<small>{tr('Install location: ')}<code>{display}</code></small>"
        )
        loc_label.setWordWrap(True)
        ag.addWidget(loc_label)
        layout.addWidget(about_group)

        # Buttons
        btn_row = QHBoxLayout()
        self.install_btn = QPushButton(tr("Install / Update Dependencies"))
        self.install_btn.setMinimumHeight(34)
        bold = QFont(); bold.setBold(True)
        self.install_btn.setFont(bold)
        self.install_btn.clicked.connect(self._start_install)
        btn_row.addWidget(self.install_btn)

        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.setMinimumHeight(34)
        self.cancel_btn.clicked.connect(self._cancel_install)
        self.cancel_btn.hide()
        btn_row.addWidget(self.cancel_btn)

        self.remove_btn = QPushButton(tr("Remove Environment"))
        self.remove_btn.setMinimumHeight(28)
        self.remove_btn.clicked.connect(self._remove_venv)
        btn_row.addWidget(self.remove_btn)
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.hide()
        layout.addWidget(self.progress_label)

        layout.addStretch()
        self.setWidget(container)

    def _refresh_status(self):
        from ..core.venv_manager import get_venv_status
        ready, msg = get_venv_status()
        if ready:
            self.status_label.setText(f"✓ {msg}")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_label.setText(f"✗ {msg}")
            self.status_label.setStyleSheet("color: #cc0000; font-weight: bold;")

    def _start_install(self):
        from ..workers.deps_install_worker import DepsInstallWorker
        self.install_btn.hide()
        self.remove_btn.hide()
        self.cancel_btn.show()
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.progress_label.setText(tr("Starting installation..."))
        self.progress_label.show()
        self._worker = DepsInstallWorker(parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel_install(self):
        if self._worker:
            self._worker.cancel()
        self.progress_label.setText(tr("Cancelling..."))

    def _on_progress(self, percent: int, message: str):
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)

    def _on_finished(self, success: bool, message: str):
        self.cancel_btn.hide()
        self.install_btn.show()
        self.remove_btn.show()
        self.progress_bar.hide()
        self.progress_label.hide()
        if success:
            from ..core.venv_manager import ensure_venv_packages_available
            ensure_venv_packages_available()
            self.status_label.setText(f"✓ {message}")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_label.setText(f"✗ {message}")
            self.status_label.setStyleSheet("color: #cc0000;")
        self._worker = None

    def _remove_venv(self):
        from qgis.PyQt.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, tr("Remove Environment"),
            tr(
                "Remove the QMaxent virtual environment?\n"
                "You will need to reinstall dependencies before using QMaxent again."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        from ..core.venv_manager import remove_venv, VENV_DIR
        ok, msg = remove_venv()
        # Always surface the outcome. The previous version dropped
        # this on the floor, so partial-removal cases (locked .pyd
        # files on Windows) were invisible to the user — and any
        # reinstall on top of a sidelined-or-half-removed folder
        # would produce the cryptic "init rasterio._base" cython
        # import error. Showing the message gives the user a chance
        # to restart QGIS first when relevant.
        if ok:
            # "Venv removed" is unremarkable; the long sideline message
            # is the one users actually need to see, so show it as info.
            if msg and msg != "Venv removed" and msg != "Venv does not exist":
                self._show_outcome_dialog(
                    QMessageBox.Information, msg, VENV_DIR,
                )
        else:
            self._show_outcome_dialog(
                QMessageBox.Warning, msg, VENV_DIR,
            )
        self._refresh_status()

    def _show_outcome_dialog(self, icon, message: str, venv_dir: str):
        """Show the remove-venv outcome with an [Open folder] button.

        Used for both the sidelined-success info and the rename-failure
        error. The auxiliary button calls the OS-native file manager so
        the user can complete a manual delete in one click instead of
        copying the path out of the message and pasting into Explorer.
        """
        import os
        import sys
        import subprocess
        from qgis.PyQt.QtWidgets import QMessageBox

        mbox = QMessageBox(self)
        mbox.setIcon(icon)
        mbox.setWindowTitle(tr("Remove Environment"))
        mbox.setText(message)

        # Only offer the open-folder button when the folder actually
        # exists on disk — otherwise the button would launch a window
        # at a non-existent path on some platforms.
        target = venv_dir if os.path.isdir(venv_dir) else (
            os.path.dirname(venv_dir)
            if os.path.isdir(os.path.dirname(venv_dir)) else None
        )
        open_btn = None
        if target:
            open_btn = mbox.addButton(
                tr("Open folder in file manager"),
                QMessageBox.ActionRole,
            )
        ok_btn = mbox.addButton(QMessageBox.Ok)
        mbox.setDefaultButton(ok_btn)
        mbox.exec_()

        if open_btn is not None and mbox.clickedButton() is open_btn:
            try:
                if sys.platform == "win32":
                    os.startfile(target)  # noqa: S606  Windows-only
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", target])
                else:
                    subprocess.Popen(["xdg-open", target])
            except Exception as e:
                # File manager launch failure is not fatal — the user
                # still has the path in the message above. Log it for
                # diagnostics and move on silently.
                from qgis.core import QgsMessageLog, Qgis
                QgsMessageLog.logMessage(
                    f"Could not open file manager for {target}: {e}",
                    "QMaxent", Qgis.Warning,
                )
