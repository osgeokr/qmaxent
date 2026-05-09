"""QMaxent QGIS Plugin — main entry point."""

from qgis.core import QgsMessageLog, Qgis
from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtCore import Qt


class QMaxentPlugin:
    """QGIS plugin class."""

    def __init__(self, iface):
        self.iface    = iface
        self._main_dock  = None
        self._setup_dock = None
        self._actions    = []

    def initGui(self):
        # ── Dependency check ──────────────────────────────────────────────
        from .i18n import tr
        from .core.venv_manager import (
            get_venv_status, ensure_venv_packages_available,
            cleanup_sidelined_venvs,
        )
        # First, sweep up any venv folders that were renamed-aside in a
        # previous QGIS session because their .pyd files were locked at
        # remove time. The locks are gone now, so rmtree usually
        # succeeds. Best-effort: any folder still holding out gets
        # retried on the next launch.
        n_swept = cleanup_sidelined_venvs()
        if n_swept:
            QgsMessageLog.logMessage(
                f"QMaxent: cleaned up {n_swept} orphaned venv folder(s) "
                f"from previous session", "QMaxent", Qgis.Info,
            )
        ready, msg = get_venv_status()
        if ready:
            ensure_venv_packages_available()
            QgsMessageLog.logMessage(f"QMaxent ready: {msg}", "QMaxent", Qgis.Info)
        else:
            QgsMessageLog.logMessage(
                f"QMaxent: dependencies not installed — {msg}", "QMaxent", Qgis.Warning
            )

        # ── Menu actions ──────────────────────────────────────────────────
        panel_action = QAction(tr("QMaxent Analysis"), self.iface.mainWindow())
        panel_action.triggered.connect(self._show_main_dock)
        self.iface.addPluginToMenu("QMaxent", panel_action)
        self._actions.append(panel_action)

        setup_action = QAction(tr("QMaxent Dependencies"), self.iface.mainWindow())
        setup_action.triggered.connect(self._show_setup_dock)
        self.iface.addPluginToMenu("QMaxent", setup_action)
        self._actions.append(setup_action)

        # Optional: a one-click way to fetch elapid's example dataset
        # so first-time users have working data without external setup.
        # Mirrors the convention of dismo, SDMtune, ENMeval, biomod2 —
        # all of which ship example data with their R packages.
        example_action = QAction(
            tr("Download Example Dataset..."), self.iface.mainWindow()
        )
        example_action.triggered.connect(self._show_example_dialog)
        self.iface.addPluginToMenu("QMaxent", example_action)
        self._actions.append(example_action)

        # ── Auto-open docks ───────────────────────────────────────────────
        if not ready:
            # Dependency missing: show setup dock FIRST and prominently
            self._show_setup_dock()
            self._show_main_dock()   # main dock behind setup
        else:
            self._show_main_dock()

    def unload(self):
        for action in self._actions:
            self.iface.removePluginMenu("QMaxent", action)
        if self._main_dock:
            self.iface.removeDockWidget(self._main_dock)
            self._main_dock = None
        if self._setup_dock:
            self.iface.removeDockWidget(self._setup_dock)
            self._setup_dock = None

    def _show_main_dock(self):
        if self._main_dock is None:
            from .dialogs.main_dock import QMaxentMainDock
            self._main_dock = QMaxentMainDock(self.iface, self.iface.mainWindow())
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self._main_dock)
        self._main_dock.show()
        self._main_dock.raise_()

    def _show_setup_dock(self):
        if self._setup_dock is None:
            from .dialogs.setup_dock import SetupDockWidget
            self._setup_dock = SetupDockWidget(self.iface.mainWindow())
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self._setup_dock)
        self._setup_dock.show()
        self._setup_dock.raise_()

    def _show_example_dialog(self):
        """Open the example-data download dialog."""
        from .dialogs.example_data_dialog import ExampleDataDialog
        dlg = ExampleDataDialog(self.iface, self.iface.mainWindow())
        dlg.exec_()

    def get_main_dock(self):
        """Return the main dock (create if needed)."""
        if self._main_dock is None:
            self._show_main_dock()
        return self._main_dock
