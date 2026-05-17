"""Background worker thread for installing QMaxent dependencies."""

import traceback

from qgis.PyQt.QtCore import QThread, pyqtSignal


class DepsInstallWorker(QThread):
    """Runs dependency installation in a background thread.

    Signals:
        progress(int, str): (percent 0-100, message)
        finished(bool, str): (success, message)
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from ..core.venv_manager import create_venv_and_install

            success, message = create_venv_and_install(
                progress_callback=lambda p, m: self.progress.emit(p, m),
                cancel_check=lambda: self._cancelled,
            )
            self.finished.emit(success, message)
        except Exception as e:
            self.finished.emit(False, f"{e}\n{traceback.format_exc()}")
