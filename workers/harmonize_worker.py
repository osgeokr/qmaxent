"""Background worker for explicit raster harmonization.

Runs bridge.raster_bridge.harmonize_rasters in a QThread so the QGIS
UI stays responsive while big rasters are reprojected, resampled, and
clipped. Triggered from the "Harmonize to Folder…" button in the
① Data tab; emits per-raster progress so the modal progress dialog
can show 0-100% and the file currently being processed.
"""

import traceback

from qgis.PyQt.QtCore import QThread, pyqtSignal


class HarmonizeWorker(QThread):
    """Background worker wrapping harmonize_rasters().

    Signals:
        progress(int, str): percent 0-100 and per-raster status text.
        finished(bool, str, list): success flag, message, and the list
            of harmonized raster paths (empty on failure).

    Cancellation:
        Call ``cancel()`` from the GUI thread to request a clean stop.
        ``harmonize_rasters`` polls this flag at the start of each
        per-raster iteration; on cancel the worker emits
        ``finished(False, "Cancelled", [])`` and any partially written
        outputs in the user-chosen folder are left in place (the user
        chose that folder, so we don't auto-delete its contents).
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, list)

    def __init__(self, raster_paths: list, output_dir: str,
                 categorical_indices: list = None,
                 template_idx: int = 0,
                 parent=None):
        super().__init__(parent)
        self._raster_paths        = list(raster_paths)
        self._output_dir          = output_dir
        self._categorical_indices = list(categorical_indices or [])
        self._template_idx        = int(template_idx)
        self._cancelled           = False

    def cancel(self):
        """Request a clean stop. Safe to call from the GUI thread."""
        self._cancelled = True

    def run(self):
        try:
            from ..bridge.raster_bridge import harmonize_rasters

            new_paths, _ = harmonize_rasters(
                self._raster_paths,
                categorical_indices=self._categorical_indices,
                template_idx=self._template_idx,
                output_dir=self._output_dir,
                progress_callback=self._on_progress,
                cancel_check=lambda: self._cancelled,
            )
            if self._cancelled:
                # cancel_check inside harmonize_rasters raises
                # RuntimeError("Cancelled"); this branch only triggers
                # if cancel was set after the loop returned normally.
                self.finished.emit(False, "Cancelled", [])
                return
            self.finished.emit(
                True,
                f"Harmonized {len(new_paths)} raster(s) to {self._output_dir}",
                new_paths,
            )
        except RuntimeError as e:
            # Distinguish user-cancel from a genuine error so the
            # dialog can show a friendlier message.
            if str(e) == "Cancelled":
                self.finished.emit(False, "Cancelled", [])
            else:
                self.finished.emit(
                    False,
                    f"{e}\n{traceback.format_exc()}",
                    [],
                )
        except Exception as e:
            self.finished.emit(
                False,
                f"{e}\n{traceback.format_exc()}",
                [],
            )

    def _on_progress(self, pct: int, msg: str):
        # Re-emit on the worker's signal so the dialog (running in the
        # GUI thread) can update its progress bar without touching
        # widgets from this thread.
        self.progress.emit(pct, msg)
