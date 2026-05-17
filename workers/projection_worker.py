"""Background worker for spatial model projection."""

import traceback

from qgis.PyQt.QtCore import QThread, pyqtSignal


class ProjectionWorker(QThread):
    """Runs apply_model_to_rasters in a background thread.

    Signals:
        progress(int, str): percent 0-100 and window status.
        finished(bool, str): success flag and message.

    Like the training worker, this also performs a raster consistency
    check before projection — but as a fail-fast safety net rather
    than an automatic harmonization step. The user is expected to
    have run "Check Raster Consistency" → "Harmonize to Folder…" in
    the ① Data tab before pressing Run Spatial Projection. If the
    rasters disagree we abort with a clear, actionable error so the
    user can harmonize and retry.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, model, raster_paths: list, output_path: str,
                 parent=None):
        super().__init__(parent)
        self._model        = model
        self._raster_paths = raster_paths
        self._output_path  = output_path

    def run(self):
        try:
            # Auto-create the parent directory of the prediction raster
            # so the user can type a fresh output path without first
            # running mkdir — matches the same guarantee made by the
            # MaxentWorker for model.pkl / results.xlsx.
            try:
                import os as _os
                _d = _os.path.dirname(self._output_path)
                if _d:
                    _os.makedirs(_d, exist_ok=True)
            except OSError as _e:
                self.finished.emit(
                    False,
                    f"Could not create output directory: {_e}",
                    "",
                )
                return

            # ── Disable tqdm's background monitor thread ─────────────────
            # tqdm starts a daemon Thread (TMonitor) the first time
            # tqdm.tqdm is instantiated — it wakes every `monitor_interval`
            # seconds (default 10) to refresh stalled progress bars. On
            # Windows + Python 3.12 + Qt threading we observed that this
            # daemon thread outlives the QThread that created it and
            # later triggers a native access-violation crash inside
            # `tqdm/_monitor.py` (`self.was_killed.wait(self.sleep_interval)`)
            # while later plugin code is running (Priority Sites reverse
            # geocoding was the most reproducible trigger). Setting
            # monitor_interval = 0 *before* the first tqdm.tqdm()
            # instantiation tells tqdm to skip starting the monitor
            # thread entirely. We don't lose anything functional —
            # tqdm's progress bars still work, only the auxiliary
            # "stalled bar refresher" is gone, and we don't even use
            # tqdm's UI here (we monkey-patch it below). Belt-and-
            # braces: also try to stop any monitor that might already
            # be running from an earlier run.
            try:
                import tqdm as _tqdm
                _tqdm.tqdm.monitor_interval = 0
                # Stop a previously-started monitor, if any. tqdm
                # exposes a private `_instances` set that the monitor
                # walks; clearing the monitor's `was_killed` event
                # then setting `monitor_interval = 0` makes the
                # current monitor (if any) exit on its next loop
                # iteration. Wrapped in try/except because the API
                # is not officially public and may differ across
                # tqdm versions.
                try:
                    from tqdm._monitor import TMonitor
                    mon = getattr(_tqdm.tqdm, "monitor", None)
                    if mon is not None and hasattr(mon, "was_killed"):
                        mon.was_killed.set()
                        _tqdm.tqdm.monitor = None
                except Exception:
                    pass
            except Exception:
                pass

            from ..bridge import elapid_bridge as eb
            from ..bridge.raster_bridge import check_raster_consistency

            # ── Pre-projection consistency check (fail-fast) ─────────────
            consistency = check_raster_consistency(self._raster_paths)
            if not consistency.get("is_consistent", False):
                mismatches = []
                if not consistency["crs_uniform"]:        mismatches.append("CRS")
                if not consistency["extent_uniform"]:     mismatches.append("extent")
                if not consistency["resolution_uniform"]: mismatches.append("resolution")
                raise RuntimeError(
                    f"Projection rasters do not share a common grid "
                    f"({', '.join(mismatches)} differ). Run \"Check "
                    f"Raster Consistency\" and \"Harmonize to Folder…\" "
                    f"in the ① Data tab to align them, then re-load "
                    f"the model and try again."
                )

            # ── Run the projection ───────────────────────────────────────
            # Exception to the bridge rule: progress reporting requires
            # monkey-patching elapid.geo.tqdm at the module level, which
            # by definition needs direct access to the elapid namespace.
            # Keep the rest of the elapid surface area in the bridge.
            import elapid.geo as elapid_geo

            _orig_tqdm = elapid_geo.tqdm
            worker_ref = self

            class _ProgressTqdm:
                """tqdm replacement that emits progress signals."""

                def __init__(self, iterable=None, total=None,
                             disable=False, **kw):
                    self._items = list(iterable) if iterable is not None else []
                    self._total = total or len(self._items)

                def __iter__(self):
                    for i, item in enumerate(self._items):
                        pct = int(((i + 1) / max(self._total, 1)) * 100)
                        worker_ref.progress.emit(
                            pct, f"Window {i+1}/{self._total}"
                        )
                        yield item

                def __enter__(self):  return self
                def __exit__(self, *a): pass
                def update(self, n=1): pass
                def close(self):       pass

            elapid_geo.tqdm = _ProgressTqdm
            try:
                eb.apply_model_to_rasters(
                    model=self._model,
                    raster_paths=self._raster_paths,
                    output_path=self._output_path,
                    quiet=False,
                )
            finally:
                elapid_geo.tqdm = _orig_tqdm

            self.finished.emit(True, "Done")
        except Exception as e:
            self.finished.emit(False, f"{e}\n{traceback.format_exc()}")
