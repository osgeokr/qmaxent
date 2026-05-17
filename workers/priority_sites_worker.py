"""Background worker for the Priority Sites for Survey feature.

Runs threshold computation and raster sampling off the GUI thread so
the dock stays responsive. Reverse geocoding is **NOT** done here —
it happens on the main thread via QTimer-driven steps because doing
HTTP calls from a QThread on Windows + Python 3.12 produced a
reproducible native access-violation crash inside socket/SSL cleanup
(seen at the 19/20 mark of the geocoding loop).

Sampling is CPU-bound and finishes in seconds, so a worker is
perfectly safe for it. Geocoding is network-bound at 1 req/sec and
can be paced by Qt's own event loop without blocking the GUI.
"""

from qgis.PyQt.QtCore import QThread, pyqtSignal


class PrioritySitesWorker(QThread):
    """Run priority-site sampling (no geocoding).

    Signals:
        progress(int, str): 0–100% plus a status message.
        log(str):           verbose line for the training log.
        finished(bool, str, list): success flag, message, and the
            list of sampled sites. Each site is a dict with keys::
              {"lat", "lon", "suitability"}
            Address fields are added later on the main thread.
    """

    progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str, list)

    def __init__(
        self,
        prediction_path: str,
        presence_xy: list,
        threshold: float,
        threshold_method: str,
        n_sites: int,
        min_dist_from_presence_m: float,
        min_dist_between_sites_m: float,
        stratify_by_quartile: bool,
        do_geocode: bool,
        sampling_order: str = "topn",
        random_seed=42,
        parent=None,
    ):
        super().__init__(parent)
        self._prediction_path = prediction_path
        self._presence_xy = list(presence_xy)
        self._threshold = float(threshold)
        self._threshold_method = threshold_method
        self._n_sites = int(n_sites)
        self._min_d_pres = float(min_dist_from_presence_m)
        self._min_d_site = float(min_dist_between_sites_m)
        self._stratify = bool(stratify_by_quartile)
        # "random" or "topn" — only meaningful when stratify is False
        # (Discovery mode). Validation mode always uses topn within
        # each quartile.
        self._sampling_order = str(sampling_order)
        # do_geocode is no longer used by the worker — it's read by
        # the main-thread caller after the worker finishes — but we
        # still accept it in the constructor so the calling code in
        # main_dock.py doesn't have to change shape.
        self._do_geocode = bool(do_geocode)
        # None when the user unchecked "Fix random seed" upstream —
        # numpy's default_rng(None) seeds from OS entropy, which is
        # exactly the behavior the unchecked state promises.
        self._random_seed = int(random_seed) if random_seed is not None else None
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            # See projection_worker.py / maxent_worker.py for the full
            # rationale. We re-apply the tqdm monitor-thread kill here
            # too as a precaution, even though sampling itself doesn't
            # use tqdm.
            try:
                import tqdm as _tqdm

                _tqdm.tqdm.monitor_interval = 0
                try:
                    mon = getattr(_tqdm.tqdm, "monitor", None)
                    if mon is not None and hasattr(mon, "was_killed"):
                        mon.was_killed.set()
                        _tqdm.tqdm.monitor = None
                except Exception:
                    pass
            except Exception:
                pass

            self.progress.emit(0, "Extracting priority sites…")
            from ..bridge.priority_sites import extract_priority_sites

            sites = extract_priority_sites(
                prediction_path=self._prediction_path,
                presence_xy=self._presence_xy,
                threshold=self._threshold,
                n_sites=self._n_sites,
                min_distance_from_presence_m=self._min_d_pres,
                min_distance_between_sites_m=self._min_d_site,
                stratify_by_quartile=self._stratify,
                sampling_order=self._sampling_order,
                random_seed=self._random_seed,
            )
            self.log.emit(
                f"Sampling complete — {len(sites)}/{self._n_sites} "
                f"sites accepted (threshold method: {self._threshold_method}, "
                f"value: {self._threshold:.4f})"
            )
            if self._cancel:
                self.finished.emit(False, "Cancelled", [])
                return
            if not sites:
                self.finished.emit(
                    False,
                    "No suitable cells satisfied the spacing constraints. "
                    "Try a lower threshold or smaller min-distance values.",
                    [],
                )
                return

            # Add empty address fields so the row schema is consistent
            # with what the main-thread geocoding step will produce.
            results = [
                {
                    **s,
                    "country": "",
                    "province": "",
                    "city_county": "",
                    "district": "",
                    "display_name": "",
                }
                for s in sites
            ]
            self.progress.emit(100, "Sampling done.")
            self.finished.emit(True, "Done", results)

        except Exception as e:
            import traceback

            self.finished.emit(False, f"{e}\n{traceback.format_exc()}", [])
