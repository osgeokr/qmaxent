__version__ = "0.1.7"


def classFactory(iface):
    # ── Pre-emptively disable tqdm's background monitor thread ───────
    # tqdm starts a daemon Thread (TMonitor) the first time tqdm.tqdm
    # is instantiated. On Windows + Python 3.12 + Qt threading we
    # observed that thread surviving past the QThread that triggered
    # it and later crashing QGIS with a native access violation
    # (`tqdm/_monitor.py: self.was_killed.wait(self.sleep_interval)`).
    # Setting monitor_interval = 0 *before* any tqdm.tqdm() call
    # tells tqdm to skip starting the thread — we lose only the
    # auxiliary "stalled bar refresher", not progress bars themselves.
    # Each worker also re-applies this guard in case classFactory ran
    # before tqdm was importable in the active venv.
    try:
        import tqdm as _tqdm

        _tqdm.tqdm.monitor_interval = 0
    except Exception:
        pass

    from .qmaxent_plugin import QMaxentPlugin

    return QMaxentPlugin(iface)
