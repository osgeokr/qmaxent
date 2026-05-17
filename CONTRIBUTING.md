# Contributing to QMaxent

Thanks for your interest. QMaxent is open to contributions from
species-distribution-modelling practitioners, conservation
biologists, and Python / QGIS developers alike.

## Quick links

- **Bug reports** → [issue tracker](https://github.com/osgeokr/qmaxent/issues)
- **Questions / discussion** → [GitHub Discussions](https://github.com/osgeokr/qmaxent/discussions)
- **Code of Conduct** → [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **User manual (EN / KO)** → https://osgeokr.github.io/qmaxent/manual/

## How development is organised

QMaxent ships as a QGIS plugin and is mirrored on PyPI-style tooling
*only* for testing. Practical consequences:

- The source tree at the repository root **is** the plugin source
  tree. Files there are installed verbatim into
  `<QGIS profile>/python/plugins/qmaxent/`.
- Runtime dependencies are NOT declared in `pyproject.toml`. They are
  managed at first launch by `core/venv_manager.py`, which builds a
  plugin-private virtual environment under `~/.qgis_qmaxent/`.
- The companion SoftwareX manuscript has specific numerical claims
  (§ 3.3) that are pinned in `tests/fixtures/pitta_golden_values.json`
  and asserted by the regression test tier. Any change that can move
  these numbers needs to update the fixture and the manuscript
  together.

## Setting up a dev environment

```bash
git clone https://github.com/osgeokr/qmaxent.git
cd qmaxent

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements-test.txt
```

To actually run the plugin inside QGIS during development, the
fastest pattern is to symlink the repo into the QGIS plugins folder:

```bash
# Linux example — adjust paths for macOS / Windows.
ln -s "$PWD" "$HOME/.local/share/QGIS/QGIS3/profiles/default/python/plugins/qmaxent"
```

Then enable the plugin in **QGIS → Plugins → Manage and Install
Plugins**, and use the
[Plugin Reloader](https://plugins.qgis.org/plugins/plugin_reloader/)
to pick up code changes without restarting QGIS.

## Running tests

```bash
# Pure-Python unit tests — fast, no QGIS, no Java.
pytest tests/unit -q

# Regression tests — defends the § 3.3 manuscript claims.
# Requires elapid + rasterio + scikit-learn.
pytest tests/regression -q

# Everything plus a coverage report.
pytest --cov --cov-report=term-missing
```

The CI matrix (`.github/workflows/ci.yml`) runs unit tests on
Ubuntu × Python 3.10 / 3.11 / 3.12, Windows × 3.12, macOS × 3.12,
and the regression tier on Ubuntu × 3.12.

## Coding conventions

- **Style**: enforced by `ruff` (configuration in `pyproject.toml`).
  Run `ruff check . && ruff format .` before committing.
- **Imports**: standard-library first, third-party next, local last;
  `ruff` reorders these automatically.
- **Docstrings**: every public function has a short module-level
  comment explaining *why* it exists. We prefer "why" over "what" —
  the code already tells you what it does.
- **Comments at decision points**: when the code chooses one approach
  over another (e.g. why `sys.prefix` is used instead of
  `sys.executable` on Windows), add a comment that future
  contributors can read in 30 seconds. The existing code in
  `core/venv_manager.py` is the model.
- **Logging**: use `QgsMessageLog.logMessage(..., "QMaxent", level=...)`
  rather than `print` so the message lands in QGIS's Log Messages
  panel under a `QMaxent` channel. The plugin-wide log helper is in
  `core/venv_manager.py:_log`.
- **i18n**: user-visible strings should go through
  `i18n.translator.tr()` so the Korean translation in
  `i18n/lang_ko.py` can keep pace.

## Numerical changes (special rules)

A PR can change numerical output in two main ways:

1. **A backend behaviour change** — for example, replacing the
   sample-bias correction in `bridge/elapid_bridge.py`, or
   normalising permutation importance differently.
2. **A dependency upgrade** that shifts the IWLR fit — for example,
   bumping the elapid minimum past a release that changed its
   convergence tolerance.

In either case:

- Regenerate `tests/fixtures/pitta_golden_values.json` by running
  the script described in `docs/golden-values.md`.
- Commit the regenerated fixture **in the same PR** as the code
  change. The diff is reviewable.
- If the manuscript text needs to change as a result, prepare a
  patch to the corresponding section and link it in the PR body.

PRs that change numerical output without touching the fixture are
caught by the regression tier in CI, which is the safety net — but
the goal is to make the intent visible in a single coordinated
commit.

## Release checklist

When tagging a release (maintainer-only):

1. Bump version in **all three** of `__init__.py`, `metadata.txt`,
   `CITATION.cff`. The `test_version_consistency` unit test will
   catch drift.
2. Update `tests/fixtures/pitta_golden_values.json` if any backend
   change in the release window could have shifted numerical output
   (`_provenance.qmaxent_version` must equal the new tag).
3. Run `pytest -q` locally. Push to a branch and ensure the full
   matrix is green.
4. Tag the release: `git tag vX.Y.Z && git push --tags`.
   `release-zenodo.yml` verifies tag↔file version agreement.
5. Publish the GitHub Release. Zenodo auto-archives via the GitHub
   integration; the issued concept + version DOIs appear in your
   Zenodo dashboard within minutes.
6. Edit `CITATION.cff` and `README.md` to replace the DOI
   placeholders with the issued values. Commit on the **post-release
   patch branch** (do NOT re-tag the existing release).

See `docs/zenodo-release.md` for the full release-day workflow.
