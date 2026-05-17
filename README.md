# QMaxent

[![CI](https://github.com/osgeokr/qmaxent/actions/workflows/ci.yml/badge.svg)](https://github.com/osgeokr/qmaxent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![QGIS plugin](https://img.shields.io/badge/QGIS-3.44%2B-green.svg)](https://plugins.qgis.org/plugins/qmaxent/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20259146.svg)](https://doi.org/10.5281/zenodo.20259146)

**Species Distribution Modeling with Maxent in QGIS.**

QMaxent integrates the [elapid](https://github.com/earth-chris/elapid) Python
library into QGIS to provide Maxent-style species distribution modelling
(SDM) directly inside the GIS desktop. Train a model on presence points and
environmental rasters, evaluate it with spatial cross-validation, inspect
variable importance via jackknife and permutation, project the model to a
habitat suitability map, and design field surveys with the integrated
priority-sites tool — all without leaving QGIS.

## Features

- **Modelling**: Maxent via the elapid library (linear, quadratic, hinge,
  product, threshold features; auto-selection by sample size following the
  documented maxnet rule).
- **Data preparation**: Check + Harmonize raster workflow (CRS, extent,
  resolution); one-click example datasets — Bradypus and Ariolimax.
- **Export for external Maxent**: Export QGIS-resident data to
  maxent.jar-compatible formats — either Samples-With-Data (SWD) CSV pairs
  (`presence.csv` + `background.csv`) or a samples CSV plus an ESRI ASCII
  Grid (.asc) folder. Generated `README.txt` contains the matching
  maxent.jar command line for cross-validation against the Java
  implementation.
- **Spatial evaluation**: Geographic K-Fold (Roberts et al. 2017,
  the default), Checkerboard (Muscarella et al. 2014, ENMeval),
  Random K-Fold, and Buffered Leave-One-Out with pooled AUC (Pearson
  et al. 2007; Ploton et al. 2020).
- **Variable importance**: Jackknife with both training and held-out test
  AUC, mirroring the original Maxent output (Phillips et al. 2006, 2017);
  permutation importance via `sklearn.inspection.permutation_importance`
  normalized to 100 % to match maxent.jar's permutation importance scale.
- **Spatial projection**: Apply trained models to raster stacks and load
  the result as a styled QGIS layer (`cloglog` / `logistic` / `raw`).
- **Priority Sites for Survey**: Discovery mode (random / top-N within a
  high-suitability band, with spacing constraints) and Validation mode
  (stratified four-quartile sampling above MTP / T10 / MaxSSS / custom
  threshold) for post-prediction field survey planning.
- **Model reuse**: Save trained models as `.pkl` and reload them later via
  a guided variable-mapping dialog that protects against the silent failure
  mode where mismatched raster ordering would yield wrong predictions.
- **Results export**: Multi-sheet styled XLSX (Times New Roman, academic
  supplementary-table convention) covering experimental setup, variable
  inventory, cross-validation, jackknife, permutation importance, and
  threshold tables — ready to attach to a manuscript supplement.
- **Bilingual UI**: English source strings with Korean translations.

## Requirements

- QGIS **3.44** or later.
- Python **3.10+** (QGIS 3.44 bundles Python **3.12** on Windows
  and macOS; Linux distribution installs vary but ship 3.10 or
  newer for QGIS 3.44).
- Internet access on first run to install dependencies into an isolated
  virtual environment (~590 MB after full install).

## Installation

1. In QGIS: **Plugins → Manage and Install Plugins → Settings**, ensure
   *"Show experimental plugins"* is unchecked.
2. Search for **QMaxent** and click **Install Plugin**.
3. After installation, open **Plugins → QMaxent → QMaxent Dependencies**
   and click **Install / Update Dependencies**. This step downloads
   `elapid`, `rasterio`, `geopandas`, `scikit-learn`, and other Python
   packages into a per-plugin virtual environment that does not affect
   your system Python or QGIS.

> The dependency-installation pattern is adapted from the
> [GeoAI](https://github.com/opengeos/geoai) QGIS plugin (Wu, 2026).

## Quick start

1. Open **Plugins → QMaxent → QMaxent Analysis**.
2. **① Data**: select a presence-point layer and add environmental rasters.
   Use **Check Raster Consistency** / **Harmonize to Folder…** to align
   grids if needed. Optionally **Export for external Maxent** to obtain a
   ready-to-run maxent.jar input directory.
3. **② Parameters**: choose feature types (Auto recommended), regularization
   strength, spatial evaluation method (Geographic K-Fold is the default),
   and output paths.
4. Click **▶ Run Maxent**. Progress is reported in the **③ Training** tab.
5. **④ Results**: inspect response curves, jackknife and permutation
   importance, and ROC. Set the output transform (`cloglog` recommended)
   and click **Run Spatial Projection** to write a habitat suitability
   raster.
6. **⑤ Priority Sites for Survey**: use the projected raster to draw
   Discovery or Validation candidate sites with the threshold and
   spacing rules described above.

To reload a previously saved `.pkl`, use the **Load existing model
(.pkl)…** button at the top of the **① Data** tab. The plugin will ask
you to map each model variable to a QGIS raster layer to ensure correct
covariate ordering at projection time.

> **Security note on `.pkl` files.** QMaxent saves trained models with
> Python's `pickle` format (via elapid's `save_object`), the same format
> used by scikit-learn and joblib. Pickle is convenient but it is a
> code-execution format: loading a malicious `.pkl` can execute
> arbitrary code on your machine. **Only load `.pkl` files that you (or
> a trusted collaborator) produced with QMaxent.** Treat unknown `.pkl`
> files with the same caution as unknown executable scripts.

## Documentation

- Official website (EN / KO): https://osgeokr.github.io/qmaxent/
- User manual (EN / KO): https://osgeokr.github.io/qmaxent/manual/

## Citation

If you use QMaxent in your research, please cite the software using
the metadata below.

**Software (CITATION.cff is the canonical source)**

> Yu, B.-H. (2026). *QMaxent: A QGIS plugin for Maxent species
> distribution modeling* (Version 0.1.7) [Computer software].
> Zenodo. https://doi.org/10.5281/zenodo.20259146

**BibTeX**

```bibtex
@software{Yu_QMaxent_2026,
  author  = {Yu, Byeong-Hyeok},
  title   = {{QMaxent: A QGIS plugin for Maxent species distribution modeling}},
  version = {0.1.7},
  date    = {2026-05-16},
  url     = {https://github.com/osgeokr/qmaxent},
  doi     = {10.5281/zenodo.20259146},
  license = {MIT}
}
```

The canonical source is [`CITATION.cff`](CITATION.cff). GitHub exposes
it via the "Cite this repository" sidebar widget, and the project
website renders the same metadata dynamically — update `CITATION.cff`
and every citation view follows.

## Methodological references

QMaxent's defaults follow established Maxent / SDM practice. Key references:

- Phillips, S. J., Anderson, R. P., & Schapire, R. E. (2006). Maximum
  entropy modeling of species geographic distributions. *Ecological
  Modelling*, 190, 231–259.
- Phillips, S. J., & Dudík, M. (2008). Modeling of species distributions
  with Maxent: new extensions and a comprehensive evaluation. *Ecography*,
  31, 161–175.
- Phillips, S. J., Anderson, R. P., Dudík, M., Schapire, R. E., & Blair,
  M. E. (2017). Opening the black box: an open-source release of Maxent.
  *Ecography*, 40, 887–893.
- Fithian, W., & Hastie, T. (2013). Finite-sample equivalence in
  statistical models for presence-only data. *Annals of Applied
  Statistics*, 7, 1917–1939.
- Anderson, C. B. (2023). elapid: Species distribution modeling tools for
  Python. *Journal of Open Source Software*, 8(84), 4930.
- Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J.,
  Guillera-Arroita, G., et al. (2017). Cross-validation strategies for data
  with temporal, spatial, hierarchical, or phylogenetic structure.
  *Ecography*, 40, 913–929.
- Veloz, S. D. (2009). Spatially autocorrelated sampling falsely inflates
  measures of accuracy for presence-only niche models. *Journal of
  Biogeography*, 36, 2290–2299.
- Wu, Q. (2026). GeoAI: a Python package for integrating artificial
  intelligence with geospatial data analysis and visualization. *Journal
  of Open Source Software*, 11(118), 9605.

A complete reference list with inline citations is provided in the source
code (see `workers/maxent_worker.py` and `bridge/elapid_bridge.py`).

## Funding

Development of QMaxent was supported by the **Satellite Information
Application** programme of the **Korea Aerospace Research Institute
(KARI)**, which also funded the Lee et al. (2025) field dataset used
for the maxent.jar numerical-compatibility benchmark in the
accompanying manuscript.

## Issues and contributions

- Bug reports and feature requests:
  [GitHub issue tracker](https://github.com/osgeokr/qmaxent/issues)
- Pull requests: see [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
  development workflow, test instructions, and code style.
- Discussion and questions:
  [GitHub Discussions](https://github.com/osgeokr/qmaxent/discussions)

All participants are expected to follow the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE) — Copyright © 2026 Byeong-Hyeok Yu.

Third-party library licenses bundled at runtime are listed in
[`