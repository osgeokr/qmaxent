# QMaxent

**Species Distribution Modeling with Maxent in QGIS.**

QMaxent integrates the [elapid](https://github.com/earth-chris/elapid) Python
library into QGIS to provide Maxent-style species distribution modelling
(SDM) directly inside the GIS desktop. Train a model on presence points and
environmental rasters, evaluate it with spatial cross-validation, inspect
variable importance via jackknife, and project the model to a habitat
suitability map — all without leaving QGIS.

## Features

- **Modelling**: Maxent via the elapid library (linear, quadratic, hinge,
  product, threshold features; auto-selection by sample size following the
  documented maxnet rule).
- **Spatial evaluation**: Geographic K-Fold (Roberts et al. 2017),
  Checkerboard (Muscarella et al. 2014, ENMeval), and Buffered Leave-One-Out
  with pooled AUC (Pearson 2007; Ploton et al. 2020).
- **Variable importance**: Jackknife with both training and held-out test
  AUC, mirroring the original Maxent output (Phillips et al. 2006, 2017).
- **Spatial projection**: Apply trained models to raster stacks and load
  the result as a styled QGIS layer.
- **Model reuse**: Save trained models as `.pkl` and reload them later via
  a guided variable-mapping dialog that protects against the silent failure
  mode where mismatched raster ordering would yield wrong predictions.
- **Bilingual UI**: English source strings with Korean translations.

## Requirements

- QGIS 3.22 or later (LTR series tested).
- Python 3.9+ (bundled with recent QGIS releases).
- Internet access on first run to install dependencies into an isolated
  virtual environment (~300–500 MB).

## Installation

1. In QGIS: **Plugins → Manage and Install Plugins → Settings**, ensure
   *"Show experimental plugins"* is unchecked.
2. Search for **QMaxent** and click **Install Plugin**.
3. After installation, open **Plugins → QMaxent → QMaxent Dependencies**
   and click **Install / Update Dependencies**. This step downloads
   `elapid`, `rasterio`, `geopandas`, `scikit-learn`, and other Python
   packages into a per-plugin virtual environment that does not affect
   your system Python or QGIS.

## Quick start

1. Open **Plugins → QMaxent → QMaxent Analysis**.
2. **① Data**: select a presence-point layer and add environmental rasters.
3. **② Parameters**: choose feature types (Auto recommended), regularization
   strength, spatial evaluation method, and output paths.
4. Click **▶ Run Maxent**. Progress is reported in the **③ Training** tab.
5. **④ Results**: inspect response curves, jackknife importance, and ROC.
   Set the output transform (`cloglog` recommended) and click **Run Spatial
   Projection** to write a habitat suitability raster.

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

## Citation

If you use QMaxent in your research, please cite the software (a
SoftwareX paper is in preparation and will replace this citation once
published):

**APA 7th**

> Yu, B.-H. (2026). *QMaxent: a QGIS plugin for Maxent species
> distribution modeling* (Version 0.1.0) [Computer software].
> https://github.com/osgeokr/qmaxent

**BibTeX**

```bibtex
@software{Yu_QMaxent_2026,
  author  = {Yu, Byeong-Hyeok},
  title   = {{QMaxent: a QGIS plugin for Maxent species distribution modeling}},
  version = {0.1.0},
  date    = {2026-05-09},
  url     = {https://github.com/osgeokr/qmaxent},
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
- Anderson, C. B. (2023). elapid: Species distribution modeling tools for
  Python. *Journal of Open Source Software*, 8(84), 4930.

A complete reference list with inline citations is provided in the source
code (see `workers/maxent_worker.py` and `bridge/elapid_bridge.py`).

## License

[MIT](LICENSE) — Copyright © 2026 Byeong-Hyeok Yu.

## Author

**Byeong-Hyeok Yu** — bhyu@knps.or.kr

## Issues and contributions

Bug reports and feature requests are welcome at the
[issue tracker](https://github.com/osgeokr/qmaxent/issues).
