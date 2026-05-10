# Introduction

QMaxent is a [QGIS](https://qgis.org/) plugin that brings the full Maxent
species distribution modeling (SDM) workflow inside the GIS. From a single
docked panel you can choose presence points, register environmental
rasters, train a regularised Maxent model, evaluate it with spatial
cross-validation, project it across the landscape, and generate field-ready
candidate survey sites — without leaving QGIS or writing a line of code.

## What is QMaxent

Under the hood, QMaxent wraps the [elapid](https://github.com/earth-chris/elapid)
Python library (Anderson 2023). Elapid implements the maxnet algorithm — a
penalised Poisson formulation that is provably equivalent to the original
Maxent model on a sufficient sample of background points (Fithian & Hastie
2013) — using the lasso-regularised generalised linear model machinery of
glmnet (Friedman, Hastie & Tibshirani 2010). The user-facing model is the
same Maxent that the broader SDM literature has used since Phillips,
Anderson & Schapire (2006); the difference is the surrounding tooling, not
the inferential engine.

## Intended users

QMaxent is for ecologists, conservation planners, and biogeographers who:

- Already use QGIS for spatial analysis and want SDM as an integrated step
    rather than a separate Java GUI
- Need cross-validated, *spatially honest* performance estimates rather
    than an over-optimistic in-sample AUC (Roberts et al. 2017)
- Want their results to be **reproducible** — fixed seeds, configuration
    written to a multi-sheet Excel supplement, model serialised as a
    portable `.pkl`
- Work bilingually (Korean / English) and need a UI in their language

You do not need to know Python. You do need a working knowledge of vector
points and raster layers in QGIS, and enough domain knowledge of your
study system to choose appropriate environmental predictors.

## How it compares to other SDM tools

| Tool | Interface | Engine | Status |
|---|---|---|---|
| **QMaxent** | QGIS plugin | elapid (maxnet, Python) | Active, this manual |
| Java MaxEnt | Standalone Java GUI | Original Maxent | Stable; broadly used |
| ENMeval (R) | R package | maxnet / dismo | Strong CV / tuning support (Muscarella et al. 2014) |
| Wallace | Shiny web app | Multiple (incl. maxnet) | Reproducibility focus (Kass et al. 2018) |
| dismo (R) | R package | Multiple SDM algorithms | Mature; broad ecosystem |

QMaxent occupies the same conceptual niche as the Java MaxEnt GUI for
QGIS users, with the added benefit of a modern Python runtime that ships
with QGIS, an integrated raster-consistency preflight, and built-in
spatial cross-validation. The [Pitta nympha worked example](examples/pitta-nympha.md)
reproduces a published Java MaxEnt analysis in QMaxent and discusses
where the two pipelines agree or diverge.

## Methodological lineage

QMaxent's defaults follow the recommendations of:

- Phillips, Anderson & Schapire (2006), Phillips & Dudík (2008), and
    Phillips et al. (2017) for the core Maxent formulation, regularization,
    and cloglog output transform.
- Elith et al. (2011) and Merow, Smith & Silander (2013) for practitioner
    guidance on inputs and settings.
- Radosavljevic & Anderson (2014) for the auto-rule that selects feature
    classes by sample size.
- Roberts et al. (2017) for spatial cross-validation as the default
    evaluation method.

Every default in the plugin can be overridden in the
[② Parameters tab](parameters-tab.md). Full citations are in
[References](references.md).
