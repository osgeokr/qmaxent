# Regenerating `pitta_golden_values.json`

The regression tier defends the manuscript's § 3.3 numerical claims
against the JSON file at `tests/fixtures/pitta_golden_values.json`.
That file is hand-curated rather than auto-generated, because the
underlying covariate rasters and presence points (Lee et al. 2025,
*GECCO* 60:e03939) are not redistributed with this repository.

Whenever a code change *can* shift the numerical output of the model
(elapid upgrade, sample-bias correction rewrite, permutation-importance
normalisation change), the fixture has to be regenerated locally
before the PR can land.

## What "regenerate" means in practice

1. Obtain the Lee et al. Pitta nympha dataset (47 presence points,
   10 covariates). The corresponding author of the manuscript has
   the canonical copy; contact bhyu@knps.or.kr.
2. Place the dataset under a folder outside the repository, e.g.
   `~/qmaxent-bench/pitta/`. Do NOT commit the rasters or
   presence-point CSVs to the repository.
3. Run the re-analysis scripts of the manuscript supplement
   (`reanalysis_v020/`; excluded from the plugin ZIP). They import
   the plugin's own `MaxentWorker` outside QGIS, so the numbers come
   from exactly the code users run:
   - `build_swd.py` draws the background exactly as QMaxent does
     (seeded) and writes identical SWD tables for maxent.jar.
   - `exp1_equivalence.py`: full-data fits on identical inputs at
     beta 1 and beta 4 (LQH); training AUC, prediction correlation,
     permutation-importance ranks (training data in both programs).
   - `exp2_lee_protocol.py`: Lee et al. (2025) final-model protocol
     (LQH, beta 4, ten identical 75/25 splits) and the rank agreement
     with Lee et al. Table 4.
   - `exp3_partition_beta.py`: beta {1, 4} x partition {Random
     10-fold, Geographic 5-fold}.
   Copy the resulting values into the JSON.
4. Open the diff. Every changed value must be explainable — either
   by the code change in the same PR, or by an upstream dependency
   bump (in which case the dependency version goes in the same PR).
5. If the manuscript's § 3.3 prose or Table 3 cites a specific value
   that has now shifted, prepare a corresponding manuscript patch.

## What changes are allowed without manuscript revision

- Training-AUC differences on identical inputs staying below 0.02,
  prediction correlation staying at r >= 0.85.
- Rank agreement staying above the recorded minima (rho >= 0.6 at
  beta 1, >= 0.8 at beta 4, >= 0.75 against Lee et al. Table 4).

## What changes require manuscript revision

- Any value crossing the thresholds above: § 3.3 describes the
  agreement as close / strong, and Table 3 and Fig. 4 report the
  values.

In both cases the PR description must (a) explain why the change
is necessary, (b) propose a manuscript-text patch, and (c) note
which referenced figures (Fig. 4 panels) need redrawing.

## History

- v0.2.0 (2026-10): fixture rebuilt from the SoftwareX re-analysis.
  The v0.1.7 record compared maxent.jar and QMaxent under different
  validation schemes, different background samples (the background
  was not seeded in v0.1.x) and different permutation-importance
  evaluation data, and listed covariates that Lee et al. did not use.
