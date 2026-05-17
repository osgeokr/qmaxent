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
3. Run the bench script (the maintainer keeps it under
   `tools/regenerate_pitta_golden_values.py`; it is excluded from
   the QGIS plugin ZIP). The script:
   - Fits the Default (β = 1, Geographic K-Fold) configuration.
   - Fits the Lee-matched (β = 4, Random K-Fold) configuration.
   - Records training AUC for both configurations.
   - Records the Spearman ρ between QMaxent and maxent.jar's
     permutation-importance rankings (the maxent.jar side is the
     static record from the manuscript's original benchmark).
   - Writes the new JSON.
4. Open the diff. Every changed value must be explainable — either
   by the code change in the same PR, or by an upstream dependency
   bump (in which case the dependency version goes in the same PR).
5. If the manuscript's § 3.3 prose or Table 3 cites a specific value
   that has now shifted, prepare a corresponding manuscript patch.

## What changes are allowed without manuscript revision

- Tolerance-band staying inside the `tolerance` field. If the new
  Δ is still `< 0.005`, the manuscript's qualitative claim is
  intact; only the JSON file changes.
- Spearman ρ at β = 1 staying inside the 0.05 band around 0.818
  (already encoded in the regression test).

## What changes require manuscript revision

- A new Δ for training AUC that exceeds 0.005 on either
  configuration. § 3.3's headline "|Δ| < 0.005" claim has moved.
- A Spearman ρ at β = 1 that drops below ~ 0.75. § 3.3 describes
  this number as "strong agreement"; a notably weaker ρ would
  change that descriptor.

In both cases the PR description must (a) explain why the change
is necessary, (b) propose a manuscript-text patch, and (c) note
which referenced figures (Fig. 4 panels) need redrawing.
