# Methodological background

This chapter collects the *why* behind the *how* — the academic reasoning
that justifies QMaxent's defaults. It is intended for users who want to
understand or cite the model choices, not for first-time users (who should
read the User guide chapters first).

## Maxent and the principle of maximum entropy

Maxent fits the **maximum-entropy distribution** consistent with the
empirical moments observed at the presence locations
([Phillips, Anderson & Schapire 2006](references.md)). Intuitively: among
all distributions over the study area that produce the same average value
of each environmental variable as the presence sample, the maximum-
entropy distribution is the *least informative* — the one that adds no
assumptions beyond what the data actually says.

The estimator is mathematically equivalent to a regularized inhomogeneous
Poisson point-process model fit to the presence-background contrast
([Fithian & Hastie 2013](references.md)) — and that equivalence is what
elapid's implementation exploits to delegate the optimisation to
scikit-learn's logistic-regression routines
([Anderson 2023](references.md);
[Pedregosa et al. 2011](references.md)).

## The maxnet default rule for feature selection

The Maxent feature classes (Linear, Quadratic, Product, Hinge, Threshold)
trade off **flexibility** against **over-fit risk**. maxent.jar and its
R counterpart maxnet both choose default feature classes from the number
of presence points. QMaxent's **Auto (maxnet default rule by sample size)**
mode applies the maxnet default
([Phillips et al. 2017](references.md);
`maxnet.formula(classes = "default")`):

| Presences | Feature classes |
|---|---|
| fewer than 10 | L |
| 10 to 14 | L, Q |
| 15 to 79 | L, Q, H |
| 80 or more | L, Q, P, H |

Unlike the maxent.jar rule, which QMaxent used up to v0.1.x, the maxnet
default never adds threshold features. This matters for QMaxent's
engine: elapid fits the model with scikit-learn, which applies one
regularization strength to all features, whereas maxent.jar regularizes
each feature class separately. Under a single penalty the many
step-shaped threshold features overfit. On the Bradypus benchmark
(random 10-fold CV), the test AUC was 0.873 with LQPHT and 0.890 with
LQPH, against 0.895 for maxent.jar with LQPHT. Threshold features are
still available under **Manual selection**.

The rule is a default, not a guarantee of the best model. The most
suitable feature classes and regularization depend on the data
([Merow, Smith & Silander 2013](references.md);
[Radosavljevic & Anderson 2014](references.md)). For a publication,
compare alternatives with **Manual selection** and spatial
cross-validation, or with a model-selection tool such as ENMeval
([Muscarella et al. 2014](references.md)).

## Why spatial cross-validation matters

Random K-Fold CV gives over-optimistic AUCs on spatial data because
random folds are likely to contain points geographically *adjacent to*
training points. Spatial autocorrelation makes the held-out points
trivially predictable.

[Roberts et al. 2017](references.md) demonstrate this empirically across
many SDM benchmarks: random-K-Fold AUCs systematically over-estimate the
performance you would actually see on a new region by 0.05–0.15. They
recommend spatially-blocked CV as the default for any spatial model.

QMaxent therefore defaults to **Geographic K-Fold (k = 5, fixed seed)**
— the recommended Roberts et al. design. *Random* K-Fold is offered only
to allow exact reproduction of older studies.

## Pooled vs per-fold AUC

Two summary statistics across folds are commonly reported:

- **Per-fold AUC then averaged** — compute AUC inside each fold,
  average across folds, report mean ± SD.
- **Pooled AUC** — concatenate all held-out predictions across folds
  and compute a single AUC.

QMaxent reports the **per-fold mean ± SD** as the headline number
because it carries the across-fold variance, which is itself an
informative diagnostic — high variance signals that the model's
geographic transferability is unstable. The pooled-AUC alternative is
included in Table 3 of the XLSX as a secondary metric.

## Jackknife variable importance

The classical Maxent jackknife
([Phillips, Anderson & Schapire 2006](references.md)) fits, for each
variable, two extra models: one with that variable alone (*with-only*)
and one with that variable removed (*without*). The drop in test AUC
from full to *without* identifies variables whose unique contribution
is hard to recover from the others.

QMaxent reports four AUCs per variable (only-train, only-test,
without-train, without-test) plus the test-side drop column, sorted by
descending drop so the most uniquely-informative variables are at the
top. This is the format used in the published Maxent literature since
Phillips et al. 2006.

## Permutation importance and its evaluation data

Permutation importance shuffles one variable at a time and measures the
drop in AUC. The result depends on the data the AUC is computed on, so
QMaxent lets you choose it (**Evaluate on** on the ② Parameters tab):

- **Training data (maxent.jar convention)** (default): the full model on
  all training presences and background points. This is how maxent.jar
  computes permutation importance, so rankings can be compared directly
  with maxent.jar studies.
- **Held-out CV test folds (all folds pooled)**: each fold model on its
  own test fold, with the results of all folds pooled. This measures how
  much predictive performance at new locations depends on each variable.
  It needs K-Fold or Checkerboard cross-validation.

Up to v0.1.x QMaxent used the test data of the first CV fold only,
which made rankings unstable and not comparable with maxent.jar.

## Threshold methods (MTP / T10 / MaxSSS)

When you need a binary suitable/unsuitable map (e.g. for reserve-design
applications), you must choose a threshold of the cloglog output. The
SDM literature standardises on three options
([Liu, Newell & White 2013](references.md)):

- **MTP — Minimum Training Presence** — the lowest cloglog value at
  any training presence. Most permissive; favours sensitivity.
- **T10 — 10th percentile training presence** — drops the bottom 10%
  of training presences as outliers. The classical Java-MaxEnt default.
- **MaxSSS — Max Sensitivity + Specificity** — the threshold that
  maximises the sum of sensitivity and specificity on the training
  ROC. Best calibration in most published comparisons.

QMaxent reports all three thresholds in Table 3 of the XLSX export.
Choose **MaxSSS** unless you are reproducing a Java-MaxEnt study (T10)
or have a specific reason to prefer permissive (MTP) or strict
classification.

## Output transforms compared

Three output transforms are available on **Spatial Projection**:

| Transform | Range | Interpretation |
|---|---|---|
| **cloglog** | 0–1 | Probability of presence given a typical sample, [Phillips et al. 2017](references.md) recommended default |
| **logistic** | 0–1 | Older, [Phillips & Dudík 2008](references.md) parameterisation; still used in some baselines |
| **raw** | unbounded | Unnormalised exponential output; advanced post-processing |

QMaxent defaults to **cloglog** because it is the form
[Phillips et al. 2017](references.md) explicitly recommend as the
modern default and because its "probability of presence" interpretation
is the most directly communicable to non-modelling stakeholders.

## Treatment of categorical variables

Java MaxEnt encodes categorical rasters via the raster's attribute
table; elapid encodes them via **one-hot expansion** internally
([Anderson 2023](references.md)). The two approaches are equivalent for
training but diverge at projection time when the raster contains class
codes the model has not seen during training:

- **Java MaxEnt** silently maps unseen codes to a random class,
  producing arbitrary suitability values at affected cells.
- **QMaxent** detects unseen codes via the unified preflight dialog
  and **auto-masks them to NoData** rather than extrapolating. If an
  unseen class still reaches the model, the projection does not fail
  (since v0.2.0): those cells get no categorical contribution, as in
  maxent.jar.

This is the recommended practice
([Elith, Kearney & Phillips 2010](references.md)) — extrapolation
beyond the training domain is fundamentally undefined and should be
flagged, not silently filled in.

## Bias correction

When occurrences are spatially clustered (e.g. road-bias, citizen-
science clusters), the model fits the *bias surface* as well as the
species' habitat preference. Two complementary corrections exist:

- **Sample-weight down-weighting** ([Phillips et al. 2009](references.md))
  — applies fractional weights to clustered presences. QMaxent
  implements this as the *Down-weight spatially clustered points* option.
- **External KDE bias raster** — a raster representing the survey
  effort, used as an explicit covariate. Not implemented in QMaxent
  v0.1.x; the closest analog is the down-weight option, which performs
  comparably for many published studies.

For a more comprehensive treatment, [Boria et al. 2014](references.md)
recommend **spatial thinning** of the presence layer before modeling
as a complementary step — implementable in QGIS via the *NNJoin*
plugin or the standalone *spThin* tool.

## Why these choices

The defaults assembled in QMaxent are the union of (a) what the
benchmark literature shows works well across many species and study
areas, and (b) what minimises the silent-failure modes documented in
[Roberts et al. 2017](references.md), [Araújo et al. 2019](references.md),
and [Elith, Kearney & Phillips 2010](references.md). Diverging from
them is sometimes the right move for a specific study; the
[Pitta nympha worked example](examples/pitta-nympha.md) shows what that
looks like for the LQH/RM=4 fixed-hyperparameter setting of
[Lee et al. 2025](references.md).
