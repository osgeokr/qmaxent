# ④ Results tab

After a successful run, the Results tab unlocks three sub-tabs:

| Sub-tab | What it shows |
|---|---|
| **Response Curves** | How each variable shapes predicted suitability across its data range |
| **Jackknife Importance** | Which variables matter most, with ROC plotted alongside |
| **Spatial Projection** | Write the prediction to a GeoTIFF and load it as a styled QGIS layer |

## Response Curves

Pick a variable from the drop-down at the top. QMaxent draws the predicted
cloglog suitability across the variable's actual training range (shaded
band) with the mean as a vertical reference line:

![Response curve for bio1 showing predicted cloglog suitability across the training range](images/ui/dock-4-response-curve-bio1.png)

How to read a response curve:

- **Y-axis** is the cloglog probability of suitability (0–1).
- **X-axis** is the variable's value in its native units. The shaded
    "Training range" band marks the data range the model actually saw;
    extending the curve outside that band means extrapolation, which should
    be interpreted cautiously (see the discussion in
    [Methodological background](methodological-background.md)).
- **Sharp peaks or steps** are usually hinge or threshold features at work.
    Smooth curves come from linear and quadratic features.
- **Per-variable interpretation**: the curve marginalizes over all other
    variables, so it shows the partial response *holding the others at their
    mean*. Use it for ecological interpretation, not for prediction.

Switch variables freely from the drop-down — the plot redraws each time
without re-running the model.

## Jackknife Importance

The Jackknife sub-tab is the most information-dense view in the plugin: it
combines the ROC analysis (training and per-fold CV) with the variable
importance bars in a single side-by-side figure:

![ROC curve plus Jackknife variable importance for the 9 Bradypus variables](images/ui/dock-4-jackknife-bars.png)

### ROC panel (left)

- **Training ROC** (solid line): the in-sample fit. Always optimistic.
- **Mean CV ROC** (dashed): the average across all CV folds. This is the
    headline performance estimate for the model.
- **Per-fold ROC** (faint lines): each spatial CV fold's ROC. The spread
    visualizes how stable the model is across spatial subsamples.
- **Random** (diagonal): the no-skill baseline.

The Bradypus example shows training AUC of 0.956 with mean CV AUC of 0.758 —
a typical, healthy gap that indicates the model has learned a real signal
without being grossly over-fitted.

### Jackknife panel (right)

For each variable, two bars:

- **With only variable** (dark): how well a model with *just this variable*
    predicts. High = the variable carries strong univariate signal.
- **Without variable** (light): how well the model performs *without it*.
    Low (compared to the full-model AUC) = the variable carries unique
    information not redundant with others.

A variable that **scores high on dark and creates a noticeable drop on
light** is unambiguously important. A variable that scores high on dark
but barely changes light is informative but redundant with others (likely
correlated). The test/train split for each bar is shown in the same view —
the value labels are the held-out test AUCs from the CV folds.

## Spatial Projection

This sub-tab applies the trained model to the entire raster stack to
produce a continuous habitat-suitability surface.

![Spatial Projection sub-tab after running, with the output GeoTIFF path reported as Done](images/ui/dock-4-projection-done.png)

### Output transforms

| Transform | Range | Interpretation |
|---|---|---|
| **cloglog** *(default)* | 0–1 | Probability of presence at average prevalence (Phillips et al. 2017) |
| **logistic** | 0–1 | Older logistic transform; included for backward compatibility |
| **raw** | unbounded | Relative occurrence rate; sums to 1 over the study area |

`cloglog` is the recommended default for almost all studies — it has a
proper probabilistic interpretation, scales linearly with relative occurrence
rate, and is what the modern Maxent literature reports.

### Auto-load result as QGIS layer

Leave this on (the default) and the resulting GeoTIFF is added to your
QGIS project the moment it is written, with an auto-applied continuous
white-to-green color ramp. For Bradypus the result is:

![Bradypus habitat-suitability map across Latin America, white=low, green=high](images/maps/quickstart-final-suitability.png)

You can override the color ramp afterward via the layer's **Symbology**
panel; QMaxent's auto-styling is just a sensible default.

### Save analysis charts as PNG

When checked, QMaxent also writes high-resolution PNG copies of the
Response Curves, ROC, and Jackknife plots next to the GeoTIFF. These are
publication-ready and match the figures you see in this tab. They are
sized for direct paste into a manuscript figure (300 dpi).

## Memory and performance

Spatial projection reads each raster cell-by-cell and applies the model.
For very large rasters (continental scale at 30 m resolution), expect the
projection step to dominate runtime. Two tips:

- **Pre-tile the rasters** to your true study extent before projection; use
    QGIS's `Clip raster by extent` or `Warp` algorithms.
- **Coarsen** if appropriate — a 30 m climate raster is rarely meaningful;
    250 m–1 km is usually fine for SDM and runs much faster.
