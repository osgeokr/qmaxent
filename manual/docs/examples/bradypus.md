# Bradypus variegatus

The brown-throated three-toed sloth — *Bradypus variegatus* — is the
canonical Maxent test dataset, originally published with
[Phillips, Anderson & Schapire (2006)](../references.md#maxent-core-methodology)
and reused in virtually every subsequent Maxent paper. We use it here as a
**guided tour of every QMaxent feature**: data loading, parameter selection,
spatial cross-validation, jackknife importance, projection, and survey
planning. By the end of this chapter you will have produced — and be able
to defend — a complete Bradypus habitat-suitability model.

## Dataset

The Phillips et al. (2006) dataset contains:

| Layer | Type | Description |
|---|---|---|
| `bradypus.shp` | Vector points | 116 occurrence records across South and Central America |
| `bio1, bio5, bio6, bio7, bio8` | Continuous raster | Temperature variables (WorldClim) |
| `bio12, bio16, bio17` | Continuous raster | Precipitation variables (WorldClim) |
| `biome` | Categorical raster | Biome type (Olson et al. 2001) |

All rasters share the same grid: EPSG:4326, 0.5° × 0.5° cells, covering
the Americas. Total dataset size is well under 100 MB.

Download via **Plugins → QMaxent → Download Example Dataset → Bradypus
variegatus**. Layers are added to the QGIS project automatically:

![Bradypus presence points overlaid on bio17 across Latin America](../images/maps/example-bradypus-loaded.png)

## Loading data into the Analysis dock

Open **Plugins → QMaxent → QMaxent Analysis**. On the **① Data** tab, choose
`bradypus` from the **Presence Points Layer** drop-down — QMaxent reports
`116 presence points loaded` immediately.

Click **Add from project** to add all the loaded raster layers in one step,
then mark `biome` as `[categorical]`. Click **Check Raster Consistency** to
verify the grid:

![Data tab with Bradypus presence layer, 7 environmental rasters, biome marked categorical, and Check Raster Consistency: All 9 rasters share grid](../images/ui/dock-1-data-with-bradypus.png)

The status line reports
`✓ All 9 rasters share grid (CRS: EPSG:4326, resolution: 0.5 × 0.5)` —
exactly what we expect from the bundled dataset.

## Configuring the model

Switch to **② Parameters**. For this tour we accept every default:

- **Feature Types**: Auto (with 116 presences, the maxnet rule selects all
    of LQPHT)
- **Regularization multiplier**: 1.0
- **Spatial evaluation**: Geographic K-Fold, 5 folds, fixed seed = 0
- **Jackknife variable importance**: enabled
- **Output Files**: `qmaxent_output/model.pkl` and `qmaxent_output/results.xlsx`

The fixed random seed means **anyone re-running this tutorial gets bitwise
identical results** — important for reproducibility.

## Running training and cross-validation

Click **▶ Run Maxent**. The Training tab takes over and the run completes
in ~30 seconds:

![Training tab at 100% with full log](../images/ui/dock-3-training-completed.png)

Reading the log top-down tells the entire story:

```text
→ 10,000 background points sampled
Extracting raster covariates for presence points…
Extracting raster covariates for background points…
→ Presence: 116, Background: 9,997
→ Feature types: ['linear', 'quadratic', 'product', 'hinge', 'threshold']
Training MaxentModel…
→ Model training complete
→ Model saved: …/model.pkl
Computing ROC curve…
→ Training AUC = 0.9562
Running cross-validation…
  Fold 1: 22 test presences, AUC = 0.7453
  Fold 2: 21 test presences, AUC = 0.7839
  Fold 3: 39 test presences, AUC = 0.8097
  Fold 4: 26 test presences, AUC = 0.8614
  Fold 5: 8 test presences, AUC = 0.5903
→ CV AUC = 0.7581 ± 0.0920  (n=5 fold(s))
```

**Train AUC = 0.956** while **CV AUC = 0.758 ± 0.092**. The gap is the cost
of holding out spatially distinct test sets — a more honest measure of
real-world predictive performance than the inflated training AUC alone.

Fold 5 has the lowest AUC (0.59) and the smallest test set (8 presences):
in spatial CV the folds are deliberately uneven by area, so a fold can land
on a small, atypical pocket. The pooled CV AUC averages out this variation.

## Inspecting variable behaviour

### Response curves

On **④ Results → Response Curves**, pick `bio1` (annual mean temperature):

![Response curve for bio1 with training range shaded](../images/ui/dock-4-response-curve-bio1.png)

The response is non-monotonic, with peaks around 150 and 290 (units of 0.1 °C
following WorldClim convention) and a depression near 240. The model used
hinge features to capture these breaks. The shaded "Training range" band
covers the values actually present in the dataset; the prediction near
−50 (extreme cold) and beyond 320 should be regarded as pure extrapolation.

Try other variables from the drop-down to see which features Maxent
recruited for each. Smooth U- or hump-shaped curves suggest quadratic
terms; sharp angled breaks come from hinge or threshold features.

### Jackknife importance and ROC

The **Jackknife Importance** sub-tab combines the ROC and the per-variable
bars in a single figure:

![ROC plus Jackknife panel for the 9 Bradypus variables](../images/ui/dock-4-jackknife-bars.png)

ROC reading:

- **Training ROC** (solid, AUC 0.956): the in-sample fit
- **Mean CV ROC** (dashed, AUC 0.758): five spatial folds averaged
- **Per-fold ROC** (faint): the spread is the model's stability across
    spatial subsamples

Jackknife reading:

The dark bars (model with **only this variable**) and light bars (model
**without this variable**) tell you each variable's *unique* contribution.
For Bradypus:

- **`biome`** (categorical) carries the strongest univariate signal (AUC ≈ 0.78),
    and the model loses meaningfully when it is removed — biome boundaries
    map closely onto sloth distribution.
- **`bio7`** (temperature annual range) comes second.
- **`bio1`** is informative on its own but redundant with several others
    (small drop when removed).
- **`bio5`** has the lowest stand-alone signal (~0.54) — close to random.

This is exactly how the original Phillips et al. (2006) paper used the
jackknife to argue that biome and seasonality jointly carry the most
information for Bradypus.

## Spatial projection

Switch to **Spatial Projection** in the same Results tab. With **cloglog**
output and **Auto-load result as QGIS layer** ticked, click
**▶ Run Spatial Projection**:

![Projection sub-tab after running, output GeoTIFF path reported as Done](../images/ui/dock-4-projection-done.png)

The map appears in QGIS, auto-styled white-to-green:

![Bradypus habitat-suitability map across Latin America](../images/maps/quickstart-final-suitability.png)

The high-suitability core covers the Atlantic Forest of southeast Brazil
and the Amazon basin — both well known sloth strongholds — with secondary
patches across Central America. The model correctly identifies the
unsuitability of the Andes (cold, high elevation) and the very dry
northeastern Brazil (caatinga).

## Saving outputs

Two files were produced automatically:

- `qmaxent_output/model.pkl` — the serialized trained model. Reload it
    later from the **Load existing model (.pkl)…** button on the Data tab,
    or share it with collaborators. See [Saving and reusing models](../saving-models.md)
    for the security considerations.
- `qmaxent_output/results.xlsx` — a multi-sheet Excel supplement with
    experimental setup, variable inventory, cross-validation, jackknife,
    and response-curve breakpoints. See [Exporting results](../exporting-results.md)
    for the sheet-by-sheet description.

If you ticked **Save analysis charts as PNG** before running projection,
you also have publication-ready 300-dpi PNGs of the response curves, ROC,
and jackknife panels next to the GeoTIFF.

## Optional: priority sites for survey

A natural next step is to use the trained model to plan a follow-up
survey. Switch to **⑤ Priority Sites for Survey**, choose **Discovery**
mode, leave the auto-set Minimum suitability at 0.9, and click
**▶ Extract Priority Sites**:

![Priority Sites tab in Discovery mode after extraction](../images/ui/dock-5-priority-sites-extracted.png)

Twenty candidate locations (red dots) appear on top of the suitability map,
with addresses populated by Nominatim reverse geocoding:

![Priority sites overlaid as red dots on the Bradypus suitability map](../images/maps/priority-sites-on-canvas.png)

Each candidate is at least 1 km from any known presence and at least 500 m
from every other candidate, so a single field trip can plausibly cover
several of them.

## Where to go next

- **Same workflow with messy rasters**: the [Ariolimax example](ariolimax.md)
    starts from rasters that don't share a CRS or resolution, exercising the
    Check + Harmonize tools.
- **Same workflow against a published study**: the [Pitta nympha example](pitta-nympha.md)
    reproduces a published Java MaxEnt analysis in QMaxent and discusses
    where the two pipelines agree or diverge.
- **Deeper theory**: [Methodological background](../methodological-background.md)
    explains why each default we accepted in this tour is the right choice.
