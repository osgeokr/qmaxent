# ① Data tab

The Data tab is where you tell QMaxent **which species** you are modeling and
**which environmental variables** to use. Three controls deserve attention:
the presence-points selector, the environmental-raster list with its
**Check Raster Consistency** action, and the background-points sample size.

![Data tab in its initial empty state](images/ui/dock-1-data-empty.png)

## Presence Points Layer

The first drop-down lists every loaded vector point layer in the current QGIS
project. Pick the layer that contains your species occurrence records. As
soon as you select it, QMaxent reads the geometry and reports the number of
points underneath the selector (e.g. `116 presence points loaded`).

Tips for preparing this layer:

- **Geometry** must be `Point` or `MultiPoint`.
- **CRS** can be any QGIS-supported CRS; QMaxent reprojects internally to the
    rasters' CRS as needed.
- **Duplicate-point removal** is left to the user. If your data has multiple
    records at exactly the same coordinates, deduplicate first
    (`Vector → Geoprocessing tools → Delete duplicate geometries`).
- **Spatial filtering**: if you only want to model a sub-region, clip the
    layer to that region before adding it here.

## Environmental Rasters list

The big list panel below holds every raster the model will see. Use the
buttons to manage it:

- **Add from project** – batch-add every loaded raster layer in the QGIS
    project. Quickest way to populate the list when you have just dragged
    in a folder of bioclim variables.
- **Remove selected** – removes the highlighted entry.
- **▲ / ▼** – reorder rasters. The order does not affect the *fitted* model,
    but it does affect the column order in the results XLSX, so it is worth
    arranging variables sensibly.

Each raster has a `[continuous]` / `[categorical]` toggle on the right. **You
must mark categorical rasters as such** — QMaxent applies one-hot encoding
to them and any continuous treatment of a categorical variable will produce
nonsense. Variables like a landcover class index, soil type, or biome ID
belong here.

After loading the Bradypus example dataset and marking `biome` as categorical,
the panel looks like this:

![Data tab with Bradypus presence layer (116 points), 7 environmental rasters, biome marked categorical, and Check Raster Consistency: All 9 rasters share grid](images/ui/dock-1-data-with-bradypus.png)

## Check Raster Consistency

This is QMaxent's silent-error gate. It verifies that every raster shares
**the same CRS, extent, and resolution**. Maxent on mismatched rasters does
not throw an error — it just produces wrong predictions because covariates
get sampled at offset cells. We strongly recommend running this check
after any change to the raster list.

Three possible outcomes:

| Status | Meaning |
|---|---|
| ✓ All rasters share grid (CRS, resolution) | You are good to proceed |
| ⚠ CRS mismatch or extent mismatch | Use **Harmonize Rasters** (next section) |
| ⚠ Resolution mismatch | Use **Harmonize Rasters** to resample to a common grid |

The status line below the button records the outcome and the shared CRS and
resolution when the check passes — a sanity-check that survives a save+reload
of the QGIS project.

For a worked example of what happens when this check fails and how to fix
it, see the [Ariolimax worked example](examples/ariolimax.md).

## Harmonize Rasters

When **Check Raster Consistency** finds a mismatch, the Harmonize Rasters
dialog (opened from the same area) reprojects, clips, and resamples each
raster to a common grid that you choose. Behind the scenes it uses
[`gdalwarp`](https://gdal.org/programs/gdalwarp.html), the same tool the QGIS
**Warp (Reproject)** algorithm wraps. Inputs are not modified; harmonized
copies are written next to your project so the originals stay intact.

## Background Points

Maxent does not need true absences; instead it samples the *available*
environmental space using **background points**. The default of 10,000 is
the recommendation in Phillips et al. (2017) and works well for almost any
study area.

When to override the default:

- **Tiny study area** (a single watershed, an island): you may not have
    10,000 distinct cells — use a smaller value such as 5,000 or even
    `n_cells × 0.5`.
- **Continental or global modeling**: keeping it at 10,000 is fine; the
    underlying maxnet algorithm scales well.
- **Heavy spatial sampling bias** in your presences: pair this with the
    *Down-weight spatially clustered points* option in the
    [Parameters tab](parameters-tab.md).

The status bar reports the actual count after sampling
(e.g. `background=10,113`); slight differences from the request reflect
NaN-cell exclusion in the rasters.

## Loading an existing model (.pkl)

The **Load existing model (.pkl)…** button at the top right opens a saved
model and walks you through mapping each model variable to a current QGIS
raster. This is useful for re-projecting to a different region, comparing
two studies, or sharing models with collaborators. See
[Saving and reusing models](saving-models.md) for the full procedure and
the important [security note on pickle files](saving-models.md).

## What the dock remembers

Layer choices and the categorical/continuous toggles are saved with your
QGIS project (`.qgs` / `.qgz`). Reopening the project restores everything in
this tab so you can resume modeling without rebuilding the configuration.
