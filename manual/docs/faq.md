# FAQ & troubleshooting

Common questions, error messages, and workarounds. If you do not see
your issue here, please open an
[issue on GitHub](https://github.com/osgeokr/qmaxent/issues).

## Installation issues

### "QMaxent environment not ready" stays red after install

Check the **Plugins → QMaxent → Manage Dependencies** dialog. If any of
the required packages (`elapid`, `numpy`, `scikit-learn`, `pandas`,
`openpyxl`) is reported missing, click **Install missing**.

If the dialog itself errors out, your QGIS Python may be missing `pip`.
On Windows, run **OSGeo4W Shell → `python -m ensurepip --upgrade`**;
on macOS/Linux, install QGIS via the OSGeo4W or Conda channel that ships
with `pip` enabled.

### "Could not find a version that satisfies the requirement elapid"

This usually means the QGIS Python is older than 3.9. QMaxent requires
**Python ≥ 3.9** because that is the floor elapid sets. Upgrade QGIS to
the **LTR ≥ 3.34** release; the bundled Python is 3.9+.

### Plugin Manager shows QMaxent but "Install" is greyed out

You probably already have QMaxent installed in a *user* plugin
directory and the Plugin Manager is showing the *system* package list.
Check **Plugins → Manage and Install Plugins → Installed** — uninstall
the older copy first, then re-install the latest.

## Modeling errors

### "ValueError: All presence points fell on NoData cells"

The presence layer's CRS or extent does not overlap the raster stack.
Check on **① Data**:

1. Click on a presence point in QGIS to read its coordinates.
2. Click on a raster cell in roughly the same area to confirm the
   raster has a value there.
3. If they disagree, the layer is misprojected — re-export the
   presence layer with the rasters' CRS using **Vector → Save Layer
   As…**.

### "Convergence not reached after 500 iterations"

The model could not find a likelihood maximum within the iteration cap.
Three common causes:

- **Too few presences** for the chosen feature classes. Switch to
  **Auto** mode on **② Parameters** and the maxnet rule will simplify
  the feature set automatically.
- **Highly collinear variables**. Use **Jackknife Importance** on the
  results tab to identify near-zero contributors and drop them.
- **Categorical variable with very rare classes**. Re-bin the rare
  classes into an `Other` category before modeling.

### "Grid mismatch — CRS, extent, resolution differ across rasters"

This is the **Check Raster Consistency** failure mode. Click
**Harmonize to Folder…** and let QMaxent reproject every raster to the
highest-resolution one. The
[Ariolimax worked example](examples/ariolimax.md) walks through this.

## Performance

### Training is very slow (>10 minutes for 100 presences)

Two common causes:

- **Very large raster stack** (e.g. 30-m resolution over a continental
  extent). Background-point extraction is the bottleneck. Either reduce
  the raster resolution to a value appropriate for the species' home-
  range size, or restrict the analysis to a relevant ROI by clipping
  the rasters first.
- **Too many background points**. The default 10,000 is appropriate for
  most studies; a value of 50,000+ rarely helps and slows the run
  proportionally.

### Spatial projection is fast but uses lots of RAM

Projection loads the full raster stack into memory cell-by-cell. For a
continental 1-km stack of 10 variables, expect ~2 GB peak. If RAM is
tight, downsample the rasters before projection — projection at the
training resolution is methodologically defensible
([Elith, Kearney & Phillips 2010](references.md)).

## Output and projection

### "Why are some cells in my projection NoData when the rasters cover them?"

The unified preflight dialog auto-masks two cell categories to NoData:

1. Categorical class codes that did not appear during training.
2. Cells where any continuous variable lies outside its training
   range — *only when the relevant safety mask is enabled*.

This is intentional ([Elith, Kearney & Phillips 2010](references.md))
— extrapolation beyond the training domain is undefined and should be
flagged, not silently filled in.

### Can I project onto a different region than I trained on?

Yes — but the preflight dialog will flag every continuous variable
whose new-region range exceeds its training range, and will auto-mask
unseen categorical codes. Read the dialog before clicking **Yes**;
treat any reported extrapolation as a finding to discuss in the paper.

### The output cloglog values look uniformly low

Probably normal — cloglog values
([Phillips et al. 2017](references.md)) are calibrated as
"probability of presence given a typical sample of the species at this
location", and that is genuinely low (<0.5) over most of any species'
range. The peaks (0.7+) are what matter; the absolute scale is not
directly comparable across species or studies.

## How to ask a good question

When opening an issue on GitHub, please include:

1. The **QMaxent version** (Plugins → Manage and Install Plugins →
   Installed, or `metadata.txt`).
2. The **QGIS version** (Help → About).
3. The **exact error message** (copy from the Training tab log or the
   QGIS Python console).
4. A **minimum reproducer** — ideally a small dataset that triggers the
   problem, or screenshots of the **① Data** and **② Parameters** tabs
   so we can see the configuration.

The fastest issues to resolve are the ones that ship with everything a
maintainer needs to recreate the failure on their own machine.
