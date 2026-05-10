# ⑤ Priority Sites for Survey

The fifth tab turns a habitat-suitability raster into a **field-ready list
of candidate survey sites**. There are two distinct purposes — discovering
new populations versus validating the model — and QMaxent supports both
with separate sampling strategies grounded in the survey-design literature.

## Survey purpose

Pick one:

| Mode | Goal | Reference |
|---|---|---|
| **Discovery** | Find new populations in unsurveyed high-suitability areas | [Williams et al. 2009](references.md) |
| **Model validation** | Test whether the suitability gradient predicts presence/absence | [Rhoden, Peterman & Taylor 2017](references.md) |

The two modes ask very different questions of the same map, so they sample
it differently:

- **Discovery** preferentially picks the highest-suitability cells far
  from any known occurrence — the most informative locations for finding
  the species where it has not been recorded yet.
- **Model validation** stratifies the suitability gradient and samples
  proportionally within strata — the design needed for an unbiased
  presence/absence test of the model's calibration.

![Priority Sites tab in Discovery mode after extraction](images/ui/dock-5-priority-sites-discovery.png)

## Sampling strategy

A second drop-down picks the within-mode sampling strategy:

| Strategy | What it does |
|---|---|
| **Top-N (highest first)** | Take the top-N cells by suitability, subject to spacing constraints |
| **Threshold-stratified** | Divide cells above the minimum suitability into N quantile bins and sample equally from each |
| **Random above threshold** | Random draw from cells above the minimum suitability |

Top-N is the right choice when you want to maximise hit-rate (the
classical *Discovery* design of [Williams et al. 2009](references.md)).
Threshold-stratified is what
[Rhoden, Peterman & Taylor 2017](references.md) recommend for model
validation because it gives the test the statistical power to detect a
declining presence-rate across the suitability gradient.

## Spacing constraints

Two distance fields define the spatial structure of the sampled set:

- **Minimum distance from existing presences** — keeps candidates away
  from known occurrences. Set to the species' detection radius
  (e.g. 1 km for a calling fairy pitta, 200 m for a sessile slug). This
  prevents the algorithm from re-sampling territory already covered by
  earlier surveys.
- **Minimum distance between candidates** — keeps the candidate set
  spatially independent. A typical value is half the *minimum distance
  from existing presences*. Whittaker-style **spatially balanced**
  sampling, in the sense of [Stevens & Olsen 2004](references.md), is
  approximated by setting this value to the median nearest-neighbour
  distance you want in the final set.

QMaxent enforces both constraints exactly — candidates are dropped, not
moved, when a constraint is violated, so the surviving set is always
within-spec.

## Reverse geocoding

When the **Reverse-geocode addresses** checkbox is on, QMaxent calls the
[Nominatim](https://nominatim.org) API for each candidate point and adds
columns for `country`, `province`, `city_county`, `district`, and a
human-readable `display_name`. This makes the resulting GeoPackage
directly usable for permit applications and field-team coordination —
particularly valuable in jurisdictions where survey access requires
prior administrative notification.

The geocoder is rate-limited to one request per second to comply with
Nominatim's public-server fair-use policy; for large candidate sets
(>1,000 points) consider running the extractor in batches or pointing
QMaxent at a self-hosted Nominatim instance via the **Advanced** options.

![Priority Sites GeoPackage attribute table with reverse-geocoded addresses](images/results/attribute-table-priority-sites.png)

## Outputs

After clicking **▶ Extract Priority Sites**, two outputs are produced:

- **`priority_sites.gpkg`** — a GeoPackage layer auto-loaded into QGIS
  and styled with red point symbols.
- **A `Priority sites` sheet** appended to the existing `results.xlsx`
  workbook, with the same columns as the GeoPackage attribute table.

![Priority Sites tab after extraction with the candidates rendered on the QGIS canvas](images/ui/dock-5-priority-sites-extracted.png)

The candidates are immediately ready for export to mobile-GIS apps such
as **QField**, **Mergin Maps**, or **Locus Map** for offline field use.

## Choosing parameter values

Some practical defaults from the literature:

- **Number of candidates**: 20–30 per species per season is a typical
  field-survey budget for a single team
  ([Robinson et al. 2018](references.md)).
- **Minimum distance from existing presences**: the species' typical
  home-range diameter is a good starting point. Too small and you
  re-sample known sites; too large and you push candidates into
  marginal habitat.
- **Minimum suitability**: 0.5 for a balanced search; 0.7+ for a
  focused high-confidence survey; lower (0.2–0.3) for *negative
  controls* in a model-validation study.

## Worked applications

- The [Bradypus example](examples/bradypus.md) walks through a Discovery
  extraction at landscape scale.
- The [Pitta nympha example](examples/pitta-nympha.md) shows the same
  workflow at municipality scale, with reverse-geocoded Korean
  administrative addresses ready for field permits.

## Next

Move to [Saving and reusing models](saving-models.md) to learn how to
share the trained model with collaborators or re-project it onto a new
raster stack later.
