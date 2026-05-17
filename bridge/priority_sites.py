"""Priority sites for field survey — post-prediction sampling.

Extracts candidate occurrence sites from a trained Maxent prediction
raster, filtered to be:

  1) above a calibrated suitability threshold (Pearson 2007's MTP /
     T10 / Liu 2013's MaxSSS — canonical SDM thresholds),
  2) geographically separated from the existing presence points
     (so the user is directed to *unsurveyed* habitat), and
  3) spatially de-clustered among themselves (so survey effort is
     spread, not piled in one hotspot).

Two sampling modes are supported:

  * Discovery (default): pick the highest-suitability cells first
    and apply the spacing filter. Use this when the goal is to
    focus field effort where the species is most likely.

  * Validation (optional, Rhoden, Peterman & Taylor 2017,
    PeerJ 5:e3632 https://doi.org/10.7717/peerj.3632): split the
    cells above the threshold into four suitability quartiles and
    pick equal numbers from each. Designed for evaluating model
    performance across the full suitability gradient — by intent,
    this includes lower-suitability cells.

Reverse geocoding (lat/lon → administrative address) is layered on
top via OpenStreetMap Nominatim. Nominatim is chosen over commercial
alternatives (VWorld, Kakao, Naver) because it requires no API key
and works globally — the trade-off is a 1 req/sec rate limit that we
respect explicitly. Korean coverage is good for province (도) / city
(시/군/구) levels; sub-district (읍/면/동) coverage varies and is
left blank when missing.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
import urllib.request
from typing import Iterable, Optional


# ─── Threshold computation ───────────────────────────────────────────────

# Standard SDM threshold methods. Each returns a single numeric cut-off
# applied to the prediction raster to define "suitable" cells.
THRESHOLD_METHODS = (
    "MTP",      # Minimum Training Presence (Pearson et al. 2007)
    "T10",      # 10th percentile training presence (Pearson et al. 2007)
    "MaxSSS",   # Maximum Sum of Sensitivity + Specificity (Liu 2013)
    "Custom",   # User-supplied numeric value
)


def compute_threshold(method: str,
                      training_predictions: list,
                      background_predictions: Optional[list] = None,
                      custom_value: Optional[float] = None) -> float:
    """Return a suitability cut-off given a method and training values.

    ``training_predictions`` are the model's output values *at the
    presence points used for training* — a 1D array-like of floats in
    the [0, 1] range for cloglog/logistic transforms. ``background_
    predictions`` are the corresponding values at background points;
    they are required for MaxSSS but not for MTP / T10.

    Raises ``ValueError`` when the method needs a value it didn't get
    (e.g. MaxSSS without background predictions, Custom without a value).
    """
    import numpy as np
    tp = np.asarray(training_predictions, dtype=float)
    if tp.size == 0:
        raise ValueError("No training predictions to compute threshold from.")

    if method == "MTP":
        # Pearson et al. 2007: most conservative — every training
        # presence must be classified as suitable.
        return float(tp.min())

    if method == "T10":
        # Pearson et al. 2007: 10% of training presences are allowed
        # to fall below the threshold (i.e., they're treated as
        # outliers / sampling errors). Standard maxent.jar default
        # threshold rule.
        return float(np.percentile(tp, 10))

    if method == "MaxSSS":
        # Liu 2013: pick the threshold maximising sensitivity +
        # specificity. Sensitivity = fraction of presences ≥ t;
        # specificity = fraction of background < t. We sweep
        # candidate thresholds drawn from the union of presence and
        # background values, which is the most informative grid.
        if background_predictions is None or len(background_predictions) == 0:
            raise ValueError(
                "MaxSSS requires background predictions; pass "
                "background_predictions=..."
            )
        bg = np.asarray(background_predictions, dtype=float)
        candidates = np.unique(np.concatenate([tp, bg]))
        # Vectorised search: for each candidate t, count presences
        # at-or-above and background below.
        best_t, best_score = float(tp.min()), -np.inf
        for t in candidates:
            sens = float((tp >= t).mean())
            spec = float((bg <  t).mean())
            score = sens + spec
            if score > best_score:
                best_score = score
                best_t = float(t)
        return best_t

    if method == "Custom":
        if custom_value is None:
            raise ValueError("Custom method requires a custom_value.")
        return float(custom_value)

    raise ValueError(f"Unknown threshold method: {method!r}")


# ─── Sampling priority sites from a prediction raster ────────────────────

def extract_priority_sites(prediction_path: str,
                           presence_xy: list,
                           threshold: float,
                           n_sites: int,
                           min_distance_from_presence_m: float = 1000.0,
                           min_distance_between_sites_m: float = 500.0,
                           stratify_by_quartile: bool = True,
                           sampling_order: str = "topn",
                           random_seed=42) -> list:
    """Sample ``n_sites`` priority sites from a Maxent prediction raster.

    Returns a list of dicts, each with keys::
      {"lat", "lon", "suitability"}

    Sampling strategy:
      1. Read the prediction raster as a 2D array.
      2. Mask cells below ``threshold`` (these are not "suitable").
      3. Optionally split the suitable cells into 4 suitability
         quartiles and sample evenly across them — the Rhoden et al.
         (2017) validation approach, which deliberately includes
         lower-suitability cells to evaluate the model across its
         full predicted gradient.
      4. For each candidate cell, enforce:
           * distance to every existing presence point ≥
             ``min_distance_from_presence_m`` (so the user is directed
             to *new* habitat), and
           * distance to every already-accepted candidate ≥
             ``min_distance_between_sites_m`` (so survey effort spreads
             rather than clustering).
      5. Stop when ``n_sites`` accepted, or report the actual count if
         the suitable cells cannot satisfy the spacing constraints.

    The function is deterministic given ``random_seed`` (an int);
    pass ``random_seed=None`` to draw from OS entropy on every call,
    matching the unchecked-state behavior of the dock's "Fix random
    seed" toggle.
    """
    import numpy as np
    import rasterio
    from pyproj import Transformer

    rng = np.random.default_rng(random_seed)

    with rasterio.open(prediction_path) as src:
        arr = src.read(1, masked=True)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    # Mask: keep only cells that are (a) not nodata and (b) ≥ threshold.
    data = np.asarray(arr.filled(np.nan), dtype=float) \
        if hasattr(arr, "filled") else np.asarray(arr, dtype=float)
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    suitable_mask = np.isfinite(data) & (data >= threshold)
    if not suitable_mask.any():
        return []

    # Get the indices of all suitable cells.
    rows, cols = np.where(suitable_mask)
    suit_values = data[rows, cols]

    # Assign each suitable cell to a quartile of suitability so we can
    # sample evenly across the [threshold, max] range when the user
    # opted into spatial stratification.
    if stratify_by_quartile and len(suit_values) >= 8:
        # numpy's quantile uses linear interpolation; we pick the
        # 25/50/75 percentiles of suitable values.
        q1, q2, q3 = np.quantile(suit_values, [0.25, 0.50, 0.75])
        bin_id = np.digitize(suit_values, [q1, q2, q3])  # 0..3
    else:
        bin_id = np.zeros_like(suit_values, dtype=int)

    # Convert pixel centres to lat/lon. We need *projected* metres for
    # the distance filter, so we'll transform on the fly.
    xs, ys = rasterio.transform.xy(transform, rows.tolist(), cols.tolist())
    xs = np.asarray(xs); ys = np.asarray(ys)

    # Build forward transformer to a metric CRS for distance checks.
    # Use an Equal-Area projection to get reasonable distances regardless
    # of the source CRS.
    is_geographic = bool(crs and crs.is_geographic) if crs else True
    if is_geographic:
        # World Equidistant Cylindrical (Plate Carrée variant) — adequate
        # for the modest distances (kilometres) we care about. For users
        # who project rasters to a local CRS, the source CRS is already
        # metric and we skip this step.
        transformer_to_metric = Transformer.from_crs(
            crs or "EPSG:4326", "EPSG:6933", always_xy=True
        )
        xm, ym = transformer_to_metric.transform(xs, ys)
        # Also project presence points to the same metric CRS.
        plons = np.asarray([p[0] for p in presence_xy])
        plats = np.asarray([p[1] for p in presence_xy])
        pxm, pym = transformer_to_metric.transform(plons, plats)
    else:
        xm, ym = xs, ys
        # Presence_xy is documented as (lon, lat); convert to source CRS.
        transformer_to_src = Transformer.from_crs(
            "EPSG:4326", crs, always_xy=True
        )
        plons = np.asarray([p[0] for p in presence_xy])
        plats = np.asarray([p[1] for p in presence_xy])
        pxm, pym = transformer_to_src.transform(plons, plats)

    # Distance filter to existing presences. Vectorised: for each
    # candidate cell, find the min distance to any presence; reject
    # cells closer than the threshold.
    if len(pxm) > 0 and min_distance_from_presence_m > 0:
        # Compute pairwise distances in chunks to avoid a giant matrix.
        keep_idx = []
        chunk = 5000
        d_thresh_sq = min_distance_from_presence_m ** 2
        for i in range(0, len(xm), chunk):
            sl = slice(i, i + chunk)
            dx = xm[sl, None] - pxm[None, :]
            dy = ym[sl, None] - pym[None, :]
            dsq = dx * dx + dy * dy
            min_dsq = dsq.min(axis=1)
            keep_idx.extend((np.where(min_dsq >= d_thresh_sq)[0] + i).tolist())
        keep_idx = np.asarray(keep_idx, dtype=int)
        if keep_idx.size == 0:
            return []
        rows, cols = rows[keep_idx], cols[keep_idx]
        suit_values = suit_values[keep_idx]
        bin_id = bin_id[keep_idx]
        xm, ym = xm[keep_idx], ym[keep_idx]
        xs, ys = xs[keep_idx], ys[keep_idx]

    # If the source raster isn't geographic, build the back-transformer
    # to EPSG:4326 *once* — building a Transformer is non-trivial
    # (CRS parse + proj pipeline construction) and we'll need it for
    # every accepted site below.
    transformer_to_geo = None
    if not is_geographic:
        transformer_to_geo = Transformer.from_crs(
            crs, "EPSG:4326", always_xy=True
        )

    # Now sample N sites with an iterative greedy spacing algorithm.
    # We pre-sort by suitability (descending) within each bin, then
    # round-robin across bins to hit the per-quartile target. Each
    # candidate is accepted only if it's far enough from every
    # already-accepted site.
    n_bins = int(bin_id.max()) + 1 if len(bin_id) else 1
    target_per_bin = math.ceil(n_sites / n_bins)
    # For each bin, order the candidates. Two modes:
    #   • "topn"   — descending suitability (Validation rounds-robin
    #                across quartiles in this order; Discovery picks
    #                from the top of a single bin).
    #   • "random" — shuffled with the supplied RNG (Discovery only;
    #                gives a statistically unbiased sample within the
    #                high-suitability band, which is more honest about
    #                Maxent's relative-suitability semantics than a
    #                strict top-N greedy pass).
    bin_indices = []
    for b in range(n_bins):
        mask = bin_id == b
        idx = np.where(mask)[0]
        if idx.size == 0:
            bin_indices.append([])
            continue
        if sampling_order == "random" and not stratify_by_quartile:
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            bin_indices.append(shuffled.tolist())
        else:
            order = np.argsort(suit_values[idx])[::-1]
            bin_indices.append(idx[order].tolist())

    # Round-robin sampling with spacing filter.
    # accepted holds (xm, ym, suitability, lat, lon, quartile) per site
    # so the GPKG layer can be categorized by quartile in Validation
    # mode (Discovery mode users get quartile=0 for every site since
    # stratify_by_quartile is False there).
    accepted = []
    d_between_sq = min_distance_between_sites_m ** 2
    cursor = [0] * n_bins
    while len(accepted) < n_sites:
        progressed = False
        for b in range(n_bins):
            if len(accepted) >= n_sites:
                break
            while cursor[b] < len(bin_indices[b]):
                idx = bin_indices[b][cursor[b]]
                cursor[b] += 1
                cand_x, cand_y = xm[idx], ym[idx]
                if accepted and d_between_sq > 0:
                    dx = np.asarray([a[0] for a in accepted]) - cand_x
                    dy = np.asarray([a[1] for a in accepted]) - cand_y
                    if (dx * dx + dy * dy).min() < d_between_sq:
                        continue  # too close to an already-accepted site
                # Convert raster coords back to lat/lon (EPSG:4326)
                # for the output. If source is already geographic, xs/ys
                # are already lon/lat; otherwise use the transformer
                # we built once at the top of the function.
                if is_geographic:
                    lat = float(ys[idx])
                    lon = float(xs[idx])
                else:
                    lon, lat = transformer_to_geo.transform(
                        float(xs[idx]), float(ys[idx])
                    )
                accepted.append((
                    float(cand_x), float(cand_y),
                    float(suit_values[idx]),
                    lat, lon,
                    int(bin_id[idx]),
                ))
                progressed = True
                break
        if not progressed:
            break  # no more candidates anywhere

    return [
        {"lat": lat, "lon": lon, "suitability": s, "quartile": q}
        for (_, _, s, lat, lon, q) in accepted
    ]


# ─── Reverse geocoding via Nominatim ─────────────────────────────────────

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_MIN_INTERVAL_SEC = 1.05  # OSM policy: ≤ 1 req/sec; we add slack
NOMINATIM_USER_AGENT = (
    "qmaxent/0.1.0 (https://github.com/osgeokr/qmaxent)"
)


def reverse_geocode_one(lat: float, lon: float,
                        language: str = "ko,en",
                        timeout: float = 15.0) -> dict:
    """Look up a single coordinate via Nominatim.

    Returns a dict with administrative levels extracted from the OSM
    address response::
      {"country", "province", "city_county", "district", "display_name"}

    On failure (network, parse, missing keys) returns the same dict
    with empty strings — geocoding is best-effort and never fatal.
    The caller is responsible for the 1-req/sec rate limit; this
    function does not sleep.
    """
    qs = urllib.parse.urlencode({
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "format": "json",
        "accept-language": language,
        "zoom": "14",         # ~ city / town level
    })
    url = f"{NOMINATIM_URL}?{qs}"
    # Defence in depth: urllib.request.urlopen accepts file:// and other
    # local schemes by default. NOMINATIM_URL is a hardcoded https URL,
    # but we validate the scheme explicitly so that a future change (or
    # any unexpected substitution) cannot turn this into a local-file
    # read. Bandit B310.
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(
            f"Refusing to fetch non-HTTP(S) URL: {url!r}"
        )
    req = urllib.request.Request(
        url, headers={"User-Agent": NOMINATIM_USER_AGENT}
    )
    try:
        # Bandit B310: scheme is validated immediately above to be
        # http(s) only, so file:// / ftp:// abuse is impossible here.
        with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return {"country": "", "province": "", "city_county": "",
                "district": "", "display_name": ""}

    addr = data.get("address", {}) or {}
    return {
        "country":     str(addr.get("country", "")),
        # Korean 도 ≈ "state"/"province"; OSM uses "state" most of the time
        "province":    str(addr.get("state", "") or addr.get("province", "")),
        # Korean 시/군/구 ≈ city / county / borough
        "city_county": str(
            addr.get("city", "")
            or addr.get("county", "")
            or addr.get("town", "")
            or addr.get("borough", "")
        ),
        # Korean 읍/면/동 ≈ suburb / village / city_district
        "district":    str(
            addr.get("suburb", "")
            or addr.get("village", "")
            or addr.get("city_district", "")
            or addr.get("neighbourhood", "")
        ),
        "display_name": str(data.get("display_name", "")),
    }


def estimate_geocoding_seconds(n: int) -> float:
    """How long will Nominatim geocoding of ``n`` points take?

    Returns seconds, including the 1 req/sec rate limit. We expose
    this so the dialog can show "약 8분 소요" before the user starts a
    long-running job.
    """
    return n * NOMINATIM_MIN_INTERVAL_SEC
