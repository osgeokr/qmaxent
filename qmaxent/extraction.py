import numpy as np
import geopandas as gpd
import pandas as pd
import os
import rasterio as rio

from typing import Union, List
from shapely.geometry import Point
from pyproj import CRS as CRSType

from .utils import to_iterable
from tqdm import tqdm
from .utils import to_iterable, format_band_labels, crs_match

def xy_to_geoseries(
    x: Union[float, list, np.ndarray], 
    y: Union[float, list, np.ndarray], 
    crs: CRSType = "epsg:4326"
) -> gpd.GeoSeries:
    """Converts x/y data into a GeoPandas GeoSeries of Points."""
    x = to_iterable(x)
    y = to_iterable(y)

    points = [Point(x, y) for x, y in zip(x, y)]
    return gpd.GeoSeries(points, crs=crs)

def sample_raster(
    raster_path: str, 
    count: int, 
    nodata: float = None, 
    ignore_mask: bool = False
) -> gpd.GeoSeries:
    """Create a random geographic sample of points based on a raster's extent."""
    with rio.open(raster_path) as src:
        if src.nodata is None or ignore_mask:
            if nodata is None:
                xmin, ymin, xmax, ymax = src.bounds
                xy = np.random.uniform((xmin, ymin), (xmax, ymax), (count, 2))
            else:
                data = src.read(1)
                mask = data != nodata
                rows, cols = np.where(mask)
                samples = np.random.randint(0, len(rows), count)
                xy = np.zeros((count, 2))
                for i, sample in enumerate(samples):
                    xy[i] = src.xy(rows[sample], cols[sample])
        else:
            if nodata is None:
                masked = src.read_masks(1)
                rows, cols = np.where(masked == 255)
            else:
                data = src.read(1, masked=True)
                data.mask += data.data == nodata
                rows, cols = np.where(~data.mask)
            samples = np.random.randint(0, len(rows), count)
            xy = np.zeros((count, 2))
            for i, sample in enumerate(samples):
                xy[i] = src.xy(rows[sample], cols[sample])

        return xy_to_geoseries(xy[:, 0], xy[:, 1], crs=src.crs)

# Optional tqdm settings
tqdm_opts = {
    "ncols": 70,
    "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt}",
}

def annotate_vector(
    vector_path: str,
    raster_paths: list,
    labels: list = None,
    drop_na: bool = True,
    quiet: bool = False,
) -> gpd.GeoDataFrame:
    """Reads and stores pixel values from rasters using a point-format vector file."""
    raster_paths = to_iterable(raster_paths)
    labels = format_band_labels(raster_paths, labels)

    gdf = gpd.read_file(vector_path)
    raster_df = annotate_geoseries(
        gdf.geometry,
        raster_paths,
        labels=labels,
        drop_na=drop_na,
        quiet=quiet
    )

    gdf = pd.concat([gdf, raster_df.drop(columns="geometry", errors="ignore")], axis=1)
    return gdf

def annotate_geoseries(
    points: gpd.GeoSeries,
    raster_paths: list,
    labels: List[str] = None,
    drop_na: bool = True,
    dtype: str = None,
    quiet: bool = False,
) -> (gpd.GeoDataFrame, np.ndarray):
    """Reads and stores pixel values from rasters using point locations."""
    raster_paths = to_iterable(raster_paths)
    labels = format_band_labels(raster_paths, labels)
    n_rasters = len(raster_paths)

    raster_values = []
    valid_idxs = []
    nodata_flag = False

    for raster_idx, raster_path in tqdm(
        enumerate(raster_paths), desc="Raster", total=n_rasters, disable=quiet, **tqdm_opts
    ):
        with rio.open(raster_path, "r") as src:
            if not crs_match(points.crs, src.crs):
                points = points.to_crs(src.crs)

            if raster_idx == 0 and dtype is None:
                dtype = src.dtypes[0]

            xys = [(point.x, point.y) for point in points]

            n_points = len(points)
            samples_iter = list(
                tqdm(
                    src.sample(xys, masked=False),
                    desc="Sample",
                    total=n_points,
                    leave=False,
                    disable=quiet,
                    **tqdm_opts,
                )
            )
            samples = np.array(samples_iter, dtype=dtype)
            raster_values.append(samples)

            if drop_na and src.nodata is not None:
                nodata_flag = True
                valid_idxs.append(samples[:, 0] != src.nodata)

    values = np.concatenate(raster_values, axis=1, dtype=dtype)

    if nodata_flag:
        valid = np.all(valid_idxs, axis=0).reshape(-1, 1)
        values = np.concatenate([values, valid], axis=1, dtype=dtype)
        labels.append("valid")

    gdf = gpd.GeoDataFrame(values, geometry=points.geometry, columns=labels)

    return gdf

def annotate(
    points: Union[str, gpd.GeoSeries, gpd.GeoDataFrame],
    raster_paths: Union[str, list],
    labels: list = None,
    drop_na: bool = True,
    quiet: bool = False,
) -> gpd.GeoDataFrame:
    """Read raster values for each point in a vector and append as new columns."""
    raster_paths = to_iterable(raster_paths)
    labels = format_band_labels(raster_paths, labels)

    if isinstance(points, gpd.GeoSeries):
        points = points.reset_index(drop=True)
        gdf = annotate_geoseries(
            points,
            raster_paths,
            labels=labels,
            drop_na=drop_na,
            quiet=quiet,
        )

    elif isinstance(points, (gpd.GeoDataFrame, pd.DataFrame)):
        points = points.reset_index(drop=True)
        gdf = annotate_geoseries(
            points.geometry,
            raster_paths,
            labels=labels,
            drop_na=drop_na,
            quiet=quiet,
        )
        gdf = pd.concat([points, gdf.drop(columns="geometry", errors="ignore")], axis=1)

    elif isinstance(points, str) and os.path.isfile(points):
        gdf = annotate_vector(points, raster_paths, labels=labels, drop_na=drop_na, quiet=quiet)

    else:
        raise TypeError("points arg must be a valid path, GeoDataFrame, or GeoSeries")

    if drop_na:
        try:
            valid = gdf["valid"] == 1
            gdf = gdf[valid].drop(columns="valid").dropna().reset_index(drop=True)
        except KeyError:
            pass

    return gdf


