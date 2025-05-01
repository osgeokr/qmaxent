import geopandas as gpd
import rasterio as rio
import numpy as np
import pickle
import gzip

from typing import Tuple, Any, Union, List, Dict, Iterable
from pyproj import CRS as CRSType
from numbers import Number

Vector = Union[gpd.GeoSeries, gpd.GeoDataFrame]

class NoDataException(Exception):
    pass

def to_iterable(var: Any) -> list:
    """Checks and converts variables to an iterable type.

    Args:
        var: The input variable to check and convert.

    Returns:
        A list containing var or var itself if already iterable (except strings).
    """
    if not hasattr(var, "__iter__"):
        return [var]
    elif isinstance(var, str):
        return [var]
    else:
        return var

def validate_gpd(geo: Vector) -> None:
    """Validates whether an input is a GeoDataFrame or a GeoSeries.

    Args:
        geo: an input variable that should be in GeoPandas format

    Raises:
        TypeError: if geo is not a GeoPandas dataframe or series
    """
    if not isinstance(geo, (gpd.GeoDataFrame, gpd.GeoSeries)):
        raise TypeError("Input must be a GeoDataFrame or GeoSeries")

def string_to_crs(crs_str: str) -> CRSType:
    """Converts a CRS string to a pyproj CRS object."""
    return CRSType.from_user_input(crs_str)

def crs_match(crs1: Union[CRSType, str], crs2: Union[CRSType, str]) -> bool:
    """Evaluates whether two coordinate reference systems are the same.

    Args:
        crs1: The first CRS (CRS object or string).
        crs2: The second CRS (CRS object or string).

    Returns:
        True if CRS match, else False.
    """
    if isinstance(crs1, str):
        crs1 = string_to_crs(crs1)
    if isinstance(crs2, str):
        crs2 = string_to_crs(crs2)

    return crs1 == crs2

def count_raster_bands(raster_paths: list) -> int:
    """Returns the total number of bands from a list of rasters.

    Args:
        raster_paths: List of raster data file paths.

    Returns:
        n_bands: total band count.
    """
    n_bands = 0
    for path in raster_paths:
        with rio.open(path) as src:
            n_bands += src.count
    return n_bands

def n_digits(number: Number) -> int:
    """Counts the number of significant integer digits of a number.

    Args:
        number: the number to evaluate.

    Returns:
        order: number of digits required to represent a number
    """
    if number == 0:
        return 1
    return int(np.floor(np.log10(abs(number))) + 1)


def make_band_labels(n_bands: int) -> list:
    """Creates a list of band names to assign as dataframe columns.

    Args:
        n_bands: total number of raster bands to create labels for.

    Returns:
        labels: list of column names.
    """
    n_zeros = n_digits(n_bands)
    labels = [f"b{i + 1:0{n_zeros}d}" for i in range(n_bands)]
    return labels

def format_band_labels(raster_paths: list, labels: List[str] = None):
    """Verify the number of labels matches the band count, create labels if none passed.

    Args:
        raster_paths: count the total number of bands in these rasters.
        labels: a list of band labels.

    Returns:
        labels: creates default band labels if none are passed.
    """
    n_bands = count_raster_bands(raster_paths)

    if labels is None:
        labels = make_band_labels(n_bands)

    n_labels = len(labels)
    assert n_labels == n_bands, f"number of band labels ({n_labels}) != n_bands ({n_bands})"

    return labels.copy()

def get_feature_types(return_string: bool = False) -> Union[list, str]:
    feature_types = "lqpht" if return_string else ["linear", "quadratic", "product", "hinge", "threshold"]
    return feature_types

def validate_feature_types(features: Union[str, list]) -> list:
    """Ensures the feature classes passed are maxent-legible

    Args:
        features: List or string that must be in ["linear", "quadratic", "product",
            "hinge", "threshold", "auto"] or string "lqphta"

    Returns:
        valid_features: List of formatted valid feature values
    """
    valid_list = get_feature_types(return_string=False)
    valid_string = get_feature_types(return_string=True)
    valid_features = list()

    # ensure the string features are valid, and translate to a standard feature list
    if type(features) is str:
        for feature in features:
            if feature == "a":
                return valid_list
            assert feature in valid_string, "Invalid feature passed: {}".format(feature)
            if feature == "l":
                valid_features.append("linear")
            elif feature == "q":
                valid_features.append("quadratic")
            elif feature == "p":
                valid_features.append("product")
            elif feature == "h":
                valid_features.append("hinge")
            elif feature == "t":
                valid_features.append("threshold")

    # or ensure the list features are valid
    elif type(features) is list:
        for feature in features:
            if feature == "auto":
                return valid_list
            assert feature in valid_list, "Invalid feature passed: {}".format(feature)
            valid_features.append(feature)

    return valid_features

def repeat_array(x: np.array, length: int = 1, axis: int = 0) -> np.ndarray:
    """Repeats a 1D numpy array along an axis to an arbitrary length

    Args:
        x: the n-dimensional array to repeat
        length: the number of times to repeat the array
        axis: the axis along which to repeat the array (valid values include 0 to n+1)

    Returns:
        An n+1 dimensional numpy array
    """
    return np.expand_dims(x, axis=axis).repeat(length, axis=axis)

def validate_boolean(var: Any) -> bool:
    """Evaluates whether an argument is boolean True/False

    Args:
        var: the input argument to validate

    Returns:
        var: the value if it passes validation

    Raises:
        AssertionError: `var` was not boolean
    """
    assert isinstance(var, bool), "Argument must be True/False"
    return var

def validate_numeric_scalar(var: Any) -> bool:
    """Evaluates whether an argument is a single numeric value.

    Args:
        var: the input argument to validate

    Returns:
        var: the value if it passes validation

    Raises:
        AssertionError: `var` was not numeric.
    """
    assert isinstance(var, (int, float)), "Argument must be single numeric value"
    return var

def save_object(obj: object, path: str, compress: bool = True) -> None:
    """Writes a python object to disk for later access.

    Args:
        obj: a python object or variable to be saved (e.g., a MaxentModel() instance)
        path: the output file path
    """
    obj = pickle.dumps(obj)

    if compress:
        obj = gzip.compress(obj)

    with open(path, "wb") as f:
        f.write(obj)

def get_raster_band_indexes(raster_paths: list) -> Tuple[int, list]:
    """Counts the number raster bands to index multi-source, multi-band covariates.

    Args:
        raster_paths: a list of raster paths

    Returns:
        (nbands, band_idx): int and list of the total number of bands and the 0-based start/stop
            band index for each path
    """
    nbands = 0
    band_idx = [0]
    for i, raster_path in enumerate(raster_paths):
        with rio.open(raster_path) as src:
            nbands += src.count
            band_idx.append(band_idx[i] + src.count)

    return nbands, band_idx

def create_output_raster_profile(
    raster_paths: list,
    template_idx: int = 0,
    windowed: bool = True,
    nodata: Number = None,
    count: int = 1,
    compress: str = None,
    driver: str = "GTiff",
    bigtiff: bool = True,
    dtype: str = "float32",
) -> Tuple[Iterable, Dict]:
    """Gets parameters for windowed reading/writing to output rasters.

    Args:
        raster_paths: raster paths of covariates to apply the model to
        template_idx: index of the raster file to use as a template. template_idx=0 sets the first raster as template
        windowed: perform a block-by-block data read. slower, but reduces memory use
        nodata: output nodata value
        count: number of bands in the prediction output
        driver: output raster file format (from rasterio.drivers.raster_driver_extensions())
        compress: compression type to apply to the output file
        bigtiff: specify the output file as a bigtiff (for rasters > 2GB)
        dtype: rasterio data type string

    Returns:
        (windows, profile): an iterable and a dictionary for the window reads and the raster profile
    """
    with rio.open(raster_paths[template_idx]) as src:
        if windowed:
            windows = [window for _, window in src.block_windows()]
        else:
            windows = [rio.windows.Window(0, 0, src.width, src.height)]

        dst_profile = src.profile.copy()
        dst_profile.update(
            count=count,
            dtype=dtype,
            nodata=nodata,
            compress=compress,
            driver=driver,
        )
        if bigtiff and driver == "GTiff":
            dst_profile.update(BIGTIFF="YES")

    return windows, dst_profile

def check_raster_alignment(raster_paths: list) -> bool:
    """Checks whether the extent, resolution and projection of multiple rasters match exactly.

    Args:
        raster_paths: a list of raster covariate paths

    Returns:
        whether all rasters align
    """
    first = raster_paths[0]
    rest = raster_paths[1:]

    with rio.open(first) as src:
        res = src.res
        bounds = src.bounds
        transform = src.transform

    for path in rest:
        with rio.open(path) as src:
            if src.res != res or src.bounds != bounds or src.transform != transform:
                return False

    return True




