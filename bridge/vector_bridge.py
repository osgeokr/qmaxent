"""Bridge: QGIS vector layers ↔ elapid GeoDataFrames."""

import os
import tempfile

from qgis.core import (
    QgsVectorLayer,
    QgsVectorFileWriter,
    QgsCoordinateTransformContext,
)


def layer_to_geodataframe(layer: QgsVectorLayer):
    """Convert a QgsVectorLayer to a geopandas GeoDataFrame.

    Exports the layer to a temporary GeoPackage and reads it with geopandas.
    This avoids any dependency on fiona being available in the QGIS environment.

    Args:
        layer: A QGIS point or polygon vector layer.

    Returns:
        geopandas.GeoDataFrame with the layer's features and geometry.
    """
    import geopandas as gpd

    # Use mkstemp for a race-free temp file. Close the OS-level fd
    # immediately so QGIS can reopen the path for writing; we keep the
    # path string and clean up everything (including any GeoPackage
    # WAL/SHM sidecars QGIS may have left behind) in the finally block.
    fd, tmp = tempfile.mkstemp(suffix=".gpkg", prefix="qmaxent_vec_")
    os.close(fd)
    try:
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.fileEncoding = "UTF-8"

        error, msg, _, _ = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer, tmp, QgsCoordinateTransformContext(), options
        )
        if error != QgsVectorFileWriter.NoError:
            raise RuntimeError(f"Failed to export layer: {msg}")

        gdf = gpd.read_file(tmp)
        return gdf
    finally:
        # Remove the gpkg and any SQLite WAL/SHM sidecars that may have
        # been written during the export. Best-effort: silently ignore
        # missing files so cleanup never masks a real error above.
        for suffix in ("", "-journal", "-shm", "-wal"):
            p = tmp + suffix
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


def presence_layer_to_geodataframe(layer: QgsVectorLayer):
    """Convert a presence point layer to a GeoDataFrame with only geometry.

    Args:
        layer: QgsVectorLayer containing species occurrence points.

    Returns:
        GeoDataFrame with Point geometry column.
    """
    import geopandas as gpd

    gdf = layer_to_geodataframe(layer)
    # Keep only geometry — attribute columns are not used for presence points
    return gdf[["geometry"]].copy()
