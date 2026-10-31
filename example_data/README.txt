QMaxent example datasets (for QMaxent 0.2.0)
=============================================

This folder contains the two public example datasets used in the QMaxent
manual and loaded by the Example Dataset Downloader of the plugin.
The files are unchanged copies of the originals listed below.

1. bradypus/  (Bradypus variegatus, brown-throated three-toed sloth)
   - bradypus.csv: 116 presence records (columns species, lon, lat; WGS 84)
   - bio1, bio5, bio6, bio7, bio8, bio12, bio16, bio17: bioclimatic rasters
   - biome: categorical biome raster
   - Raster format: raster package .grd/.gri pairs (readable by GDAL/QGIS)
   - Source: example data of the dismo R package (inst/ex), via the CRAN
     GitHub mirror https://github.com/cran/dismo
   - Original study: Phillips SJ, Anderson RP, Schapire RE (2006) Maximum
     entropy modeling of species geographic distributions. Ecological
     Modelling 190: 231-259.
   - The dismo package is distributed under GPL (>= 3).

2. ariolimax/  (Ariolimax columbianus, Pacific banana slug)
   - ariolimax-ca.gpkg: 3,732 presence points (EPSG:32610)
   - ca-cloudcover-mean/stdv, ca-leafareaindex-mean/stdv,
     ca-surfacetemp-mean/stdv: six environmental rasters for California
     (GeoTIFF, EPSG:32610, 1 km)
   - Source: example data of the elapid Python package, hosted by its
     author at https://earth-chris.github.io/images/research/
   - Reference: Anderson CB (2023) elapid: Species distribution modeling
     tools for Python. Journal of Open Source Software 8(84): 4930.

Use
   In QGIS, add the presence layer and the rasters to a project and open
   Plugins > QMaxent > QMaxent Analysis. Alternatively, use
   Plugins > QMaxent > Download Example Dataset, which downloads the same
   files from the sources above.

Please cite the original sources when you use these data. The terms of the
original providers apply.
