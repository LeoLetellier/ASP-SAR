#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_raster.py
------------

Usage: index_raster.py <size> [<offset>] <prefix>

Options:
-h | --help         Show this screen
"""

import docopt
from osgeo import gdal
import numpy as np


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    size = [int(k) for k in arguments["<size>"].split(",", 2)]
    offset = arguments["<offset>"]
    if offset is not None:
        offset = [int(k) for k in offset.split(",", 2)]
    else:
        offset = [0, 0] 
    prefix = arguments["<prefix>"]

    j, i = np.indices((size[1], size[0]))
    i += offset[0]
    j += offset[1]

    driver = gdal.GetDriverByName("GTiff")
    ds_i = driver.Create(prefix + "_i.tif", size[0], size[1], 1, gdal.GDT_Float32)
    ds_j = driver.Create(prefix + "_j.tif", size[0], size[1], 1, gdal.GDT_Float32)

    ds_i.GetRasterBand(1).WriteArray(i)
    ds_j.GetRasterBand(1).WriteArray(j)
