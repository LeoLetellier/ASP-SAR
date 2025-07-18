#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_x_y_index.py
------------

Usage: gen_x_y_index.py <xdim> <ydim>
gen_x_y_index.py <raster>

Options:
-h | --help         Show this screen
<xdim>              X dimension of the indexing
<ydim>              Y dimension of the indexing
<raster>            A reference raster for dimension extracting
"""

import docopt
import numpy as np
from osgeo import gdal
import os


def fetch_raster_dim(raster):
    ds = gdal.Open(raster)
    return ds.RasterXSize, ds.RasterYSize


def save_int_raster(data, target):
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(target, data.shape[1], data.shape[0], 1, gdal.GDT_Int32)
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    ds.FlushCache()
    print("Saved:", target)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    x_dim, y_dim = arguments.get("<xdim>", None), arguments.get("<ydim>", None)
    raster = arguments.get("raster", None)

    if raster is not None:
        x_dim, y_dim = fetch_raster_dim(raster)
    else:
        x_dim, y_dim = int(x_dim), int(y_dim)

    x_out = "x_index_{}_{}.tif".format(x_dim, y_dim)
    y_out = "y_index_{}_{}.tif".format(x_dim, y_dim)

    if os.path.isfile(x_out) or os.path.isfile(y_out):
        raise FileExistsError('index file already exists, exit now')

    y_index, x_index = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing='ij')

    save_int_raster(x_index, x_out)
    save_int_raster(y_index, y_out)
