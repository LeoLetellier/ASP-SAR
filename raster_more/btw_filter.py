#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
btw_filter.py

Usage: btw_filter.py <infile> [<cutfreq>] [--hf]
btw_filter.py -h | --help

Options:
-h | --help             Show this screen
--infile                Path to the raster file to filter
--kernel                Size of the filter kernel (can be one nb "3" or array like "2 3")
"""


import numpy as np
from osgeo import gdal
from skimage.filters import butterworth
import os
import docopt


def open_tif(path):
    if not os.path.exists(path):
        print("infile:", path)
        raise FileNotFoundError
    ds = gdal.OpenEx(path, allowed_drivers=["GTiff"])
    ds_band = ds.GetRasterBand(1)
    ndv = ds_band.GetNoDataValue()
    values = ds_band.ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)
    if ndv and ndv != np.nan:
        values[values == ndv] = np.nan
    ncols, nlines = ds.RasterYSize, ds.RasterXSize
    proj = ds.GetProjection()
    geotransform = ds.GetGeoTransform()
    return (values, ncols, nlines, proj, geotransform, ndv)


def save_tif(data, path, ncols, nlines, proj, geotransform, ndv):
    drv = gdal.GetDriverByName('GTiff')
    dst_ds = drv.Create(path, ncols, nlines, 1, gdal.GDT_Float32)
    dst_band = dst_ds.GetRasterBand(1)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(proj)
    data[data == np.nan] = ndv
    dst_band.WriteArray(data)
    dst_ds.FlushCache()


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    infile = arguments["<infile>"]
    cut = arguments["<cutfreq>"]
    
    (tif, nlines, ncols, proj, geotr, ndv) = open_tif(infile)
    
    if cut is not None:
        cut = float(cut)
        filtered = butterworth(tif, cutoff_frequency_ratio=cut, high_pass=arguments["--hf"])
    else:
        filtered = butterworth(tif, high_pass=arguments["--hf"])

    save_tif(filtered, os.path.splitext(infile)[0] + "_btw.tif", 
             ncols, nlines, proj, geotr, ndv)
    
    print("done!")
    