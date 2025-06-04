#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r_stats.py
--------------
Compute stack statistics on multiple images for each pixel.

Usage: r_stats.py --infile=<infile> --outfile=<outfile>
r_stats.py -h | --help

Options:
-h | --help             Show this screen
--infile                Path to the file with one image per line
--outfile
"""


import numpy as np
from osgeo import gdal
import os
import docopt
import matplotlib.pyplot as plt
from scipy import ndimage


def open_raster_crop(path):
    ds = gdal.Open(path)
    bd = ds.GetRasterBand(1)
    ndv = bd.GetNoDataValue()
    array = bd.ReadAsArray()
    if ndv is not None and ndv != np.nan:
        array[array == ndv] = np.nan
    return array


def moving_std(data):
    mv_std = ndimage.generic_filter(data, np.nanstd, size=(11, 11), mode='nearest')
    return mv_std


def save_out(inpath, outpath, data):
    ds_template = gdal.Open(inpath)
    
    ds = gdal.GetDriverByName('GTiff').CreateCopy(outpath, ds_template)
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    ds.FlushCache()


if __name__ == "__main__":
    args = docopt.docopt(__doc__)

    infile = args["--infile"]
    outfile = args["--outfile"]

    data = open_raster_crop(infile)
    mv_std = moving_std(data)
    save_out(infile, outfile, mv_std)
