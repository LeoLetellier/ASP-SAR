#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r_ps_select.py
--------------
Select the persistent scatterer pixels base on the D_A (std/mean) properties.

Usage: r_ps_select.py --infile=<infile> --outfile=<outfile> --da=<da> [--threshold=<threshold>]
r_ps_select.py -h | --help

Options:
-h | --help             Show this screen
--infile                Path to the file with one image per line
--outfile               Path to the output file
--da                    D_A file
--threshold             D_A threshold
"""


import numpy as np
from osgeo import gdal
import os
import docopt


def da_removal(ampl, da, threshold=None):
    """ remove px with da sup threshold
    """
    threshold = threshold if threshold else 0.25
    ampl[da > threshold] = np.nan
    return ampl


def input_to_array(path):
    """ Open a GeoTiff and retrieve the data into a ndarray
    """
    ds = gdal.Open(path)
    # print("Opened ", path)
    ndv = ds.GetRasterBand(1).GetNoDataValue()
    array = ds.GetRasterBand(1).ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)
    if ndv and ndv != np.nan:
        array[array == ndv] = np.nan
    return array


def output_to_raster(array, path, template):
    """ Export the outputs to a raster, using the template of one of the input rasters
    """
    ds_template = gdal.Open(template)
    
    ds = gdal.GetDriverByName('GTiff').CreateCopy(path, ds_template)
    band = ds.GetRasterBand(1)
    ndv = band.GetNoDataValue()
    if ndv != np.nan:
        array[array == np.nan] = ndv
    band.WriteArray(array)
    ds.FlushCache()
    print("Saved:", path)


if __name__ == "__main__":
    args = docopt.docopt(__doc__)

    infile = args["--infile"]
    outfile = args["--outfile"]
    da = args["--da"]
    thresh = args["--threshold"]

    ampl = input_to_array(infile)
    da = input_to_array(da)
    ampl = da_removal(ampl, da, float(thresh))
    output_to_raster(ampl, outfile, infile)
