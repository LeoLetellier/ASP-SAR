#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r_clean_range.py
--------------


Usage: r_clean_range.py --infile=<infile> --outfile=<outfile> [--vmin=<vmin> --vmax=<vmax>]
r_clean_range.py -h | --help

Options:
-h | --help             Show this screen
--infile                Path to the file with one image per line
--outfile               Path to the output file
"""


import numpy as np
from osgeo import gdal
import os
import docopt


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
    vmin = args["--vmin"]
    vmax = args["--vmax"]

    data = input_to_array(infile)
    if vmin is not None:
        data[data < float(vmin)] = np.nan
    if vmax is not None:
        data[data > float(vmax)] = np.nan
    output_to_raster(data, outfile, infile)
