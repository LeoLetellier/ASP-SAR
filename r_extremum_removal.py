#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r_extremum_removal.py
--------------
Compute stack statistics on multiple images for each pixel.

Usage: r_extremum_removal.py --infile=<infile> --outfile=<outfile> [--value=<value>]
r_extremum_removal.py -h | --help

Options:
-h | --help             Show this screen
--infile                Path to the file with one image per line
--outfile               Path to the output file
--value                 Value to replace extremum with
"""


import numpy as np
from osgeo import gdal
import os
import docopt


def r_extremum_removal(ampl, value=None):
    """ Compute the mean, squared standard deviation, amplitude deviation from mutliple arrays
    """
    value = value if value else np.nan
    threshold = 3.1 * np.std(ampl)
    ampl[np.abs(ampl) - np.nanmean(ampl) > threshold] = value
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
    band.WriteArray(array)
    ds.FlushCache()
    print("Saved:", path)


if __name__ == "__main__":
    args = docopt.docopt(__doc__)

    infile = args["--infile"]
    outfile = args["--outfile"]
    value = args["--value"]

    ampl = input_to_array(infile)
    ampl = r_extremum_removal(ampl, value)
    output_to_raster(ampl, outfile, infile)
