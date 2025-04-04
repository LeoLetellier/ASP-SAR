#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
r_stats.py
--------------
Compute stack statistics on multiple images for each pixel.

Usage: r_stats.py --infile=<infile> --prefix=<prefix>
r_stats.py -h | --help

Options:
-h | --help             Show this screen
--infile                Path to the file with one image per line
--prefix                Prefix of the outputs
"""


import numpy as np
from osgeo import gdal
import os
import docopt


def r_stats(inputs, input_to_array):
    """ Compute the mean, squared standard deviation, amplitude deviation from mutliple arrays
    """
    shape = input_to_array(inputs[0]).shape
    amp = np.zeros(shape=shape)
    amp2 = np.zeros(shape=shape)
    valid = np.zeros(shape=shape)

    for i in inputs:
        # print(i)
        array = input_to_array(i)
        amp += array
        amp2 += np.square(array)
        valid += np.where(np.isnan(array), 0, 1)

    mean = np.full(shape=shape, fill_value=np.nan)
    std2 = np.full(shape=shape, fill_value=np.nan)
    da = np.full(shape=shape, fill_value=np.nan)

    # Use masking to avoid /0
    # mean is sum / px_nb
    mean[valid!=0] = amp[valid!=0] / valid[valid!=0]
    # std^2 is sum(x^2)/N - (sum(x)/N)^2
    std2[valid!=0] = amp2[valid!=0] / valid[valid!=0] - np.square(mean[valid!=0])
    # da is std / mean
    da[mean!=0] = np.sqrt(std2[mean!=0]) / mean[mean!=0]
    return mean, std2, da


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


def output_to_raster(mean, std2, da, prefix, template):
    """ Export the outputs to a raster, using the template of one of the input rasters
    """
    files = ["_mean.tif", "_std2.tif", "_da.tif"]
    ds_template = gdal.Open(template)

    for i, r in enumerate([mean, std2, da]):
        ds = gdal.GetDriverByName('GTiff').CreateCopy(prefix + files[i], ds_template)
        band = ds.GetRasterBand(1)
        band.WriteArray(r)
        ds.FlushCache()
        print("Saved:", prefix + files[i])


def parse_input_file(path):
    """ Parse the input file into separate paths, one for each line
    """
    with open(path, 'r') as infile:
        raster_paths = infile.read().split('\n')
    raster_paths = [r.strip() for r in raster_paths if r.strip()]
    return raster_paths


if __name__ == "__main__":
    args = docopt.docopt(__doc__)

    infile = args["--infile"]
    prefix = args["--prefix"]

    raster_paths = parse_input_file(infile)
    mean, std2, da = r_stats(raster_paths, input_to_array)
    output_to_raster(mean, std2, da, prefix, raster_paths[0])