#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r_histo.py
--------------
Show the histogram of pixel values of multiple rasters

Usage: r_histo.py raster1 raster2 ...
r_histo.py -h | --help

Options:
-h | --help             Show this screen
"""


import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import sys


def r_histo(array, name=None):
    """ Compute the mean, squared standard deviation, amplitude deviation from mutliple arrays
    """
    hist_values, bin_edges = np.histogram(array[~np.isnan(array)].flatten(), bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, hist_values / np.sum(hist_values), label=name)


def input_to_array(path):
    """ Open a GeoTiff and retrieve the data into a ndarray
    """
    ds = gdal.Open(path)
    ndv = ds.GetRasterBand(1).GetNoDataValue()
    array = ds.GetRasterBand(1).ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)
    if ndv and ndv != np.nan:
        array[array == ndv] = np.nan
    return array


if __name__ == "__main__":
    args = sys.argv[1:]

    for infile in args:
        print("histo line for:", infile)
        array = input_to_array(infile)
        print("min: ", str(np.nanmin(array)), " ; max: ", str(np.nanmax(array)))
        r_histo(array, name=infile)

    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Pixel Amplitude')
    plt.legend()
    plt.show()
