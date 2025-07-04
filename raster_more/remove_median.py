#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
remove_median.py
--------------
Prepare the directory structure for further processing. Link all ALL2GIF results in the given destination dir.

Usage: remove_median.py <raster> [--crop=<crop>] <output> [--band=<band>] [--plot]
remove_median.py -h | --help

Options:
-h | --help         Show this screen
<raster>            Raster to correct
<output>            Corrected output raster
<crop>              Area for median estimation
<band>              Band number

"""

import docopt
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

XMAXSIZE, YMAXSIZE = 2048, 2048

def open_raster(path, band=1, maxsize=False, crop=None):
    ds = gdal.Open(path)
    band = ds.GetRasterBand(band)
    ndv = band.GetNoDataValue()
    if crop is None:
        crop = [0, 0, band.XSize, band.YSize]
    if not maxsize:
        array = band.ReadAsArray(crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1])
    else:
        array = band.ReadAsArray(crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1], XMAXSIZE, YMAXSIZE, resample_alg=gdal.GRIORA_Bilinear)
    if ndv is not None and ndv != np.nan:
        array[array == ndv] = np.nan
    return array


def copy_raster(raster, output,band=1):
    ds_raster = gdal.Open(raster)
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(output, ds_raster, band)


def median_raster(raster, crop=None, band=1, plot=False):
    r4med = open_raster(raster, band=band, maxsize=True, crop=crop)
    if plot:
        plt.imshow(r4med)
        plt.show()
    return np.nanmedian(r4med.flatten())


def substract_median(median, output):
    ds = gdal.OpenEx(output, gdal.OF_VERBOSE_ERROR|gdal.OF_UPDATE)
    band = ds.GetRasterBand(1)
    ndv = band.GetNoDataValue()

    for y in range(band.YSize):
        line = open_raster(output, crop=[0, y, band.XSize, 1 + y])
        new_line = line - median
        new_line[new_line == np.nan] = ndv
        band.WriteArray(new_line, 0, y)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    raster = arguments["<raster>"]
    output = arguments["<output>"]
    crop = arguments["--crop"]
    band = arguments["--band"]
    plot = arguments["--plot"]
    if band is not None:
        band = int(band)
    else:
        band = 1
    if crop is not None:
        crop = [int(c) for c in crop.split(",")]

    print("Compute median...")
    median = median_raster(raster, crop=crop, band=band, plot=plot)
    print("median =", median)

    print("Copy raster...")
    copy_raster(raster, output)

    print("Substract median and save...")
    substract_median(median, output)
