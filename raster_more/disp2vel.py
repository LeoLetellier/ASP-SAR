#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
disp2vel.py
--------------
convert displacement to velocity

Usage: disp2vel.py <raster> <output> [--res=<res>] --time=<time> [--band=<band>] [--flip]
disp2vel.py -h | --help

Options:
-h | --help         Show this screen
<raster>            Displacement raster
<output>            Velocity output raster
--res               Resolution in m
--time              Time between acquisition in year
--band              Band number
--flip              Flip the velocity sign (for radar acquisition D -> flip horz, V -> flip Vert)

"""

import docopt
import numpy as np
from osgeo import gdal


def open_raster(path, band=1, crop=None):
    ds = gdal.Open(path)
    band = ds.GetRasterBand(band)
    ndv = band.GetNoDataValue()
    if crop is None:
        crop = [0, 0, band.XSize, band.YSize]
    array = band.ReadAsArray(crop[0], crop[1], crop[2], crop[3])
    if ndv is not None and ndv != np.nan:
        array[array == ndv] = np.nan
    return array


def copy_raster(raster, output, band=1):
    ds_raster = gdal.Open(raster)
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(output, ds_raster, band)


def disp2vel(output, res, time, flip=False):
    ds = gdal.OpenEx(output, gdal.OF_VERBOSE_ERROR|gdal.OF_UPDATE)
    band = ds.GetRasterBand(1)
    ndv = band.GetNoDataValue()

    for y in range(band.YSize):
        line = open_raster(output, crop=[0, y, band.XSize, 1])
        new_line = line * res / time
        if flip:
            new_line = -new_line
        new_line[new_line == np.nan] = ndv
        band.WriteArray(new_line, 0, y)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    raster = arguments["<raster>"]
    output = arguments["<output>"]
    res = arguments["--res"]
    time = arguments["--time"]
    band = arguments["--band"]
    flip = arguments["--flip"]
    if res is not None:
        res = float(res)
    else:
        res = 1
    time = float(time)
    if band is not None:
        band = int(band)
    else:
        band=1

    print("copy raster...")
    copy_raster(raster, output, band=band)

    print("convert to velocity...")
    disp2vel(output, res, time, flip=flip)

