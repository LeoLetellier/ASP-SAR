#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tsai.py
--------------
Time Series Activity Index

compute a cumulative index over the cube with (abs(di - di-1))^2 highlighting high displacement areas

Usage: tsai.py <cube> [<stable_area>]
tsai.py -h | --help

Options:
-h | --help         Show this screen
cube                Cumulative Displacement cube
stable_area         Coordinates of a stable area for estimating measurement noise

"""

import numpy as np
from osgeo import gdal
from docopt import docopt
import os
from tqdm import tqdm
gdal.UseExceptions()


def open_gdal(file, band=1):
    ds = gdal.Open(file)
    band = ds.GetRasterBand(band)
    data = band.ReadAsArray()
    ndv = band.GetNoDataValue()
    if ndv is not None and ndv != 0:
        data[data==ndv] = 0
    return data


def save_gdal(outfile, data, template):
    ds_template = gdal.Open(template)
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(outfile, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform(ds_template.GetGeoTransform())
    
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    ds.FlushCache()


def stack_index(cube, stable_area=None):
    ds_cube = gdal.Open(cube)
    nb_band = ds_cube.RasterCount
    size = [ds_cube.RasterXSize, ds_cube.RasterYSize]
    stack = np.zeros(shape=(size[1], size[0]), dtype=float)
    data2 = open_gdal(cube, band=1)
    # for b in range(2, 10):
    for b in tqdm(range(2, nb_band + 1)):
        data1 = data2
        data2 = open_gdal(cube, band=b)
        diff = np.abs(data1 - data2)
        if stable_area is not None:
            sigma = np.nanmedian(diff[stable_area[2]:stable_area[3], stable_area[0], stable_area[1]])
        else:
            sigma = 1
        stack += np.square(diff / sigma)
    return stack


if __name__ == "__main__":
    arguments = docopt(__doc__)
    cube = arguments["<cube>"]
    stable_area = arguments["<stable_area>"]
    if stable_area is not None:
        stable_area = [int(k) for k in stable_area.split(",", 4)]
    
    stack = stack_index(cube, stable_area)

    save_gdal(os.path.splitext(cube)[0] + "_tsai.tif", stack, cube)
