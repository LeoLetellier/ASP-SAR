#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r4_to_gdal.py

Conversion between r4 and gdal GTiff

Usage: r4_to_gdal.py <input> <output> [<ncol> <nrow>] [--r]
r4_to_gdal.py -h | --help

Options:
-h | --help             Show this screen
"""

import numpy as np
from osgeo import gdal
import docopt
import os

def open_r4(path, dim=None):
    if dim is not None:
        ncol, nrow = dim
    else:
        # TODO
        pass
    with open(path, 'r') as infile:
        data = np.fromfile(infile, dtype=np.float32)[:nrow * ncol * 1].reshape((nrow, ncol))
    return data

def open_gdal(path):
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    return data

def save_r4(path, data):
    with open(path, 'wb') as outfile:
        data.flatten().astype('float32').tofile(outfile)
    ncol, nrow = data.shape[1], data.shape[0]
    with open(path + '.rsc', "w") as rsc_file:
        rsc_file.write("""\
    WIDTH                 %d
    FILE_LENGTH           %d
    XMIN                  0
    XMAX                  %d
    YMIN                  0
    YMAX                  %d""" % (ncol, nrow, ncol-1, nrow-1))

def save_gdal(path, data):
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(path, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    ds.FlushCache()


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    input = arguments["<input>"]
    output = arguments["<output>"]
    ncol = arguments["<ncol>"]
    nrow = arguments["<nrow>"]
    r4_to_gdal = True
    reverse = arguments["--r"]

    if ncol is not None:
        ncol = int(ncol)
    if nrow is not None:
        nrow = int(nrow)

    input_ext = os.path.splitext(input)[1]
    output_ext = os.path.splitext(output)[1]


    # if input_ext == '.r4':
    #     r4 = input
    #     gd = output
    # elif output_ext == '.r4':
    #     r4 = output
    #     gd = input
    #     r4_to_gdal = False
    # elif input_ext is None:
    #     r4 = input
    #     gd = output
    # else:
    #     print("guessing r4")
    #     try:
    #         gdal.Open(input)
    #     except:
    #         try:
    #             gdal.Open(output)
    #         except:
    #             raise ValueError('no gdal raster')
    #         else:
    #             gd = output
    #             r4 = input
    #     else:
    #         gd = input
    #         r4 = output
    #         r4_to_gdal = False
    
    if not reverse:
        input_data = open_r4(input, dim=(ncol, nrow))
        save_gdal(output, input_data)
    else:
        input_data = open_gdal(output)
        save_r4(input, input_data)
