#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raster_nan.py

Usage: raster_nan.py <infile> <outfile> [<ndv>]
raster_nan.py -h | --help

Options:
-h | --help             Show this screen
"""


import numpy as np
from osgeo import gdal
import os
import docopt


def open_gdal(file, supp_ndv=None):
    ds = gdal.Open(file)
    band = ds.GetRasterBand(1)
    ndv = band.GetNoDataValue()
    data = band.ReadAsArray()
    if ndv is not None and ndv != np.nan:
        data[data==ndv] = np.nan
    if supp_ndv is not None and supp_ndv != np.nan:
        data[data==supp_ndv] = np.nan
    return data

def save_gdal(file, template, data):
    ds_template = gdal.Open(template)
    ds = gdal.GetDriverByName('GTiff').CreateCopy(file, ds_template)
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(np.nan)
    ds.FlushCache()


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    infile = arguments["<infile>"]
    outfile = arguments["<outfile>"]
    ndv = arguments['<ndv>']
    # in_nan = arguments["in-nan"]
    # out_nan = arguments["out-nan"]

    # if in_nan is not None:
    #     if in_nan in ['nan', 'NaN', 'NAN']:
    #         in_nan = np.nan
    #     else:
    #         in_nan = float(in_nan)
    # if out_nan is not None:
    #     if out_nan in ['nan', 'NaN', 'NAN']:
    #         out_nan = np.nan
    #     else:
    #         out_nan = float(out_nan)

    if ndv is not None:
        ndv = float(ndv)

    data = open_gdal(infile, ndv)
    save_gdal(outfile, infile, data)

