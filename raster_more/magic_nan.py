#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
magic_nan.py
------------

Usage: magic_nan.py <infile> <outfile> [--input-ndv=<input-ndv>] [--output-ndv=<output-ndv>]

Options:
-h | --help         Show this screen
"""

import docopt
from osgeo import gdal
import numpy as np


def gdal_open(raster, supp_ndv):
    ds = gdal.Open(raster)
    nb_bd = ds.RasterCount
    datas = []
    for b in range(nb_bd):
        band = ds.GetRasterBand(b + 1)
        data = band.ReadAsArray()
        ndv = band.GetNoDataValue()
        if ndv is not None and ndv != np.nan:
            data[data==ndv] = np.nan
        if supp_ndv is not None and supp_ndv != np.nan:
            data[data==supp_ndv] = np.nan
        datas.append(data)
    return datas


def gdal_save(outfile, template, datas, ndv):
    ds_template = gdal.Open(template)
    ds = gdal.GetDriverByName("GTiff").CreateCopy(outfile, ds_template)
    nb_bd = ds.RasterCount
    for b in range(nb_bd):
        band = ds.GetRasterBand(b + 1)
        data = np.nan_to_num(datas[b], nan=ndv)
        band.WriteArray(data)
        band.SetNoDataValue(ndv)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    infile = arguments["<infile>"]
    outfile = arguments["<outfile>"]
    supp_ndv = arguments["--input-ndv"]
    if supp_ndv is not None:
        supp_ndv = float(supp_ndv)
    output_ndv = arguments["--output-ndv"]
    output_ndv = float(output_ndv) if output_ndv is not None else np.nan

    datas = gdal_open(infile, supp_ndv=supp_ndv)
    gdal_save(outfile, infile, datas, output_ndv)
