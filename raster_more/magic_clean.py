#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
magic_clean.py
------------

Usage: magic_clean.py <infile> <outfile> [--vmin=<vmin> --vmax=<vmax> --pmin=<pmin> --pmax=<pmax> --mode=<mode>]

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
    vmin = float(arguments["--vmin"]) if arguments["--vmin"] is not None else None
    vmax = float(arguments["--vmax"]) if arguments["--vmax"] is not None else None
    pmin = float(arguments["--pmin"]) if arguments["--pmin"] is not None else None
    pmax = float(arguments["--pmax"]) if arguments["--pmax"] is not None else None
    mode = arguments["--mode"] if arguments["--mode"] is not None else "nan"

    datas = gdal_open(infile, supp_ndv=None)
    for d in datas:
        if vmin is not None:
            d[d < vmin] = np.nan
        if vmax is not None:
            d[d > vmax] = np.nan
    ds = gdal.Open(infile)
    band = ds.GetRasterBand(1)
    ndv = band.GetNoDataValue()
    gdal_save(outfile, infile, datas, ndv)
