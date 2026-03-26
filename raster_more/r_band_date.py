#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r_band_date.py
------------

Usage: r_band_date.py <raster> <dates> [--skiprows=<r> --col=<c>]

Options:
-h | --help         Show this screen
<raster>            Raster to update
<dates>             List of dates
"""

import docopt
from osgeo import gdal
import numpy as np

gdal.UseExceptions()


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    ds = gdal.Open(arguments["<raster>"], gdal.GA_Update)
    skiprows = int(arguments["--skiprows"]) if arguments["--skiprows"] is not None else 0
    usecols = int(arguments["--col"]) if arguments["--col"] is not None else 0
    dates = np.loadtxt(arguments["<dates>"], skiprows=skiprows, usecols=(usecols), dtype=str)

    if len(dates) != ds.RasterCount:
        raise ValueError("Raster has {} bands but {} dates were found".format(ds.RasterCount, len(dates)))

    for b in range(1, 1 + ds.RasterCount):
        band = ds.GetRasterBand(b)
        band.SetDescription(dates[b - 1])
        print("Set description: '{}' for band {}".format(dates[b - 1], b))
    
    ds.FlushCache()
    ds = None
    print("done")
