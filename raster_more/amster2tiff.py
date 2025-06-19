#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amster2tiff.py
------------
Converts AMSTer raster file to GTiff.

Usage: amster2tiff.py --infile=<infile> --outfile=<outfile> (--params=<params> | <nrow> <ncol>) [--db]
amster2tiff.py -h | --help

Options:
-h | --help         Show this screen
--infile              Path to directory with linked data
--outfile
--params            InSARParameters file
--db                Convert to dB
"""

import numpy as np
from osgeo import gdal
from math import *
import docopt


def open_amster(file, nrow, ncol):
    array = np.fromfile(file, dtype=np.float32)
    return array[:nrow * ncol].reshape((nrow, ncol))


def open_params(param_file):
    with open(param_file, 'r') as pfile:
        lines = [''.join(l.strip().split('\t\t')[0]) for l in pfile.readlines()]
        jump_index = lines.index('/* -5- Interferometric products computation */')
        img_dim = lines[jump_index + 2: jump_index + 4]
    return (int(img_dim[1].strip()), int(img_dim[0].strip()))


def save_gtiff(array, outfile):
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(outfile, array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
    band = ds.GetRasterBand(1)
    band.WriteArray(array)
    return True


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    infile = arguments["--infile"]
    outfile = arguments["--outfile"]
    params = arguments["--params"]
    nrow = arguments["<nrow>"]
    ncol = arguments["<ncol>"]
    do_db = arguments["--db"]

    if params is not None:
        (nrow, ncol) = open_params(params)
    else:
        (nrow, ncol) = (int(nrow), int(ncol))
    array = open_amster(infile, nrow, ncol)
    if do_db:
        array = 10 * np.log10(array)
        # array = np.log(array)
    if save_gtiff(array, outfile):
        print("Saved {}".format(outfile))
