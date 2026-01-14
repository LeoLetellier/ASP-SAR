#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r4_hdr.py

Generate a GDAL header ".hdr" for a REAL4 file

Usage: r4_hdr.py <infile> [<lectfile>]
r4_hdr.py -h | --help

Options:
-h | --help             Show this screen
"""

import os
from osgeo import gdal
import docopt
gdal.UseExceptions()


def read_lectfile(lectfile, ndates=None):
    ncol, nlign = list(map(int, open(lectfile).readline().split(None, 2)[0:2]))
    if ndates is None:
        try:
            ndates = int(open(lectfile).readlines(4)[-1])
        except:
            ndates = 1
    return ncol, nlign, ndates


def generate_header(raster, dims):
    content = """ENVI
samples =   {}
lines =   {}
bands =  {}
header offset = 0
data type = 4
interleave = bip
byte order = 0""".format(dims[0], dims[1], dims[2])
    with open(os.path.splitext(raster)[0] + ".hdr", 'w') as headerfile:
        headerfile.write(content)


def ensure_gdal_header(file, lectfile=None, ndates=None):
    try:
        gdal.Open(file)
    except:
        if lectfile is None:
            lectfile = "lect.in"
        elif type(lectfile) is str:
            if os.path.isfile(lectfile):
                dims = read_lectfile(lectfile, ndates)
                generate_header(file, dims)
        elif type(lectfile) is list:
            generate_header(file, lectfile)
        else:
            raise ValueError('Cannot read raster:', file)
        

if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    infile = arguments["<infile>"]
    lectfile = arguments["<lectfile>"]
    if "," in lectfile:
        lectfile = [float(k) for k in lectfile.split(",")]
    ensure_gdal_header(infile, lectfile=lectfile)
