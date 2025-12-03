#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
value_classifier.py
--------------
Value Classifier

separate each set of data between cut values into a subclass

Usage: value_classifier.py <infile> <outfile> <cut_values>
value_classifier.py -h | --help

Options:
-h | --help         Show this screen

"""

import numpy as np
from osgeo import gdal
from docopt import docopt
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


def classify_values(data, values):
    values = values
    class_values = [k for k in range(len(values) + 1)]
    class_data = np.zeros(shape=data.shape)

    for v in range(len(values)):
        class_data[data>values[v]] = class_values[v + 1]
    
    return class_data


if __name__ == "__main__":
    arguments = docopt(__doc__)
    infile = arguments["<infile>"]
    outfile = arguments["<outfile>"]
    values = arguments["<cut_values>"]
    values = [float(k) for k in values.split(',')]

    data = open_gdal(infile)
    class_data = classify_values(data, values)

    save_gdal(outfile, class_data, infile)
