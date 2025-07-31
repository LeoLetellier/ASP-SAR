#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ra2aa.py

range-azimuth to amplitude-aspect

Usage: ra2aa.py --range=<range> --azimuth=<azimuth> [--incidence=<incidence>] --heading=<heading> --output=<output> [-d | -a] [--r_res=<r_res>] [--a_res=<a_res>]
ra2aa.py -h | --help

Options:
-h | --help             Show this screen
"""

import docopt
from osgeo import gdal
import numpy as np


def open_gdal(path):
    ds = gdal.Open(path)
    bd = ds.GetRasterBand(1)
    ndv = bd.GetNoDataValue()
    array = bd.ReadAsArray()
    if ndv is not None and ndv != np.nan:
        array[array == ndv] = np.nan
    return array

def save_gdal(data, target, template):
    ds_template = gdal.Open(template)
    ds = gdal.GetDriverByName('GTiff').CreateCopy(target, ds_template)
    band = ds.GetRasterBand(1)
    ndv = band.GetNoDataValue()
    if ndv is not None:
        data[data == np.nan] = ndv
    band.WriteArray(data)
    ds.FlushCache()


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    range_file = arguments["--range"]
    azimuth_file = arguments["--azimuth"]
    incidence_file = arguments["--incidence"]
    heading_value = arguments["--heading"]
    output_prefix = arguments["--output"]
    descending = arguments["-d"]
    ascending = arguments["-a"]
    r_res = arguments["--r_res"]
    a_res = arguments["--a_res"]

    range_data = open_gdal(range_file)
    azimuth_data = open_gdal(azimuth_file)

    if r_res is not None:
        range_data /= float(r_res)
    if a_res is not None:
        azimuth_data /= float(a_res)

    heading_value = np.mod(float(heading_value), 360)
    if heading_value > 180:
        heading_value -= 180

    if descending:
        print("Data in descending geometry: flipping range")
        range_data = -range_data
    if ascending:
        print("Data in ascending geometry: flipping azimuth")
        azimuth_data = -azimuth_data

    amplitude_data = np.sqrt(np.square(range_data) + np.square(azimuth_data))
    aspect_data = np.rad2deg(np.arctan2(azimuth_data, range_data))
    aspect_data = np.mod((aspect_data + 90) - heading_value, 360)

    save_gdal(amplitude_data, output_prefix + "_amplitude.tif", range_file)
    save_gdal(aspect_data, output_prefix + "_aspect.tif", range_file)

    print("All done.")
