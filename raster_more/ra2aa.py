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


def open_gdal(path, band=1):
    ds = gdal.Open(path)
    bd = ds.GetRasterBand(band)
    ndv = bd.GetNoDataValue()
    array = bd.ReadAsArray()
    if ndv is not None and ndv != np.nan:
        array[array == ndv] = np.nan
    return array


def copy_gdal(target, template):
    ds_template = gdal.Open(template)
    ds = gdal.GetDriverByName('GTiff').CreateCopy(target, ds_template)
    return ds


def save_gdal_band(ds, bd, data):
    band = ds.GetRasterBand(bd)
    ndv = band.GetNoDataValue()
    if ndv is not None:
        data[data == np.nan] = ndv
    band.WriteArray(data)
    ds.FlushCache()


def save_gdal(data, target, template):
    ds_template = gdal.Open(template)
    ds = gdal.GetDriverByName('GTiff').CreateCopy(target, ds_template)
    band = ds.GetRasterBand(1)
    ndv = band.GetNoDataValue()
    if ndv is not None:
        data[data == np.nan] = ndv
    band.WriteArray(data)
    ds.FlushCache()


def check_is_cube(range, azimuth):
    bd_range = gdal.Open(range).RasterCount
    bd_azimuth = gdal.Open(azimuth).RasterCount
    return bd_range == bd_azimuth and bd_range > 1


def xy_to_north(x, y, heading):
    aspect_data = np.pi/2 - np.arctan2(y, x) - np.deg2rad(heading)
    aspect_data[aspect_data < 0] = aspect_data[aspect_data < 0] + 2*np.pi
    return np.rad2deg(aspect_data)
    

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

    if descending:
        print("Descending: flipping range data")
    elif ascending:
        print("Ascending: flipping azimuth data")

    is_cube = check_is_cube(range_file, azimuth_file)

    bd_nb = gdal.Open(range_file).RasterCount if is_cube else 1
    target_amp = copy_gdal(output_prefix + "_amplitude.tif", range_file)
    target_asp = copy_gdal(output_prefix + "_aspect.tif", range_file)

    heading_value = np.mod(float(heading_value), 360)
    if heading_value > 180:
        heading_value -= 180

    x = [1, 1, -1, -1]
    y = [1, -1, 1, -1]

    for b in range(1, bd_nb + 1):
        print(f"\tband {b}")
        range_data = open_gdal(range_file, b)
        # azimuth disparity is up to bottom where axis is bottom to up
        azimuth_data = -open_gdal(azimuth_file, b)

        if r_res is not None:
            range_data /= float(r_res)
        if a_res is not None:
            azimuth_data /= float(a_res)

        if descending:
            range_data = -range_data
        if ascending:
            azimuth_data = -azimuth_data

        amplitude_data = np.sqrt(np.square(range_data) + np.square(azimuth_data))
        aspect_data = xy_to_north(range_data, azimuth_data, heading_value)
        print(xy_to_north(x, y, heading_value))

        save_gdal_band(target_amp, b, amplitude_data)
        save_gdal_band(target_asp, b, aspect_data)

    print("All done.")
