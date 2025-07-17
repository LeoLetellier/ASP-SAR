#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
texture_enhancement.py
--------------
Prepare the directory structure for further processing. Link all ALL2GIF results in the given destination dir.

Usage: texture_enhancement.py <input> <output> <mode>
texture_enhancement.py -h | --help

Options:
-h | --help         Show this screen
<input>
<output>
<mode>

"""

import docopt
from skimage import exposure
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from osgeo import gdal
import os


def input_to_array(path):
    """ Open a GeoTiff and retrieve the data into a ndarray
    """
    ds = gdal.Open(path)
    # print("Opened ", path)
    ndv = ds.GetRasterBand(1).GetNoDataValue()
    array = ds.GetRasterBand(1).ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)
    if ndv and ndv != np.nan:
        array[array == ndv] = np.nan
    return array


def output_to_raster(array, path, template):
    """ Export the outputs to a raster, using the template of one of the input rasters
    """
    ds_template = gdal.Open(template)
    
    ds = gdal.GetDriverByName('GTiff').CreateCopy(path, ds_template)
    band = ds.GetRasterBand(1)
    band.WriteArray(array)
    ds.FlushCache()
    print("Saved:", path)


def texture(data, mode):
    p2, p98 = np.percentile(data, (2, 98))
    data[data <= 0] = np.nan
    if mode == 'rescale_intensity':
        data = exposure.rescale_intensity(data, in_range=(p2, p98))
    elif mode == 'equalize_hist':
        data = exposure.equalize_hist(data)
    elif mode =='equalize_adapthist':
        data = data / np.nanmax(data)
        data = exposure.equalize_adapthist(data, clip_limit=0.03)
    elif mode == 'entropy':
        data = data / np.nanmax(data)
        data = entropy(data, disk(3))
    elif mode == 'log':
        data[data>0] = np.log(data[data>0])
    else:
        raise ValueError('mode {} is not valid'.format(mode))
    return data


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    input = arguments["<input>"]
    output = arguments["<output>"]
    mode = arguments["<mode>"]

    data = input_to_array(input)
    data = texture(data, mode)
    output_to_raster(data, output, input)

    print("All done.")
