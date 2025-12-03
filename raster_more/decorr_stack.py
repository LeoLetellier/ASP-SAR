#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decorr_stack.py
------------

Usage: decorr_stack.py <aspsar> [-f]

Options:
-h | --help         Show this screen
"""

import docopt
import os
from osgeo import gdal
import numpy as np
from tqdm import tqdm

gdal.UseExceptions()


def open_gdal(file, band=1, supp_ndv=None, crop=None):
    ds = gdal.Open(file)
    band = ds.GetRasterBand(band)
    ndv = band.GetNoDataValue()
    if crop is None:
        data = band.ReadAsArray()
    else:
        data = band.ReadAsArray(crop[0], crop[2], crop[1] - crop[0], crop[3] - crop[2])
    if ndv is not None and ndv != np.nan:
        data[data==ndv] = np.nan
    if supp_ndv is not None and supp_ndv != np.nan:
        data[data==supp_ndv] = np.nan
    return data


def open_r4(file, dim=False):
    lines = open(file + '.rsc').read().strip().split('\n')
    x_dim, y_dim = None, None
    for l in lines:
        if 'WIDTH' in l:
            x_dim = int(''.join(filter(str.isdigit, l)))
        elif 'FILE_LENGTH' in l:
            y_dim = int(''.join(filter(str.isdigit, l)))
        if x_dim is not None and y_dim is not None:
            break
    if x_dim is None and y_dim is None:
        raise ValueError('could not completely read', file + '.rsc')
    data = np.fromfile(file, dtype=np.float32)[:y_dim * x_dim].reshape((y_dim, x_dim))
    return data


def write_gdal(data, path, template_ds, band=1):
    ds = gdal.GetDriverByName('GTiff').CreateCopy(path, template_ds)
    band = ds.GetRasterBand(band)
    band.WriteArray(data)
    ds.FlushCache()


def write_tiff(data, path):
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(path, data.shape[1], data.shape[0], 1, gdal.GDT_Int16)
    band = ds.GetRasterBand(1)
    band.WriteArray(data.astype(np.int16))
    ds.FlushCache()


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    folder = arguments["<aspsar>"]
    force = arguments["-f"]

    export_h = os.path.join(folder, 'EXPORT', 'H')
    files = [os.path.join(export_h, k) for k in list(filter(lambda f: os.path.splitext(f)[1] == ".r4", os.listdir(export_h)))]

    data1 = open_r4(files[0])
    
    decorr_px = np.zeros(shape=data1.shape)

    for f in tqdm(files):
        amp = open_r4(f)
        decorr_px += np.isnan(amp)

    if not os.path.isfile(os.path.join(folder, 'EXPORT', 'DECORR_PIXEL_' + ".tif")) or force:
        write_tiff(decorr_px, os.path.join(folder, 'EXPORT', 'DECORR_PIXEL_' + ".tif"))
