#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_ts.py
------------

Usage: figure_ts.py <nsbas> [--ref=<ref> --table=<table>]

Options:
-h | --help         Show this screen
"""

import docopt
import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

gdal.UseExceptions()


def read_pairs_file(file):
    pairs = np.loadtxt(file, skiprows=2, usecols=(0, 1), delimiter='\t', dtype=str)
    return pairs


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


def plot_raster(data, title, ax):
    vmin, vmax = np.nanpercentile(data, (2, 98))
    ims = ax.imshow(data, "Greys_r", interpolation="nearest", vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    c = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(ims, cax=c)
    ax.set_title(title)


def compute_mean_da(folder, subset):
    geotiff = os.path.join(folder, "GEOTIFF")
    rasters = [f for f in os.listdir(geotiff) if f[-len('.mod_log.tif'):] == ".mod_log.tif" and f[:8] in subset]
    shape = open_gdal(os.path.join(folder, 'GEOTIFF', rasters[0])).shape
    
    mean = np.zeros(shape=shape)
    sigma = np.zeros(shape=shape)
    
    for r in tqdm(rasters):
        data = open_gdal(os.path.join(folder, 'GEOTIFF', r))
        mean += data
        sigma += np.square(data)
    
    mean /= len(rasters)
    sigma = np.sqrt(sigma - np.square(mean))
    da = sigma / mean

    return mean, da


def savefig(path):
    plt.tight_layout()
    print("Saving:", path)
    plt.savefig(path, dpi=300)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    folder = arguments["<nsbas>"]
    ref = arguments["--ref"]
    if ref is not None:
        ref = [int(k) for k in ref.split(',', 4)]
    table = arguments["--table"]
    if table is None:
        table = os.path.join(folder, 'PAIRS', 'table_pairs.txt')

    
    
