#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore_pairs.py
------------

Usage: explore_pairs.py <aspsar> [--area=<area>]

Options:
-h | --help         Show this screen
"""

import docopt
from os import path as p
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

gdal.UseExceptions()


def read_pairs_file(file):
    pairs = np.loadtxt(file, skiprows=2, usecols=(1, 2), delimiter='\t', dtype=str)
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


def plot_geotiff(folder):
    print("Plot Geotiff")
    ampli_mean = p.join(folder, "GEOTIFF", "AMPLI_MEAN.tif")
    da = p.join(folder, "GEOTIFF", "AMPLI_DA.tif")
    fig = plt.figure(figsize=(20, 10))

    print("\tAMPLI_MEAN")
    ax1 = plt.subplot(121)
    data = open_gdal(ampli_mean)
    plot_raster(data, "AMPLI_MEAN", ax1)

    print("\tDa")
    ax2 = plt.subplot(122)
    data = open_gdal(da)
    plot_raster(data, "Da", ax2)


def savefig(path):
    plt.tight_layout()
    print("Saving:", path)
    plt.savefig(path, dpi=300)





if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    folder = arguments["<aspsar>"]
    area = arguments["--area"]
    if area is not None:
        area = [int(k) for k in area.split(",", 4)]

    pairs_file = p.join(folder, "PAIRS", "table_pairs.txt")
    pairs = read_pairs_file(pairs_file)

    plot_geotiff(folder)
    savefig(p.join(folder, "FIGURES", "geotiff.pdf"))



