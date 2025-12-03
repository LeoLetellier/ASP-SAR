#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rsubview.py
-------------
Create a view of a raster image

Usage: rsubview.py <infile> [--band=<band> --ndv=<ndv> --crop=<crop> --cpt=<cpt> --vmin=<vmin> --vmax=<vmax> --nodecorator --noshow --outfile=<outfile> --text=<text>]


Options:
-h --help               Show this screen.
<infile>                Raster to be displayed
"""

import docopt
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
gdal.UseExceptions()


def open_gdal(file, band, supp_ndv=None, crop=None):
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


def plot_image(data, cpt=None, vmin=None, vmax=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1)
    cpt = "Greys_r" if cpt is None else None
    if vmin is None and vmax is None:
        vmin, vmax = np.percentile(data, (2, 98))
    elif vmin is None:
        if vmax > 0:
            vmin = -vmax
        else:
            vmin = 0
    elif vmax is None:
        if vmin < 0:
            vmax = -vmin
        else:
            vmax = 0
    
    hax = ax.imshow(data, cmap=cpt, vmin=vmin, vmax=vmax)
    plt.axis('off')
    divider = make_axes_locatable(ax)
    c = divider.append_axes("right", size='5%', pad=0.05)
    plt.colorbar(hax, cax=c)
    return fig, ax


def decorator(fig, ax, text):
    ax.text(25, 85, text, backgroundcolor='black', color='white', figure=fig, family='sans-serif', size=30)
    return fig, ax


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    infile = arguments["<infile>"]
    band = arguments["--band"]
    band = 1 if band is None else int(band)
    ndv = arguments["--ndv"]
    ndv = None if ndv is None else float(ndv)
    crop = arguments["--crop"]
    if crop is not None:
        crop = [int(k) for k in crop.split(",", 4)]
    cpt = arguments['--cpt']
    vmin = arguments["--vmin"]
    vmax = arguments["--vmax"]
    nodecorator = arguments["--nodecorator"]
    noshow = arguments["--noshow"]
    outfile = arguments["--outfile"]
    text = arguments["--text"]

    data = open_gdal(infile, band, ndv, crop)

    fig, ax = plot_image(data, cpt, vmin, vmax)

    if not nodecorator:
        fig, ax = decorator(fig, ax, text)
    
    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile)
    
    if not noshow:
        plt.show()
