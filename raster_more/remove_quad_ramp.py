#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remove_quad_ramp.py

Usage: remove_quad_ramp.py <infile> <outfile> [--plot]
remove_quad_ramp.py -h | --help

Options:
-h | --help             Show this screen
"""

import docopt
import numpy as np
import scipy.optimize as opt
import scipy.linalg as lst
from osgeo import gdal


def open_gdal(path, band=1):
    ds = gdal.Open(path)
    bd = ds.GetRasterBand(band)
    ndv = bd.GetNoDataValue()
    data = bd.ReadAsArray()
    if ndv is not None and ndv != np.nan:
        data[data == ndv] = np.nan
    return data


def save_gdal(path, data, template, ndv=None):
    ds_template = gdal.Open(template)
    ds = gdal.GetDriverByName('GTiff').CreateCopy(path, ds_template)
    band = ds.GetRasterBand(1)
    band_ndv = band.GetNoDataValue()

    if band_ndv is not None:
        ndv = band_ndv
    elif ndv is not None:
        band.SetNoDataValue(ndv)

    if ndv is not None and ndv != np.nan:
        data[data==np.nan] = ndv
    
    band.WriteArray(data)
    ds.FlushCache()


def deramp(data, plot=False):
    """ Derived from clean_raster, Pygdalsar, Simon Daout
    """
    data_max, data_min = np.nanpercentile(data, 99.5), np.nanpercentile(data, 0.5)
    to_clean = np.nonzero(np.logical_or(data < data_min, data > data_max))
    data_clean = data.copy()
    data_clean[to_clean] = np.nan

    # Fetch index
    index = np.nonzero(~np.isnan(data_clean))
    mi = data[index].flatten()
    az = np.asarray(index[0])
    rg = np.asarray(index[1])

    G=np.zeros((len(mi),4))
    G[:,0] = rg**2
    G[:,1] = rg
    G[:,2] = az
    G[:,3] = 1

    x0 = lst.lstsq(G,mi)[0]
    _func = lambda x: np.sum(((np.dot(G,x)-mi))**2)
    _fprime = lambda x: 2*np.dot(G.T, (np.dot(G,x)-mi))
    pars = opt.fmin_slsqp(_func,x0,fprime=_fprime,iter=50,full_output=True,iprint=0)[0]
    a, b, c, d = pars[0], pars[1], pars[2], pars[3]
    if plot:
        print('Remove ramp %f x**2 %f x  + %f y + %f'%(a,b,c,d))

    G = np.zeros((len(data.flatten()), 4))
    ncols = data.shape[1]
    nlines = data.shape[0]
    for i in range(nlines):
        G[i*ncols:(i+1)*ncols,0] = np.arange((ncols))**2
        G[i*ncols:(i+1)*ncols,1] = np.arange((ncols))
        G[i*ncols:(i+1)*ncols,2] = i
    G[:,3] = 1
    # apply ramp correction
    ramp = np.dot(G,pars)
    ramp = ramp.reshape(nlines, ncols)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        raster = data
        p2, p98 = np.nanpercentile(raster, (2, 98))
        plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
        plt.title("data")

        plt.figure()
        raster = ramp
        p2, p98 = np.nanpercentile(raster, (2, 98))
        plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
        plt.title("ramp")

        plt.figure()
        raster = data - ramp
        p2, p98 = np.nanpercentile(raster, (2, 98))
        plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
        plt.title("data deramp")
        plt.show()
    
    return data - ramp


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    infile = arguments["<infile>"]
    outfile = arguments["<outfile>"]
    do_plot = arguments["--plot"]

    raster = open_gdal(infile)
    deramp_raster = deramp(raster, plot=do_plot)
    save_gdal(outfile, deramp_raster, infile)

