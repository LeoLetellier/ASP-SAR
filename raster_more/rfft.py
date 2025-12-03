#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rfft.py
-------------
Compute a fft 2d for a GDAL raster

Usage: rfft.py <infile> [--band=<band> --ndv=<ndv>] [--dx=<dx>] [--dy=<dy>] [--cut=<cut>]


Options:
-h --help               Show this screen.
<infile>                Raster to be displayed
"""

import docopt
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
gdal.UseExceptions()
from skimage.filters import butterworth


def open_gdal(file, band, supp_ndv=None):
    ds = gdal.Open(file)
    band = ds.GetRasterBand(band)
    ndv = band.GetNoDataValue()
    data = band.ReadAsArray()
    if ndv is not None and ndv != np.nan:
        data[data==ndv] = np.nan
    if supp_ndv is not None and supp_ndv != np.nan:
        data[data==supp_ndv] = np.nan
    return data


def fft_spectrum(data):
    data = np.nan_to_num(data, nan=np.nanmean(data))
    fft = fft2(data)
    power_spectrum = np.abs(fft) ** 2
    return fftshift(power_spectrum)


def plot_spectrum(data, dx=1, dy=1):
    spectrum = fft_spectrum(data)
    fx = fftshift(np.fft.fftfreq(data.shape[1], dx))
    fy = fftshift(np.fft.fftfreq(data.shape[0], dy))
    plt.figure()
    plt.imshow(np.log1p(spectrum), cmap='magma', extent=[
        fx.min(), fx.max(),
        fy.min(), fy.max(),
    ])
    plt.colorbar()
    plt.title("Frequency Spectrum (FFT2)")


def plot_data_freq(data, cut, dx=1, dy=1, highpass=False):
    data = np.nan_to_num(data, nan=np.nanmean(data))

    fs = 1 / np.sqrt(dx**2 + dy**2)
    return butterworth(data, cut, high_pass=highpass)


    fft = fftshift(fft2(data))
    fx = fftshift(np.fft.fftfreq(data.shape[1], dx))
    fy = fftshift(np.fft.fftfreq(data.shape[0], dy))
    FX, FY = np.meshgrid(fy, fx, indexing='ij')
    R = np.sqrt(FX**2 + FY**2)
    if lowpass:
        mask = R <= cut
    else:
        mask = R > cut
    data_cut = np.real(ifft2(fftshift(fft * mask)))
    return data_cut


def plot_img(data):
    plt.figure()
    plt.imshow(data, cmap='RdBu_r', vmin=-5, vmax=5)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    infile = arguments["<infile>"]
    band = arguments["--band"]
    band = 1 if band is None else int(band)
    ndv = arguments["--ndv"]
    ndv = None if ndv is None else float(ndv)
    dx = arguments["--dx"]
    dy = arguments["--dy"]
    dx = 1 if dx is None else float(dx)
    dy = 1 if dy is None else float(dy)
    cut = arguments["--cut"]
    cut = None if cut is None else float(cut)

    data = open_gdal(infile, band, ndv)

    plot_spectrum(data, dx, dy)

    data_low = plot_data_freq(data, cut, dx, dy, highpass=False)
    data_high = plot_data_freq(data, cut, dx, dy, highpass=True)

    plot_img(data_low)
    plt.title("data low pass")
    plot_img(data_high)
    plt.title("data high pass")
    plot_img(data_low + data_high)

    plt.show()
