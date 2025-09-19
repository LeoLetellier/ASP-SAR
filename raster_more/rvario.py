#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rvario.py
-------------
Compute a variogram for a GDAL raster

Usage: rvario.py <infile> [--band=<band> --ndv=<ndv>]


Options:
-h --help               Show this screen.
<infile>                Raster to be displayed
"""

import docopt
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import rasterio.windows
from scipy.fft import fft2, fftshift
gdal.UseExceptions()
from skgstat import Variogram
import rasterio
import pandas as pd


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


def raster_vario(file, band=1, supp_ndv=None):
    '''GPT generated
    '''
    with rasterio.open(file) as src:
        # band1 = src.read(band, window=rasterio.windows.Window(2500, 3500, 1000, 1000))  # Read the first band
        band1 = src.read(band)  # Read the first band
        transform = src.transform

    # Step 2: Create a DataFrame with coordinates and values
    rows, cols = np.indices(band1.shape)
    x_coords, y_coords = rasterio.transform.xy(transform, rows, cols)

    # Flatten the arrays
    x_coords = np.array(x_coords).flatten()
    y_coords = np.array(y_coords).flatten()
    values = band1.flatten()

    # Create a DataFrame
    data = pd.DataFrame({'x': x_coords, 'y': y_coords, 'value': values})

    # Step 3: Remove NaN values
    if supp_ndv is not None and supp_ndv != np.nan:
        data[data == supp_ndv] = np.nan
    data = data.dropna()

    # Step 4: Calculate the variogram
    # Create a Variogram object
    # v = Variogram(data[['x', 'y']].values, data['value'].values, model='spherical', use_nugget=True, samples=10000, maxlag='median', n_lags=20)
    print(data[['x', 'y']].values)
    v = Variogram(data[['x', 'y']].values, data['value'].values, model='spherical', use_nugget=True, samples=10000, maxlag=400, n_lags=25)
    return v


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    infile = arguments["<infile>"]
    band = arguments["--band"]
    band = 1 if band is None else int(band)
    ndv = arguments["--ndv"]
    ndv = None if ndv is None else float(ndv)

    data = open_gdal(infile, band, ndv)

    # fft_data = fft2(data)
    # power_spectrum = np.abs(fft_data)**2
    # power_spectrum_shifted = fftshift(power_spectrum)
    # noise_level = np.mean(power_spectrum[power_spectrum > np.percentile(power_spectrum, 90)])

    # gt = gdal.Open(infile).GetGeoTransform()

    # freq_y = np.fft.fftfreq(data.shape[0], d=gt[5])
    # freq_x = np.fft.fftfreq(data.shape[1], d=gt[1])
    # freq_y_shifted = fftshift(freq_y)
    # freq_x_shifted = fftshift(freq_x)

    # print(noise_level)
    # # plt.imshow(fft_data)
    # # plt.figure()
    # # plt.imshow(power_spectrum)
    # plt.imshow(
    # np.log1p(power_spectrum_shifted),
    #     cmap='inferno',
    #     extent=[freq_x_shifted.min(), freq_x_shifted.max(),
    #             freq_y_shifted.min(), freq_y_shifted.max()]
    # )
    # plt.show()

    v = raster_vario(infile, band)
    v.plot()
    print(v.describe())
    print(v.model_deviations())
    print(v.parameters)
    plt.show()


