#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r_stats.py
--------------
Compute stack statistics on multiple images for each pixel.

Usage: r_stats.py --infile=<infile> --crop=<crop>
r_stats.py -h | --help

Options:
-h | --help             Show this screen
--infile                Path to the file with one image per line
--crop
"""


import numpy as np
from osgeo import gdal
import os
import docopt
import matplotlib.pyplot as plt
from scipy.stats import describe


def open_raster_crop(path, left, right, top, bottom):
    ds = gdal.Open(path)
    bd = ds.GetRasterBand(1)
    ndv = bd.GetNoDataValue()
    array = bd.ReadAsArray()[left:right, top:bottom]
    print(array.shape)
    if ndv is not None and ndv != np.nan:
        array[array == ndv] = np.nan
    return array


def plot_crop(array):
    plt.figure()
    plt.imshow(array)


def get_stats(array):
    print(describe(array, axis=None, nan_policy="omit"))


def semi_var(array):
    (n, m) = np.shape(array)[:2]
    distances = []
    semi_variogram = []
    for i in range(n):
        for j in range(m):
            for x in range(n):
                for y in range(m):
                    if (i, j) != (x, y):  # Exclude same point
                        distance = np.sqrt((i - x)**2 + (j - y)**2)
                        semi_variogram_value = 0.5 * (array[i, j] - array[x, y])**2
                        distances.append(distance)
                        semi_variogram.append(semi_variogram_value)

    # Since distances are in pixels and can be very small, 
    # let's bin them to get a more meaningful plot
    bins = np.linspace(0, np.sqrt(2), 10)
    binned_distances = np.digitize(distances, bins)
    binned_semivariogram = [np.mean([semi_variogram[i] for i, x in enumerate(binned_distances) if x == j]) for j in range(1, len(bins))]

    # Plotting
    plt.figure()
    plt.scatter(bins[1:], binned_semivariogram)
    plt.xlabel('Distance')
    plt.ylabel('Semi-Variogram')
    plt.title('Semi-Variogram of 2D Grid')


if __name__ == "__main__":
    args = docopt.docopt(__doc__)

    infile = args["--infile"]
    crop = args["--crop"]
    crop = [int(k) for k in crop.split(",")]

    data = open_raster_crop(infile, crop[0], crop[1], crop[2], crop[3])
    get_stats(data)
    plot_crop(data)
    plt.show()
    # semi_var(data)
    plt.show()
