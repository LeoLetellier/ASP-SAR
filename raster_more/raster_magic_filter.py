#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raster_magic_filter.py
--------------
Compute a median filter on a one band raster, weighted by the standard deviation with other images.

Usage: raster_median_filter.py --infile=<infile>  [--kernel=<kernel>] [--std-threshold=<std-threshold>] [--weighted] [--lee]
raster_median_filter.py -h | --help

Options:
-h | --help             Show this screen
--infile                Path to the text file listing all images, first is the one filtered
--kernel                Size of the filter kernel (can be one nb "3" or array like "2 3")
--std-threshold         Threshold on the std value
--weighted              Whether to used weighting or only threshold
--lee                   Use Lee filter instead. Override --weighted if set.
"""


import numpy as np
from osgeo import gdal
from scipy.signal import medfilt2d
import os
import docopt
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


class GeoTiff:
    def __init__(self):
        self.xsize = None
        self.ysize = None
        self.data = None
        self.proj = None
        self.geotr = None
        self.ndv = None

    def open(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ds = gdal.OpenEx(path, allowed_drivers=["GTiff"])
        band1 = ds.GetRasterBand(1)
        self.ndv = band1.GetNoDataValue()
        self.data = band1.ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)
        if self.ndv and self.ndv != np.nan:
            self.data[self.data == self.ndv] = np.nan
        self.xsize = ds.RasterXSize
        self.ysize = ds.RasterYSize
        self.geotr = ds.GetGeoTransform()
        self.proj = ds.GetProjection()

        return self

    def save(self, path):
        drv = gdal.GetDriverByName('GTiff')
        dst_ds = drv.Create(path, self.xsize, self.ysize, 1, gdal.GDT_Float32)
        dst_band = dst_ds.GetRasterBand(1)
        dst_ds.SetGeoTransform(self.geotr)
        dst_ds.SetProjection(self.proj)
        if self.ndv:
            self.data[self.data == np.nan] = self.ndv
        dst_band.WriteArray(self.data)
        dst_ds.FlushCache()
    
    def replicate_empty(self):
        rep = GeoTiff()
        rep.xsize = self.xsize
        rep.ysize = self.ysize
        rep.ndv = self.ndv
        rep.proj = self.proj
        rep.geotr = self.geotr

        return rep


def stack_median_std(imgs, img):
    mean = np.zeros(shape=img.shape)

    for i in imgs:
        mean += i

    mean = mean / len(imgs)
    std = np.square(img - mean)

    return mean, std


def weighted_median_filter(img, std, threshold, kernel):
    masked_in = std <= threshold
    masked_out = std > threshold

    weights = np.zeros(shape=std.shape)
    weights[masked_out] = np.square(std[masked_out] - threshold)
    # std[masked_out] contains values between threshold+ to +inf
    # std[masked_out] - threshold between 0+ to inf+

    filtered = medfilt2d(img, kernel)

    weighted_filtered = filtered * (1 - weights) + img * weights
    return weighted_filtered


def threshold_median_filter(img, std, threshold, kernel):
    # masked_img = img[std >= threshold]
    masked_img = img
    masked_img[std > threshold] = np.nan
    filtered_img = medfilt2d(masked_img, kernel)
    max_it = 5
    it = 0
    while np.any(np.isnan(filtered_img)) and it < max_it:
        filtered_img = medfilt2d(filtered_img)
        it += 1
    return filtered_img


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    infile = arguments["--infile"]
    files = []
    with open(infile, "r") as f:
        for line in f:
            files.append(line.strip())
    
    kernel = [int(k) for k in arguments["--kernel"].split(" ")] if arguments["--kernel"] else 3

    std_threshold = arguments["--std-threshold"]

    do_weighted = arguments["--weighted"]
    do_lee = arguments["--lee"]

    imgs = []
    for i in files:
        imgs.append(GeoTiff().open(i))
    
    im = imgs[0].data # interest image

    mean, std = stack_median_std([i.data for i in imgs], im)

    std_gt = imgs[0].replicate_empty()
    std_gt.data = std
    std_gt.save(os.path.splitext(files[0])[0] + "_std.tif")

    if do_lee:
        res = lee_filter(im, 3)
    elif do_weighted:
        res = weighted_median_filter(im, std, float(std_threshold), kernel)
    else:
        res = threshold_median_filter(im, std, float(std_threshold), kernel)
    
    filtered = imgs[0].replicate_empty()
    filtered.data = res

    outfile = os.path.splitext(files[0])[0] + "_magicfiltered.tif"
    filtered.save(outfile)

    print('Saved as:', outfile)
