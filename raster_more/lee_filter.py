#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lee_filter.py
--------------
Compute a lee filter

Usage: lee_filter.py <infile> [--size=<size>]
lee_filter.py -h | --help

Options:
-h | --help             Show this screen
<infile>                Path to the raster, or text file with one raster name each line
--size                  Size of the window
"""


import numpy as np
from osgeo import gdal
import os
import docopt
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance
from tqdm import tqdm

gdal.UseExceptions()


def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (np.square(img_mean) * img_variance + overall_variance)
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


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    infile = arguments["<infile>"]
    files = []
    if os.path.splitext(infile)[1] == '.txt':
        with open(infile, "r") as f:
            for line in f:
                if line is not None:
                    files.append(line.strip())
    else:
        files = infile.split(',')
    
    kernel = int(arguments["--size"]) if arguments["--size"] is not None else 3

    for f in tqdm(files):
        ds = GeoTiff().open(f)
        ds_out = ds.replicate_empty()
        ds_out.data = lee_filter(ds.data, kernel)
        ds_out.save(os.path.splitext(f)[0] + "_lee_" + str(kernel) + ".tif")
