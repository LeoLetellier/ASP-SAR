#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raster_intersect.py

Crop all raster to the common intersection

Usage: raster_intersect.py <input>... --suffix=<suffix>
raster_intersect.py -h | --help

Options:
-h | --help             Show this screen
"""

from osgeo import gdal
import docopt
import os
import sys
import subprocess

gdal.UseExceptions()

def sh(cmd: str, shell: bool = True):
    return subprocess.run(
        cmd,
        shell=shell,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        env=os.environ,
    )


def get_bbox(file):
    ds = gdal.Open(file)
    gt = ds.GetGeoTransform()
    ulx = gt[0]
    bry = gt[3]
    resx = gt[1]
    resy = gt[5]
    brx = ulx + resx * ds.RasterXSize
    uly = bry + resy * ds.RasterYSize
    return [ulx, uly, brx, bry], [resx, resy]


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    input = arguments["<input>"]
    suffix = arguments["--suffix"]

    bbox_list = []
    res_list = []
    for i in input:
        bbox, res = get_bbox(i)
        bbox_list.append(bbox)
        res_list.append(res)
    
    min_bbox = bbox_list[0]
    for i in range(1, len(input)):
        if res_list[i][0] != res_list[0][0] or res_list[i][1] != res_list[0][1]:
            raise ValueError('Inconsistent resolution')
        if min_bbox[0] > bbox_list[i][0]:
            min_bbox[0] = bbox_list[i][0]
        if min_bbox[1] < bbox_list[i][1]:
            min_bbox[1] = bbox_list[i][1]
        if min_bbox[2] > bbox_list[i][2]:
            min_bbox[2] = bbox_list[i][2]
        if min_bbox[3] < bbox_list[i][3]:
            min_bbox[3] = bbox_list[i][3]
    
    if min_bbox[1] < min_bbox[0] or min_bbox[3] < min_bbox[2]:
        raise ValueError("Not all raster intersect")
    
    print("Found common BBOX: " + " ".join([str(k) for k in min_bbox]))
    print("Running gdalwarp for all rasters")
    
    for i in input:
        cmd = "gdalwarp -te {} {} {} -multi -wo NUM_THREADS=4".format(
            " ".join([str(k) for k in min_bbox]),
            i,
            os.path.splitext(i)[0] + "-" + suffix + ".tif"
        )
        print(">> " + cmd)
        sh(cmd)
