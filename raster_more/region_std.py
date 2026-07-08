#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
region_std.py
-------------
Compute the std in regions of some rasters

For correct log output, raster filename must be like */YYYYMMDD_YYYYMMDD/*

Usage: region_std.py --pattern=<pattern> [<regionfile>] [--ndv=<ndv>] [--band=<band>] [--outlog] [--pos=<pos>]

Options:
-h --help               Show this screen.
<file-pattern>          Bash like pattern to the file locations (i.e ./*/stereo-F.tif)
<region-file>           Path to a file containing all regions to use, one region (xmin xmax ymin ymax) per line
--ndv                   Additionnal ndv to use
--band                  Band to use for all rasters
"""

import docopt
import numpy as np
from osgeo import gdal
from glob import glob
from tqdm import tqdm
gdal.UseExceptions()


def compute_std_region(infile, region, band=None, ndv=None):
    """ Open the raster over a specified region, handle ndv values, 
    then return the spatial standard deviation
    """
    band = band if band is not None else 1
    ds = gdal.Open(infile)
    bd = ds.GetRasterBand(band)
    data = bd.ReadAsArray(region[0], region[2], region[1] - region[0], region[3] - region[2])
    rndv = bd.GetNoDataValue()
    if rndv is not None and rndv != np.nan:
        data[data==rndv] = np.nan
    if ndv is not None and ndv != np.nan:
        data[data==ndv] = np.nan
    # print(data.shape[0] * data.shape[1], "\t total pixels")
    # print(np.sum(~np.isnan(data)), "\t valid pixels")

    return np.nanstd(data)


# Source - https://stackoverflow.com/a/1884277
# Posted by Todd Gamblin, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-23, License - CC BY-SA 4.0

def find_nth(haystack: str, needle: str, n: int) -> int:
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start



def save_log(file, raster, value, pos):
    """ Write std results to file for each raster for a specific region
    """
    content = ""
    for k in range(len(raster)):
        # Hardcoded: raster file name must be smth like */date1_date2/*
        if pos is None:
            dates = raster[k].split('/')[-2].split('_')
        else:
            split_id = find_nth(raster[k], '_', pos)
            dates = [raster[k][split_id-8: split_id], raster[k][split_id+1: split_id+9]]
        content += " ".join([str(k + 1), dates[0], dates[1], str(value[k])]) + "\n"
    with open(file, 'w') as outfile:
        outfile.write(content)
    print("Saved:", file)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    pattern = arguments["--pattern"]
    region_file = arguments["<regionfile>"]
    ndv = float(arguments["--ndv"]) if arguments["--ndv"] is not None else None
    band = int(arguments["--band"]) if arguments["--band"] is not None else None
    outlog = arguments["--outlog"]
    pos = arguments["--pos"]
    pos = int(pos) if pos is not None else None

    rasters = glob(pattern)
    print("Number of selected rasters:", len(rasters))

    regions = []
    if region_file is not None:
        with open(region_file, 'r') as infile:
            for l in infile:
                regions.append([int(k) for k in l.replace('\t', ' ').split(' ')])
    else:
        ds = gdal.Open(rasters[0])
        regions = [[0, ds.RasterXSize, 0, ds.RasterYSize]]
        del ds

    stds_reg = []
    for i, re in enumerate(regions):
        print(">> Region ({}):".format(i), re)
        stds = []
        for ra in tqdm(rasters):
            std = compute_std_region(ra, re, band, ndv)
            stds.append(std)
            # print(ra, ":", std)
        stds_reg.append(stds)
    
    if outlog:
        # content = ""
        # Save one file per region (same format as RMSinterfero)
        for i, re in enumerate(stds_reg):
            save_log("log_region_" + "_".join([str(l) for l in regions[i]]) + ".txt", rasters, re, pos)
            # content += "\n\nRegion " + " ".join([str(l) for l in regions[i]]) + "\n"
            # for j, ra in enumerate(re):
            #     content += rasters[j] + "\t\t" + str(ra) + "\n"
        # with open(outlog, 'w') as outfile:
        #     outfile.write(content)
        # print("Saved:", outlog)
