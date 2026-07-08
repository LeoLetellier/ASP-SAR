#!/usr/bin/env python3
# -*- coding: utf-8 -*-
############################################
# Author        : Leo L
############################################


"""\
offset_misclosure.py
-------------
Compute the amplitude offset misclosure over a specific dataset.


Usage: offset_misclosure.py <pattern> --list_pairs=<l> --outfile=<o>
offset_misclosure.py  -h | --help

Options:
-h --help           Show this screen.
<pattern>           Pattern to the files to use
--list_pairs        List of the pairs (AMSTer file, d1|d2|bp|bt)
--outfile           Path to outfile containing amplitude offset misclosure
"""

import docopt
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from glob import glob
import re

def extract_dates_from_path(file_path):
    # Extract YYYYMMDD_YYYYMMDD from the file path
    match = re.search(r'(\d{8})_(\d{8})', file_path)
    if match:
        return (match.group(1), match.group(2))
    return None

def determine_triplet_pairing(files, pairs):
    # Map each pair to its file path
    pair_to_file = {}
    for file_path in files:
        dates = extract_dates_from_path(file_path)
        if dates:
            pair_to_file[dates] = file_path

    # Create a set of pairs for quick lookup
    pair_set = set(pairs)

    triplets = set()

    # Find all valid triplets (d1, d2, d3)
    for d1, d2 in pairs:
        for d3 in [d for (a, d) in pairs if a == d2]:
            if (d1, d3) in pair_set:
                triplets.add((d1, d2, d3))

    # Convert triplets to file paths
    result = []
    for d1, d2, d3 in sorted(triplets):
        path12 = pair_to_file.get((d1, d2))
        path23 = pair_to_file.get((d2, d3))
        path13 = pair_to_file.get((d1, d3))

        if path12 and path23 and path13:
            # Extract dates from paths
            dates12 = extract_dates_from_path(path12)
            dates23 = extract_dates_from_path(path23)
            dates13 = extract_dates_from_path(path13)

            if dates12 and dates23 and dates13:
                # Direct string comparison for YYYYMMDD
                start12, end12 = dates12
                start23, end23 = dates23
                start13, end13 = dates13

                # Check if path13 covers both path12 and path23
                if (start13 <= start12 and end13 >= end12) and (start13 <= start23 and end13 >= end23):
                    result.append((path12, path23, path13))

    return result


def open_gdal(file):
    ds = gdal.Open(file)
    bd = ds.GetRasterBand(1)
    data = bd.ReadAsArray()
    return data


def compute_triplet(pairing, shape):
    misclosure = np.zeros(shape=shape)
    for (f1, f2, f3) in tqdm(pairing):
        misclosure += open_gdal(f1)
        misclosure += open_gdal(f2)
        misclosure -= open_gdal(f3)
    return misclosure


def save_gdal(template, outfile, data):
    ds_template = gdal.Open(template)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.CreateCopy(ds_template, outfile)
    del ds_template
    bd = ds.GetRasterBand(1)
    bd.WriteArray(data)
    ds.FlushCache()


def offset_misclosure(pattern, list_pairs, outfile):
    files = glob(pattern)
    ds = gdal.Open(files[0])
    shape = (ds.RasterYSize, ds.RasterXSize)
    del ds
    pairs = np.loadtxt(list_pairs, skiprows=2, usecols=(1, 2), dtype=str)
    pairing = determine_triplet_pairing(files, pairs)
    misclosure = compute_triplet(pairing, shape)
    save_gdal(files[0], outfile, misclosure)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    offset_misclosure(arguments["<pattern>"], arguments["--list_pairs"], arguments["--outfile"])
