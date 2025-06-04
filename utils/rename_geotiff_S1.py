#!/usr/bin/env python
"""
rename_geotiff_S1.py
-----------------
Small script to rename converted S1 geotiff files in format for processing.

Usage: rename_geotiff_S1.py --data=<path>
rename_geotiff_S1.py -h | --help

Options:
-h | --help             Show this screen
--data                  Path to processing directory with GEOTIFF directory

"""

##########
# IMPORT #
##########

import os
from pathlib import Path
from math import *
import docopt
import glob

########
# MAIN #
########

arguments = docopt.docopt(__doc__)

data_path = arguments['--data']

geotiff_dir = os.path.join(data_path, 'GEOTIFF')
# geotiff_original_dir = os.path.join(data_path, 'GEOTIFF_ORIGINAL')

# os.rename(geotiff_dir, geotiff_original_dir)

# Path(geotiff_dir).mkdir(parents=True, exist_ok=True)

# # run through GEOTIFF_ORIGINAL - link all the data to GEOTIFF in correct format
# for f in os.listdir(geotiff_original_dir):
#     if('mod_log.tif' in f):
#         print(f.split('_')[2])
#         os.symlink(os.path.join(geotiff_original_dir, f), os.path.join(geotiff_dir, '{}.VV.mod_log.tif'.format(f.split('_')[2])))
print("search for", geotiff_dir + "/S1*.mod_log.tif")
files = glob.glob(geotiff_dir + "/S1*.tif")
# print("found:", files)
for f in files:
    name = os.path.basename(f)
    _, _, date, ext1, ext2 = name.split("_")
    new_name = date + ext1[1:] + "_" + ext2
    os.symlink(name, os.path.join(geotiff_dir, new_name))