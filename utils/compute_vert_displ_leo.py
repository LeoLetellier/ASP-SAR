#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_vert_displ.py
--------------
Compute the vertical displacements given horizontal displacements and vertical difference.

Usage: compute_vert_displ.py --we=<path> --ns=<path> --dsm1=<path> --dsm2=<path> --dest=<path> --name=<value>
compute_vert_displ.py -h | --help

Options:
-h | --help             Show this screen
--we                    Path to WE-displacement map (positive towards east)
--sn                    Path to NS-displacement map (positive towards south)
--vert                  Path to vertical difference map
--dsm                   Path to DSM map
--dest                  Path to destination directory
--name                  Name of output file
"""

##########
# IMPORT #
##########

import numpy as np
import os
from osgeo import gdal
import docopt

#############
# FUNCTIONS #
#############

def read_tif(input_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found")
    ds = gdal.OpenEx(input_file, allowed_drivers=['GTiff'])
    ds_band = ds.GetRasterBand(1)
    ndv = ds_band.GetNoDataValue()
    values = ds_band.ReadAsArray()
    if ndv is not None and ndv != np.nan:
        values[values == ndv] = np.nan
    ncols, nlines = ds.RasterYSize, ds.RasterXSize
    proj = ds.GetProjection()
    geotransform = ds.GetGeoTransform()
    return (values, nlines, ncols, proj, geotransform)

def save_to_file(data, output_path, ncol, nrow, proj, geotransform):
    drv = gdal.GetDriverByName('GTiff')
    dst_ds = drv.Create(output_path, ncol, nrow, 1, gdal.GDT_Float32)
    dst_band = dst_ds.GetRasterBand(1)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(proj)
    data[data==np.nan] = -9999
    dst_band.WriteArray(data)
    dst_ds.FlushCache()

########
# MAIN #
########

arguments = docopt.docopt(__doc__)

we_displ_file = arguments['--we']
ns_displ_file = arguments['--ns']
dsm1_file = arguments['--dsm1']
dsm2_file = arguments['--dsm2']
dest_path = arguments['--dest']
name = arguments['--name']

# Default values for optional arguments
filter_size = float(arguments['--filter_size']) if arguments['--filter_size'] else 10.0
angle_threshold = float(arguments['--angle_threshold']) if arguments['--angle_threshold'] else 30.0

# Read data
we_displ = read_tif(we_displ_file)[0]  # positive towards east
ns_displ = read_tif(ns_displ_file)[0] # positive towards south 
dsm1 = read_tif(dsm1_file)[0] # convention: negative when erosion
dsm2, dsm_cols, dsm_lines, dsm_proj, dsm_geotransf = read_tif(dsm2_file)

if not (we_displ.shape == ns_displ.shape == dsm1.shape == dsm2.shape):
    raise ValueError("Dimensions between files do not match")


x_size = dsm_cols
y_size = dsm_lines

# get indices
y_index, x_index = np.meshgrid(np.arange(0, y_size, 1), np.arange(0, x_size, 1), indexing='ij')

# get target index displacement wise
x_search = x_index + we_displ
y_search = y_index + ns_displ

# Get containing index
xf = np.floor(x_search).astype(int)
xc = np.ceil(x_search).astype(int)
yf = np.floor(y_search).astype(int)
yc = np.ceil(y_search).astype(int)
# get proximity to containing index
d1 = np.abs(np.sqrt(2) - np.sqrt(np.square((x_search - xf)) + np.square((y_search - yf))))
d2 = np.abs(np.sqrt(2) - np.sqrt(np.square((x_search - xf)) + np.square((y_search - yc))))
d3 = np.abs(np.sqrt(2) - np.sqrt(np.square((x_search - xc)) + np.square((y_search - yf))))
d4 = np.abs(np.sqrt(2) - np.sqrt(np.square((x_search - xc)) + np.square((y_search - yc))))
sumd = d1 + d2 + d3 + d4

# mask to not try to reach outside of array
mask = (x_search >= x_size - 2) | (y_search >= y_size - 2) | (x_search < 0) | (y_search < 0) | (sumd == 0)

res = np.full(shape=dsm2.shape, fill_value=np.nan)

# compute topo by linear interpolating second dsm at target point
res[~mask] = (d1[~mask] * dsm2[yf[~mask], xf[~mask]] +
                   d2[~mask] * dsm2[yc[~mask], xf[~mask]] +
                   d3[~mask] * dsm2[yf[~mask], xc[~mask]] +
                   d4[~mask] * dsm2[yc[~mask], xc[~mask]]) / sumd[~mask]

# compute diff: old topo at point k - new topo at point targeted by the displacement at k
vert = res - dsm1
vert[mask] = np.nan

save_to_file(vert, dest_path + '/' + name + '.tif', dsm_cols, dsm_lines, dsm_proj, dsm_geotransf)
