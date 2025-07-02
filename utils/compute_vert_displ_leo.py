#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_vert_displ.py
--------------
Compute the vertical displacements given horizontal displacements and vertical difference.

Usage: compute_vert_displ.py --we=<path> --ns=<path> --dsm_pre=<path> --dsm_post=<path> --outfile=<outfile>
compute_vert_displ.py -h | --help

Options:
-h | --help             Show this screen
--we                    Path to WE-displacement map (positive towards east)
--ns                    Path to NS-displacement map (positive towards south)
--dsm_pre               Path to first DSM map pre displacement (old)
--dsm_post              Path to second DSM map post displacement (new)
--outfile               Path to output vertical displacement file
"""

import numpy as np
from osgeo import gdal
import docopt


def open_gdal(input_file):
    ds = gdal.Open(input_file)
    band = ds.GetRasterBand(1)
    ndv = band.GetNoDataValue()
    data = band.ReadAsArray()
    if ndv is not None and ndv != np.nan:
        data[data == ndv] = np.nan
    return data


def save_gdal_from_copy(template, path, data, ndv=-9999):
    ds_template = gdal.Open(template)
    ds = gdal.GetDriverByName('GTiff').CreateCopy(path, ds_template)
    band = ds.GetRasterBand(1)
    data[data==np.nan] = ndv
    band.SetNoDataValue(ndv)
    band.WriteArray(data)
    ds.FlushCache()


def dsm_with_offsets(dsm, x_offset, y_offset):
    """Interpolate a DSM at offset positions

    :param dsm: dsm to interpolate from
    :param x_offset: offset to apply at each original x index (in pixel nb)
    :param y_offset: offset to apply at each original y index (in pixel nb)
    """
    x_size = dsm.shape[1]
    y_size = dsm.shape[0]

    # get indices
    y_index, x_index = np.meshgrid(np.arange(0, y_size, 1), np.arange(0, x_size, 1), indexing='ij')

    # get target index displacement wise
    x_search = x_index + x_offset
    y_search = y_index + y_offset

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

    dsm_interp = np.full(shape=dsm_post.shape, fill_value=np.nan)

    # compute topo by linear interpolating second dsm at target point
    dsm_interp[~mask] = (d1[~mask] * dsm_post[yf[~mask], xf[~mask]] +
                    d2[~mask] * dsm_post[yc[~mask], xf[~mask]] +
                    d3[~mask] * dsm_post[yf[~mask], xc[~mask]] +
                    d4[~mask] * dsm_post[yc[~mask], xc[~mask]]) / sumd[~mask]
    return dsm_interp
    

if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    we_displ_file = arguments['--we']
    ns_displ_file = arguments['--ns']
    dsm_pre_file = arguments['--dsm_pre']
    dsm_post_file = arguments['--dsm_post']
    outfile = arguments['--outfile']

    # Read data
    we_displ = open_gdal(we_displ_file)     # positive towards east
    ns_displ = open_gdal(ns_displ_file)     # positive towards south
    dsm_pre = open_gdal(dsm_pre_file)       # positive towards up
    dsm_post = open_gdal(dsm_post_file)     # positive towards up

    if not (we_displ.shape == ns_displ.shape == dsm_pre.shape == dsm_post.shape):
        raise ValueError("Dimensions between files do not match")

    dsm_new_interp = dsm_with_offsets(dsm_post, we_displ, ns_displ)

    # compute diff: new topo at point targeted by the displacement at k - old topo at point k
    vert = dsm_new_interp - dsm_pre

    save_gdal_from_copy(dsm_pre_file, outfile, vert)
