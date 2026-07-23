#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_divergence.py
------------

Usage: compute_divergence.py <infile_h> <infile_v> --dem=<dem> --outfile=<o> --dem-sigma=<ds> [--band=<b>]

Options:
-h | --help         Show this screen
"""

import docopt
import numpy as np
import rasterio as rio
from scipy.ndimage import gaussian_filter, correlate
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin


def open_data(file, band=1):
    with rio.open(file) as ds:
        data = ds.read(band)
        nodata = ds.nodata
        if nodata is not None:
            data = np.where(np.isnan(data) | ~np.isfinite(data), np.nan, data)
    return data


def get_common_grid(files, target_resolution=None):
    """
    Compute the intersection extent + a common resolution across rasters.
    target_resolution: if None, uses the coarsest (largest) pixel size found.
    Returns: (crs, transform, width, height)
    """
    bounds_list, res_list = [], []
    ref_crs = None

    for f in files:
        with rio.open(f) as ds:
            if ref_crs is None:
                ref_crs = ds.crs
            b = ds.bounds
            if ds.crs != ref_crs:
                b = transform_bounds(ds.crs, ref_crs, *b)
            bounds_list.append(b)
            res_list.append(max(abs(ds.res[0]), abs(ds.res[1])))

    left = max(b[0] for b in bounds_list)
    bottom = max(b[1] for b in bounds_list)
    right = min(b[2] for b in bounds_list)
    top = min(b[3] for b in bounds_list)

    if left >= right or bottom >= top:
        raise ValueError("Rasters do not overlap.")

    res = target_resolution or max(res_list)
    width = int(np.floor((right - left) / res))
    height = int(np.floor((top - bottom) / res))
    transform = from_origin(left, top, res, res)

    return ref_crs, transform, width, height


def align_to_grid(file, crs, transform, width, height, band=1,
                   resampling=Resampling.bilinear):
    """Reproject/resample one band of `file` onto the common grid."""
    with rio.open(file) as src:
        dst = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=resampling,
        )
    return dst


def create_empty_raster(
        out_path, 
        crs,
        transform,
        width,
        height,
        dtype="float32",
        count=1,
        nodata=np.nan,
        driver="GTiff",
        **kwargs):
    with rio.open(
        out_path,
        "w",
        driver=driver,
        width=width,
        height=height,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        **kwargs
    ) as _dst:
        pass


def write_band(array, raster_path, band, dtype=None):
    with rio.open(raster_path, "r+") as dst:
        ndv = dst.nodata
        array = np.where(np.isnan(array) | ~np.isfinite(array), ndv, array)
        dst.write(array.astype(dtype or array.dtype), band)



def nan_gaussian_filter(data, sigma):
    if sigma == 0:
        return data
    mask = np.isnan(data)
    filled = np.where(mask, 0, data)
    weight = np.where(mask, 0, 1.0)
    filled_s = gaussian_filter(filled, sigma)
    weight_s = gaussian_filter(weight, sigma)
    with np.errstate(invalid="ignore"):
        out = filled_s / weight_s
    out[weight_s == 0] = np.nan
    return out



def compute_divergence(infile_h, infile_v, dem, outfile, dem_sigma, band):
    crs, transform, width, height = get_common_grid([infile_h, infile_v, dem])
    raster_h = align_to_grid(infile_h, crs, transform, width, height, band=band)
    raster_v = align_to_grid(infile_v, crs, transform, width, height, band=band)
    dsm = align_to_grid(dem, crs, transform, width, height, band=band)
    dsm_filter = nan_gaussian_filter(dsm, dem_sigma / transform.a)

    dsm_angle = dsm2angle(dsm_filter, transform.a)
    disp_angle = disp2angle(raster_v, raster_h)
    angle_diff = cyclic_diff(dsm_angle, disp_angle, 360)

    create_empty_raster(outfile, crs, transform, width, height)
    write_band(angle_diff, outfile)



def cyclic_diff(a, b, N):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = (b - a) % N
    diff = np.where(diff > N / 2, diff - N, diff)
    return diff


def disp2angle(disp_v, disp_h):
    north_comp = -disp_v
    east_comp  = disp_h

    magnitude = np.hypot(disp_h, disp_v)
    angle = np.degrees(np.arctan2(east_comp, north_comp)) % 360
    return magnitude, angle


def dsm2angle(dsm, resolution):
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]) / (8 * resolution)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]]) / (8 * resolution)

    dzdx = correlate(dsm, kx, mode="nearest")   # + = increasing east
    dzdy = correlate(dsm, ky, mode="nearest")   # + = increasing south (row-index direction)

    # downslope compass bearing, clockwise from north, 0=N 90=E 180=S 270=W
    aspect = np.degrees(np.arctan2(-dzdx, dzdy)) % 360
    aspect[(dzdx == 0) & (dzdy == 0)] = np.nan
    return aspect


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    infile_h = arguments["<infile_h>"]
    infile_v = arguments["<infile_v>"]
    dem = arguments["--dem"]
    outfile = arguments["--outfile"]
    dem_sigma = arguments["--dem-sigma"]
    band = arguments["--band"]
    if band is None:
        band = 1
    else:
        band = int(band)

    compute_divergence(infile_h, infile_v, dem, outfile, dem_sigma, band)
