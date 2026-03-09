#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remove_topo_ramp.py

Usage: remove_topo_ramp.py <infile> <outfile> [--dem=<dem>] [--plot] [--band=<band>] [--chunk-size=<chunk-size>] [--overwrite] [--ovr-ratio=<ovr-ratio>] [--ndv=<ndv>] [--inv-iter=<inv-iter>]
remove_topo_ramp.py -h | --help

Options:
-h | --help             Show this screen
"""

import docopt
import numpy as np
import scipy.optimize as opt
import scipy.linalg as lst
from osgeo import gdal
import os
from tqdm import tqdm

# import matplotlib.pyplot as plt

gdal.UseExceptions()


def gdal_raster_size(path):
    ds = gdal.Open(path)
    return ds.RasterXSize, ds.RasterYSize, ds.RasterCount


def open_gdal(path, band=1, chunk=None):
    ds = gdal.Open(path)
    bd = ds.GetRasterBand(band)
    ndv = bd.GetNoDataValue()
    if chunk is not None:
        data = bd.ReadAsArray(chunk[0], chunk[1], chunk[2], chunk[3])
    else:
        data = bd.ReadAsArray()
    if ndv is not None and ndv != np.nan:
        data[data == ndv] = np.nan
    return data


def open_gdal_reduced(path, band=1, size=None, resample_alg=gdal.GRIORA_Bilinear):
    ds = gdal.Open(path)
    bd = ds.GetRasterBand(band)
    ndv = bd.GetNoDataValue()
    data = bd.ReadAsArray(buf_xsize=size[0], buf_ysize=size[1], resample_alg=resample_alg)
    if ndv is not None and ndv != np.nan:
        data[data == ndv] = np.nan
    return data


def create_gdal_from_template(path, template):
    """ Initialize a new raster with same config as an existing one (proj and geotransform) using gdal create copy
    """
    ds_template = gdal.Open(template)
    ds = gdal.GetDriverByName('GTiff').CreateCopy(path, ds_template)
    return ds


def write_gdal_chunk(path, data, offset, ndv=None):
    ds = gdal.Open(path, gdal.GA_Update)
    band = ds.GetRasterBand(1)
    band_ndv = band.GetNoDataValue()

    if band_ndv is not None:
        ndv = band_ndv
    elif ndv is not None:
        band.SetNoDataValue(ndv)
    
    if ndv is not None and ndv != np.nan:
        data[data==np.nan] = ndv
    
    band.WriteArray(data, xoff=offset[0], yoff=offset[1])
    ds.FlushCache()


def define_chunks(dimensions, target_size):
    """ Chunks are squared divisions of a grid each defined by [xoff, yoff, xsize, ysize]
    """
    chunks = []
    # Number of full sized chunks
    full_x = dimensions[0] // target_size[0]
    full_y = dimensions[1] // target_size[1]
    # Residuals for padding chunks
    trunc_x = dimensions[0] % target_size[0]
    trunc_y = dimensions[1] % target_size[1]

    # Full sized chunks
    for kx in range(full_x):
        for ky in range(full_y):
            chunks.append([kx * target_size[0], ky * target_size[1], target_size[0], target_size[1]])
    
    # Pad in x
    if trunc_x > 0:
        for ky in range(full_y):
            chunks.append([full_x * target_size[0], ky * target_size[1], trunc_x, target_size[1]])

    # Pad in y
    if trunc_y > 0:
        for kx in range(full_x):
            chunks.append([kx * target_size[0], full_y * target_size[1], target_size[0], trunc_y])
    
    # Pad the corner
    if trunc_x > 0 and trunc_y > 0:
        chunks.append([full_x * target_size[0], full_y * target_size[1], trunc_x, trunc_y])
    
    return chunks


def save_gdal(path, data, template, ndv=None):
    ds_template = gdal.Open(template)
    ds = gdal.GetDriverByName('GTiff').CreateCopy(path, ds_template)
    band = ds.GetRasterBand(1)
    band_ndv = band.GetNoDataValue()

    if band_ndv is not None:
        ndv = band_ndv
    elif ndv is not None:
        band.SetNoDataValue(ndv)

    if ndv is not None and ndv != np.nan:
        data[data==np.nan] = ndv
    
    band.WriteArray(data)
    ds.FlushCache()


def invers_ramp(data, iter=200):
    """ Derived from clean_raster, Pygdalsar, Simon Daout
    """
    data_max, data_min = np.nanpercentile(data, (98, 2))
    data = np.where(np.logical_or(data < data_min, data > data_max), np.nan, data)

    index = np.nonzero(~np.isnan(data))
    mi = data[index].flatten()
    az = np.asarray(index[0]) / data.shape[0]
    rg = np.asarray(index[1]) / data.shape[1]

    G=np.zeros((len(mi), 3))
    G[:,0] = rg
    G[:,1] = az
    G[:,2] = 1

    x0 = lst.lstsq(G,mi)[0]
    _func = lambda x: np.sum(((np.dot(G, x) - mi))**2)
    _fprime = lambda x: 2 * np.dot(G.T, (np.dot(G, x) - mi))
    pars = opt.fmin_slsqp(_func, x0, fprime=_fprime, iter=iter, full_output=True, iprint=0)[0]

    # print("invers_ramp g 0", G[:, 0])
    # print("invers_ramp g 1", G[:, 1])

    return pars[0], pars[1], pars[2]


def generate_ramp(chunk, dimensions, ramp_coeffs):
    xoff, yoff, xsize, ysize = chunk
    # print("chunk", chunk)
    # print("dimensions", dimensions)

    G = np.zeros((xsize * ysize, 3))
    x_index = np.arange(xoff, xoff + xsize) / dimensions[0]
    y_index = np.arange(yoff, yoff + ysize) / dimensions[1]
    # print("x_index", x_index[0], x_index[-1])
    # print("y_index", y_index[0], y_index[-1])
    
    G[:, 0] = np.tile(x_index, ysize)
    G[:, 1] = np.repeat(y_index, xsize)
    G[:, 2] = 1

    # plt.figure()
    # raster = G[:, 0].reshape((xsize, ysize))
    # p2, p98 = np.nanpercentile(raster, (2, 98))
    # plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
    # plt.colorbar()
    # plt.title("G 0")
    # plt.figure()
    # raster = G[:, 1].reshape((xsize, ysize))
    # p2, p98 = np.nanpercentile(raster, (2, 98))
    # plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
    # plt.colorbar()
    # plt.title("G 1")

    # print(G[:, 0])
    # print(G[:, 1])

    ramp = np.dot(G, ramp_coeffs)
    ramp = ramp.reshape(ysize, xsize)

    return ramp


def invers_ramp_dem(data, dem, iter=200):
    """ Derived from clean_raster, Pygdalsar, Simon Daout
    """
    data_max, data_min = np.nanpercentile(data, (98, 2))
    data = np.where(np.logical_or(data < data_min, data > data_max), np.nan, data)
    
    index = np.nonzero(~(np.isnan(data) | np.isnan(dem)))
    mi = data[index].flatten()
    demi = dem[index].flatten()
    az = np.asarray(index[0]) / data.shape[0]
    rg = np.asarray(index[1]) / data.shape[1]

    G=np.zeros((len(mi), 4))
    G[:,0] = rg
    G[:,1] = az
    G[:,2] = 1
    G[:, 3] = demi

    x0 = lst.lstsq(G,mi)[0]
    _func = lambda x: np.sum(((np.dot(G, x) - mi))**2)
    _fprime = lambda x: 2 * np.dot(G.T, (np.dot(G, x) - mi))
    pars = opt.fmin_slsqp(_func, x0, fprime=_fprime, iter=iter, full_output=True, iprint=0)[0]

    # print("invers_ramp g 0", G[:, 0])
    # print("invers_ramp g 1", G[:, 1])

    return pars


def generate_ramp(chunk, dimensions, ramp_coeffs):
    xoff, yoff, xsize, ysize = chunk
    # print("chunk", chunk)
    # print("dimensions", dimensions)

    G = np.zeros((xsize * ysize, 3))
    x_index = np.arange(xoff, xoff + xsize) / dimensions[0]
    y_index = np.arange(yoff, yoff + ysize) / dimensions[1]
    # print("x_index", x_index[0], x_index[-1])
    # print("y_index", y_index[0], y_index[-1])
    
    G[:, 0] = np.tile(x_index, ysize)
    G[:, 1] = np.repeat(y_index, xsize)
    G[:, 2] = 1

    # plt.figure()
    # raster = G[:, 0].reshape((xsize, ysize))
    # p2, p98 = np.nanpercentile(raster, (2, 98))
    # plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
    # plt.colorbar()
    # plt.title("G 0")
    # plt.figure()
    # raster = G[:, 1].reshape((xsize, ysize))
    # p2, p98 = np.nanpercentile(raster, (2, 98))
    # plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
    # plt.colorbar()
    # plt.title("G 1")

    # print(G[:, 0])
    # print(G[:, 1])

    ramp = np.dot(G, ramp_coeffs)
    ramp = ramp.reshape(ysize, xsize)

    return ramp


def generate_ramp_topo(chunk, dimensions, ramp_coeffs, dem):
    xoff, yoff, xsize, ysize = chunk
    # print("chunk", chunk)
    # print("dimensions", dimensions)

    G = np.zeros((xsize * ysize, 4))
    x_index = np.arange(xoff, xoff + xsize) / dimensions[0]
    y_index = np.arange(yoff, yoff + ysize) / dimensions[1]
    # print("x_index", x_index[0], x_index[-1])
    # print("y_index", y_index[0], y_index[-1])
    
    G[:, 0] = np.tile(x_index, ysize)
    G[:, 1] = np.repeat(y_index, xsize)
    G[:, 2] = 1
    G[:, 3] = dem.flatten()

    # plt.figure()
    # raster = G[:, 0].reshape((xsize, ysize))
    # p2, p98 = np.nanpercentile(raster, (2, 98))
    # plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
    # plt.colorbar()
    # plt.title("G 0")
    # plt.figure()
    # raster = G[:, 1].reshape((xsize, ysize))
    # p2, p98 = np.nanpercentile(raster, (2, 98))
    # plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
    # plt.colorbar()
    # plt.title("G 1")

    # print(G[:, 0])
    # print(G[:, 1])

    ramp = np.dot(G, ramp_coeffs)
    ramp = ramp.reshape(ysize, xsize)

    return ramp


if __name__ == "__main__":
    # from time import time
    # t = time()
    arguments = docopt.docopt(__doc__)
    infile = arguments["<infile>"]
    outfile = arguments["<outfile>"]
    do_plot = arguments["--plot"]
    band = int(arguments["--band"]) if arguments["--band"] is not None else 1
    target_size = [int(s) for s in arguments["--chunk-size"].split(",")] if arguments["--chunk-size"] is not None else [512, 512]
    if len(target_size) == 1:
        target_size = [target_size[0], target_size[0]]
    overwrite = arguments["--overwrite"]
    ratio = float(arguments["--ovr-ratio"]) if arguments["--ovr-ratio"] is not None else 0.2
    add_ndv = arguments["--ndv"]
    inv_iter = int(arguments["--inv-iter"]) if arguments["--inv-iter"] is not None else 100
    dem = arguments["--dem"]

    raster_size = gdal_raster_size(infile)
    chunks = define_chunks([raster_size[0], raster_size[1]], target_size)
    size = [int(ratio * raster_size[0]), int(ratio * raster_size[1])]
    data_reduced = open_gdal_reduced(infile, band=band, size=size)
    if dem is not None:
        dem_reduced = open_gdal_reduced(dem, size=size)
        ramp_coeffs = invers_ramp_dem(data_reduced, dem_reduced, iter=inv_iter)
    else:
        ramp_coeffs = invers_ramp(data_reduced, iter=inv_iter)
    # print("ramp estimation", time() - t)

    if do_plot:
        import matplotlib.pyplot as plt

        print("coeffs:", ramp_coeffs)

        if dem is not None:
            ramp = generate_ramp_topo([0, 0, size[0], size[1]], size, ramp_coeffs, dem_reduced)
        else:
            ramp = generate_ramp([0, 0, size[0], size[1]], size, ramp_coeffs)

        plt.figure()
        raster = data_reduced
        p2, p98 = np.nanpercentile(raster, (2, 98))
        plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
        plt.title("data")

        plt.figure()
        raster = ramp
        p2, p98 = np.nanpercentile(raster, (2, 98))
        plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
        plt.title("ramp")

        plt.figure()
        raster = data_reduced - ramp
        p2, p98 = np.nanpercentile(raster, (2, 98))
        plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
        plt.title("data deramp")
        plt.show()

    del data_reduced
    if os.path.isfile(outfile):
        if not overwrite:
            raise FileExistsError("Raster already exists: {}. Use --overwrite".format(outfile))
    else:
        create_gdal_from_template(outfile, infile)

    # from time import time

    for i, c in enumerate(tqdm(chunks, unit="chunk")):
    #     if i != 4:
    #         continue
        # t = time()
        data = open_gdal(infile, band, chunk=c)
        # print("open time", time() - t)
        # t = time()
        if dem is not None:
            dem_chunk = open_gdal(dem, chunk=c)
            ramp = generate_ramp_topo(c, raster_size, ramp_coeffs, dem_chunk)
        else:
            ramp = generate_ramp(c, raster_size, ramp_coeffs)
        # print("gen ramp time", time() - t)

        # import matplotlib.pyplot as plt

        # plt.figure()
        # raster = data
        # p2, p98 = np.nanpercentile(raster, (2, 98))
        # plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
        # plt.colorbar()
        # plt.title("data")

        # plt.figure()
        # raster = ramp
        # p2, p98 = np.nanpercentile(raster, (2, 98))
        # plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
        # plt.colorbar()
        # plt.title("ramp")

        # plt.figure()
        # raster = data - ramp
        # p2, p98 = np.nanpercentile(raster, (2, 98))
        # plt.imshow(raster, cmap="RdBu_r", vmin=p2, vmax=p98)
        # plt.colorbar()
        # plt.title("data deramp")
        # plt.show()
        

        # t = time()
        write_gdal_chunk(outfile, data - ramp, [c[0], c[1]])
        # print("write chunk time", time() - t)

