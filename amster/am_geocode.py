#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
am_geocode.py
--------------
Geocode images using AMSTer.

Usage: am_geocode.py <infile> --amster=<amster> [--params=<params>] --outdir=<outdir>  [--crop=<crop>]
am_geocode.py -h | --help

Options:
-h | --help             Show this screen
--amster-dir            Corresponding AMSTer working dir image subfolder (which contain i12 folder and corresponds to the same projection)
--param                 Parameter file for AMSTer processing
--infile                Path to the file with one image per line
"""


import os
import sys
import glob
import docopt
from subprocess import run, STDOUT
from shutil import copy, rmtree

from osgeo import gdal


def sh(cmd: str, shell=True):
    run(cmd, shell=shell, stdout=sys.stdout, stderr=STDOUT, env=os.environ)


def resolve_infile(infile):
    if os.path.splitext(infile)[1] == ".txt":
        with open(infile, 'r') as f:
            paths = f.read().split('\n')
        infile = [r.strip() for r in paths if r.strip()]
    else:
        infile = [infile]
    return infile


def move_to_amster(amster, files):
    dst = os.path.join(amster, "i12", "InSARProducts")
    links = []
    for f in files:
        target = os.path.join(dst, "REGEOC." + os.path.basename(f))
        if check_gdal(f):
            ds = gdal.Open(f)
            band_nb = ds.RasterCount
            if band_nb == 1:
                target = os.path.splitext(target)[0] + ".bil"
                if os.path.exists(target):
                    print(">> WARNING: skipped, target already exists:", target)
                else:
                    cmd = "gdal_translate {} {} -of ENVI".format(f, target)
                    sh(cmd)
                    print("Made ENVI copy from {} to {}".format(f, target))
                links.append(target)
            else:
                for b in range(1, band_nb + 1):
                    target = os.path.join(dst, "REGEOC.b" + str(b) + "_" + os.path.basename(f) + ".r4")
                    if os.path.exists(target):
                        print(">> WARNING: skipped, target already exists:", target)
                    else:
                        band = ds.GetRasterBand(b)
                        data = band.ReadAsArray()
                        save_real4(target, data)
                        print("Saved band {} at {}".format(b, target))
                    links.append(target)
        else:
            if os.path.exists(target):
                print(">> WARNING: skipped, target already exists:", target)
            else:
                copy(f, target)
                print("Simple copy from {} to {}".format(f, target))
            links.append(target)
    return links


# def cube_to_amster(amster, file, dim=None, crop=None):
#     dst = os.path.join(amster, "i12", "InSARProducts")
#     links = []
#     ds = gdal.Open(file)
#     band_nb = ds.RasterCount
#     for b in range(1, band_nb + 1):
#         target = os.path.join(dst, "REGEOC.b" + str(b) + "_" + os.path.basename(file) + ".r4")
#         if os.path.exists(target):
#             print(">> WARNING: skipped, target already exists:", target)
#         else:
#             band = ds.GetRasterBand(b)
#             data = band.ReadAsArray()
#             save_real4(target, data)
#             print("Saved band {} at {}".format(b, target))
#             # if dim is not None and crop is not None:
#             #     save_unpad_gdal(file, target, dim, b, crop)
#             #     print("unpack {} to {}".format(file, target))
#             # else:
#             #     cmd = "gdal_translate {} {} -of ENVI -b {}".format(file, target, b)
#             #     sh(cmd)
#             #     print("Made ENVI copy from {} to {} (band {})".format(file, target, b))
#         links.append(target)
#     return links


def save_real4(dst, data):
    with open(dst, 'wb') as outfile:
        data = data.flatten().astype('float32').tofile(outfile)


# def open_gdal(src, band):
#     with gdal.Open(src) as ds:
#         bd = ds.GetRasterBand(band)
#         ndv = bd.GetNoDataValue()
#         if ndv is None:
#             ndv = 9999
#         data = bd.ReadAsArray()
#     return data


# def save_unpad_gdal(src, dst, dim=None, band=1, crop=None):
#     with gdal.Open(src) as ds:
#         bd = ds.GetRasterBand(band)
#         ndv = bd.GetNoDataValue()
#         if ndv is None:
#             ndv = 9999
#         data = bd.ReadAsArray()

#         full_data = np.full(shape=(dim[1], dim[0]), fill_value=ndv)
#         full_data[crop[2]:crop[3], crop[0]:crop[1]] = data
#         del data
#     drv = gdal.GetDriverByName('ENVI')
#     ds = drv.Create(dst, dim[0], dim[1], 1, gdal.GDT_Float32)
#     bd = ds.GetRasterBand(1)
#     bd.SetNoDataValue(ndv)
#     bd.WriteArray(full_data)
#     ds.FlushCache()


def launch_geocode(amster, param):
    cmd = "cd {} ; ReGeocode_AmpliSeries.sh {}".format(amster, param)
    print("Launch geocoding:", cmd)
    sh(cmd)


def get_back_geocoded(amster, outdir, links):
    if os.path.exists(outdir):
        print(">> Warning: output dir already exists. Proceed with care")
    else:
        os.mkdir(outdir)

    geocoded_envi = [glob.glob(os.path.join(amster, "i12", "GeoProjection", os.path.basename(l) + ".*.bil"))[0] for l in links]
    geocoded_hdr = [os.path.splitext(f)[0] + ".hdr" for f in geocoded_envi]
    print("Retrieve geocoded results:")
    print(geocoded_envi)
    print(geocoded_hdr)
    get_files = []
    for k in range(len(geocoded_envi)):
        cmd = "mv {} {} ; mv {} {}".format(
            geocoded_envi[k],
            os.path.join(outdir, os.path.basename(geocoded_envi[k])),
            geocoded_hdr[k],
            os.path.join(outdir, os.path.basename(geocoded_hdr[k])),
        )
        print("Move", cmd)
        sh(cmd)
        get_files.append(os.path.join(outdir, os.path.basename(geocoded_envi[k])))
    return get_files


def get_dim(folder):
    insar_param_file = os.path.join(folder, 'i12', 'TextFiles', 'InSARParameters.txt')
    with open(insar_param_file, 'r') as f:
        lines = [''.join(l.strip().split('\t\t')[0]) for l in f.readlines()]
        jump_index = lines.index('/* -5- Interferometric products computation */')
        img_dim = lines[jump_index + 2: jump_index + 4]

        dim = int(img_dim[0].strip()), int(img_dim[1].strip())
    return dim


def clean_amster(amster, infile, links):
    print("Cleaning amster dir")
    for l in links:
        print("Removing:", l)
        os.remove(l)
    dir = os.path.join(amster, "i12", "GeoProjection")

    if type(infile) is str:
        infile = ["b{}_{}".format(b, os.path.splitext(os.path.basename(infile))[0]) for b in range(1, gdal.Open(infile).RasterCount + 1)]
    
    for f in infile:
        ras = glob.glob(os.path.join(dir, "REGEOC." + os.path.splitext(os.path.basename(f))[0]) + ".*.ras")
        sh = glob.glob(os.path.join(dir, "REGEOC." + os.path.splitext(os.path.basename(f))[0]) + ".*.ras.sh")
        if len(ras) != 0:
            print("Removing:", ras[0])
            os.remove(ras[0])
        if len(sh) != 0:
            print("Removing:", sh)
            os.remove(sh[0])


def merge_rasters(raster_list):
    name = raster_list[0][11:-13]
    target = os.path.join(os.path.dirname(raster_list[0]), name + ".bil")

    sh("gdal_merge.py -o {} -separate {}".format(target, " ".join(raster_list)))

    # sh("gdal_translate -of ENVI -b 1 {} {}".format(raster_list[0], target))
    # for r in raster_list[1:]:
    #     sh("gdal_translate -of ENVI -b 1 {} {}".format(r, target))
    # print(f"merged into {target}")

    for r in raster_list:
        rmtree(r)


def check_gdal(file):
    ds = gdal.Open(file)
    return ds is not None


if __name__ == "__main__":
    args = docopt.docopt(__doc__)

    infile = args["<infile>"]
    amster = args["--amster"]
    param = args["--params"]
    outdir = args["--outdir"]
    # crop = args["--crop"]
    merge = False

    if param is None:
        param = glob.glob(os.path.join(amster, "Launch*.txt"))[0]
    
    infile = resolve_infile(infile)
    print("Processing files:", infile)
    # if crop is not None:
    #     crop = [int(c) for c in crop.split(",")[:4]]
    
    # # TODO need clarify this condition
    # if all([not check_gdal(f) for f in infile]):
    links = move_to_amster(amster, infile)
    # else:
    #     infile = infile[0]
    #     dim = get_dim(amster)
    #     print("dim:", dim)
    #     links = cube_to_amster(amster, infile, dim=dim, crop=crop)
        

    launch_geocode(amster, param)

    files = get_back_geocoded(amster, outdir, links)

    clean_amster(amster, infile, links)

    # merge_rasters(files)

    print("done")
    