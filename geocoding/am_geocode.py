#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
am_geocode.py
--------------
Geocode images using AMSTer.

Usage: am_geocode.py --amster-dir=<amster-dir> --param=<param> --infile=<infile> --outdir=<outdir>
am_geocode.py -h | --help

Options:
-h | --help             Show this screen
--amster-dir            Correponding AMSTer working dir image subfolder (which contain i12 folder and corresponds to the same projection)
--param                 Parameter file for AMSTer processing
--infile                Path to the file with one image per line
"""


import os
import sys
import glob
import docopt
from subprocess import run, STDOUT

from osgeo import gdal


def sh(cmd: str, shell=True):
    """
    Launch a shell command

    As shell=True, all single call is made in a separate shell

    # Example

    ````
    sh("ls -l | wc -l")
    ````

    """
    run(cmd, shell=shell, stdout=sys.stdout, stderr=STDOUT, env=os.environ)


def resolve_infile(infile):
    if os.path.splitext(infile)[1] == ".txt":
        with open(infile, 'r') as f:
            paths = f.read().split('\n')
        infile = [r.strip() for r in paths if r.strip()]
    else:
        infile = [infile]
    return infile


# def raster_to_amster(raster, target, band=1):
#     ds = gdal.Open(raster)
#     band = ds.GetRasterBand(band)
#     data = band.ReadAsArray()
#     data.flatten().astype('float32').tofile(target)


def move_to_amster(amster, files):
    dst = os.path.join(amster, "i12", "InSARProducts")
    links = []
    for f in files:
        target = os.path.join(dst, "REGEOC." + os.path.basename(f))
        target = os.path.splitext(target)[0] + ".bil"
        if os.path.exists(target):
            print(">> WARNING: skipped, target already exists:", target)
            links.append(target)
        else:
            cmd = "gdal_translate {} {} -of ENVI".format(f, target)
            sh(cmd)
            links.append(target)
            print("Made ENVI copy from {} to {}".format(f, target))
    return links


def cube_to_amster(amster, file):
    dst = os.path.join(amster, "i12", "InSARProducts")
    links = []
    ds = gdal.Open(file)
    band_nb = ds.RasterCount
    for b in range(1, band_nb + 1):
        target = os.path.join(dst, "REGEOC.b" + str(b) + "_" + os.path.basename(file))
        target = os.path.splitext(target)[0]
        if os.path.exists(target):
            print(">> WARNING: skipped, target already exists:", target)
            links.append(target)
        else:
            cmd = "gdal_translate {} {} -of ENVI -b {}".format(f, target, b)
            sh(cmd)
            links.append(target)
            print("Made ENVI copy from {} to {}".format(f, target))
    return links


def launch_geocode(amster, param):
    cmd = "cd {} ; ReGeocode_AmpliSeries.sh {}".format(amster, param)
    print("Launch geocoding:", cmd)
    sh(cmd)


def get_back_geocoded(amster, outdir, infile):
    if os.path.exists(outdir):
        print(">> Warning: output dir already exists. Proceed with care")
    else:
        os.mkdir(outdir)
    
    if type(infile) is str:
        print("infile is str")
        infile = ["b{}_{}".format(b, os.path.splitext(os.path.basename(infile))[0]) for b in range(1, gdal.Open(infile).RasterCount + 1)]
        print(infile)

    print(os.path.join(amster, "i12", "GeoProjection", "REGEOC." + os.path.splitext(os.path.basename(f))[0]))

    geocoded_envi = [glob.glob(os.path.join(amster, "i12", "GeoProjection", "REGEOC." + os.path.splitext(os.path.basename(f))[0]) + ".*.bil")[0] for f in infile]
    geocoded_hdr = [os.path.splitext(f)[0] + ".hdr" for f in geocoded_envi]
    print("Retrieve geocoded results:")
    print(geocoded_envi)
    print(geocoded_hdr)
    for k in range(len(geocoded_envi)):
        cmd = "mv {} {} ; mv {} {}".format(
            geocoded_envi[k],
            os.path.join(outdir, os.path.basename(geocoded_envi[k])),
            geocoded_hdr[k],
            os.path.join(outdir, os.path.basename(geocoded_hdr[k])),
        )
        print("Move", cmd)
        sh(cmd)


def clean_amster(amster, infile, links):
    print("Cleaning amster dir")
    for l in links:
        print("Removing:", l)
        os.remove(l)
    dir = os.path.join(amster, "i12", "GeoProjection")

    if type(infile) is str:
        infile = ["b{}_{}".format(b, os.path.splitext(os.path.basename(infile))[0]) for b in range(1, gdal.Open(infile).RasterCount + 1)]
    
    for f in infile:
        ras = glob.glob(os.path.join(dir, "REGEOC." + os.path.splitext(os.path.basename(f))[0]) + ".*.ras")[0]
        sh = glob.glob(os.path.join(dir, "REGEOC." + os.path.splitext(os.path.basename(f))[0]) + ".*.ras.sh")[0]
        print("Removing:", ras)
        print("Removing:", sh)
        os.remove(ras)
        os.remove(sh)


if __name__ == "__main__":
    args = docopt.docopt(__doc__)

    infile = args["--infile"]
    amster = args["--amster-dir"]
    param = args["--param"]
    outdir = args["--outdir"]
    merge = False

    infile = resolve_infile(infile)
    print("Processing files:", infile)
    for f in infile:
        # check if can be opened by gdal
        gdal.Open(f)

    if len(infile) == 1 and gdal.Open(infile[0]).RasterCount > 1:
        print(">> Processing cube")
        merge = True
        infile = infile[0]
        links = cube_to_amster(amster, infile)
    
    else:
        links = move_to_amster(amster, infile)

    launch_geocode(amster, param)

    get_back_geocoded(amster, outdir, infile)

    clean_amster(amster, infile, links)

    # if merge:
    #     cmd = "gdal_merge -o geoc.{} -of ENVI {}".format(
    #         os.path.splitext(os.path.basename(infile))[0] + ".bil",
    #         " ".join()
    #     )

    print("done")
