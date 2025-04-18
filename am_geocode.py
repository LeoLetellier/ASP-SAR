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


def link_to_amster(amster, files):
    dst = os.path.join(amster, "i12", "InSARProducts")
    links = []
    for f in files:
        target = os.path.join(dst, "REGEOC." + os.path.basename(f))
        if os.path.exists(target):
            print(">> WARNING: skipped, target already exists:", target)
        else:
            os.symlink(f, target)
            links.append(target)
            print("Created link from {} to {}".format(f, target))
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
    
    geocoded_envi = [glob.glob(os.path.join(amster, "i12", "GeoProjection", "REGEOC." + os.path.basename(f)) + ".*.bil")[0] for f in infile]
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
    for f in infile:
        ras = glob.glob(os.path.join(dir, "REGEOC." + os.path.basename(f)) + ".*.ras")[0]
        sh = glob.glob(os.path.join(dir, "REGEOC." + os.path.basename(f)) + ".*.ras.sh")[0]
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

    infile = resolve_infile(infile)
    print("Processing files:", infile)

    links = link_to_amster(amster, infile)

    launch_geocode(amster, param)

    get_back_geocoded(amster, outdir, infile)

    clean_amster(amster, infile, links)

    print("done")
