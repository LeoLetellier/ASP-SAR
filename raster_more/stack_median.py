#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stack_median.py
------------

Usage: stack_median.py <rasters>... --outfile=<outfile>

Options:
-h | --help         Show this screen
"""
import docopt
import os, sys, subprocess


def sh(cmd: str, shell: bool = True):
    return subprocess.run(
        cmd,
        shell=shell,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        env=os.environ,
    )


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    rasters = arguments["<rasters>"]
    outfile = arguments["--outfile"]
    
    raster_entry = ""
    letters = []
    for i, r in enumerate(rasters):
        letter = chr(ord('A') + i)
        letters.append(letter)
        raster_entry += " -" + letter + " " + r
    
    calc = "numpy.median([{}], axis=0)".format(", ".join(letters))

    cmd = '''gdal_calc{} --calc="{}" --outfile={}'''.format(
        raster_entry,
        calc,
        outfile
    )

    print(">>", cmd)
    sh(cmd)

