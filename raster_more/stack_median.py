#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stack_median.py
------------

Usage: stack_median.py <rasters>... --outfile=<outfile> [--no-shell] [--no-nodata] [--expect-nans]

Options:
-h | --help         Show this screen
<rasters>           All rasters names participating to the median, space separated
--outfile           Output file name
--no-shell          Don't print the running command
--no-nodata         Don't associate a nodata value to the resulting raster
--expect-nans       Use numpy.nanmedian instead of numpy.median when computing the stack
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
    no_shell = arguments["--no-shell"]
    no_nodata = arguments["--no-nodata"]
    expect_nans = arguments["--expect-nans"]
    
    raster_entry = ""
    letters = []
    for i, r in enumerate(rasters):
        letter = chr(ord('A') + i)
        letters.append(letter)
        raster_entry += " -" + letter + " " + r
    
    calc = "numpy.median([{}], axis=0)".format(", ".join(letters))
    if expect_nans:
        calc = "numpy.nanmedian([{}], axis=0)".format(", ".join(letters))
    
    cmd = '''gdal_calc{} --calc="{}" --outfile={}'''.format(
        raster_entry,
        calc,
        outfile
    )

    if no_nodata:
        cmd = '''gdal_calc{} --calc="{}" --outfile={} --NoDataValue=none'''.format(
            raster_entry,
            calc,
            outfile
        )

    if not no_shell:
        print(">>", cmd)
    sh(cmd)

