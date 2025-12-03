#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_gif.py

Usage: make_gif.py <infile> <outfile>


Options:
-h --help               Show this screen.
<infile>                Raster to be displayed
<outfile>
"""

import docopt
from PIL import Image


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)

    infile = arguments["<infile>"]
    outfile = arguments["<outfile>"]

    infiles = infile.split(",")

    imgs = []
    for f in infiles:
        img = Image.open(f)
        imgs.append(img)
    
    imgs[0].save(outfile, save_all=True, append_images=imgs[1:], duration=500, loop=0, optimize=False)
